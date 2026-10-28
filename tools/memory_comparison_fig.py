import argparse
from pathlib import Path
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def KIVI_memory(batch, hidden_dim, seq_len, group_size=32, bits=4):
    key_states_quant_trans_bytes = batch * hidden_dim * seq_len * bits / 8
    key_scale_trans_bytes = batch * hidden_dim * seq_len / group_size * 2  # FP16
    key_mn_trans_bytes = batch * hidden_dim * seq_len / group_size * 2

    val_states_quant_trans_bytes = batch * hidden_dim * seq_len * bits / 8
    val_scale_trans_bytes = batch * hidden_dim * seq_len / group_size * 2
    val_mn_trans_bytes = batch * hidden_dim * seq_len / group_size * 2

    quant_bytes = key_states_quant_trans_bytes + val_states_quant_trans_bytes
    meta_bytes = key_scale_trans_bytes + key_mn_trans_bytes + \
        val_scale_trans_bytes + val_mn_trans_bytes

    return quant_bytes, meta_bytes


def MXFP4_memory(batch, hidden_dim, seq_len, group_size=32):
    bits = 4
    key_quant_bytes = batch * hidden_dim * seq_len * bits / 8
    key_scale_factor_bytes = batch * hidden_dim * \
        seq_len / group_size  # 8-bit common scale

    val_quant_bytes = batch * hidden_dim * seq_len * bits / 8
    val_scale_factor_bytes = batch * hidden_dim * \
        seq_len / group_size  # 8-bit common scale

    quant_bytes = key_quant_bytes + val_quant_bytes
    meta_bytes = key_scale_factor_bytes + val_scale_factor_bytes
    return quant_bytes, meta_bytes


def ADKV_memory(batch, hidden_dim, seq_len, bits=4):
    # To evaluate checkpoint mode for ADKV, set ckpt_num > 1
    ckpt_num = 1

    key_quant_bytes = batch * hidden_dim * seq_len * bits / 8
    if ckpt_num == 1:
        key_zero_point_bytes = hidden_dim * 2  # FP16
        key_scaling_factor_bytes = hidden_dim * 2
    else: # Checkpoint mode
        key_zero_point_bytes = ckpt_num * batch * hidden_dim * 2
        key_scaling_factor_bytes = ckpt_num * batch * hidden_dim * 2
    key_alpha_bytes = hidden_dim * 2
    key_beta_bytes = hidden_dim * 2
    key_target_bytes = hidden_dim * 2

    val_quant_bytes = batch * hidden_dim * seq_len * bits / 8
    if ckpt_num == 1:
        val_zero_point_bytes = hidden_dim * 2
        val_scaling_factor_bytes = hidden_dim * 2
    else: # Checkpoint mode
        val_zero_point_bytes = ckpt_num * batch * hidden_dim * 2
        val_scaling_factor_bytes = ckpt_num * batch * hidden_dim * 2
    val_alpha_bytes = hidden_dim * 2
    val_beta_bytes = hidden_dim * 2
    val_target_bytes = hidden_dim * 2

    # Value is same as key
    quant_bytes = key_quant_bytes + val_quant_bytes
    meta_bytes = key_zero_point_bytes + key_scaling_factor_bytes + key_alpha_bytes + key_beta_bytes + key_target_bytes + \
        val_zero_point_bytes + val_scaling_factor_bytes + \
        val_alpha_bytes + val_beta_bytes + val_target_bytes
    return quant_bytes, meta_bytes


def KVQuant_memory(batch, hidden_dim, seq_len, bits=4, outlier_ratio=0.01):
    """
    KVQuant (arxiv 2401.18079) memory breakdown, per layer.

    Layout follows the deployed cache (QuantK / QuantV in
    deployment/.../models/llama/modeling_llama.py):
      1. 4-bit packed K and V, per request.
      2. Key per-channel LUT, shape (hidden_dim, 2^bits), FP32, plus zero point
         and upper/lower outlier thresholds (3 x FP32 per channel). These are
         calibration artefacts, so they are model-static and shared across batch.
      3. Value per-token LUT, shape (seq_len, 2^bits), FP32 per request; scale
         and zero point are baked into this table, there are no separate tensors.
      4. Dense-and-sparse outliers: 2 * (int(outlier_ratio / 2 * hidden_dim) + 1)
         entries per token, FP32 value + INT32 index. The deployed cache uses the
         capped layout, so it stores no CSR pointer arrays. Values share the same
         per-token count (value topk keeps one extra slot per side for the
         min/max thresholds, which are not stored).

    Supports MHA: hidden_dim = num_heads * head_dim. The deployed cache does not
    have a batched implementation, so batch > 1 is an extrapolation (per-request tensors are
    replicated, calibration-derived ones are not).
    """
    # Quantized KV data
    key_quant_bytes = batch * hidden_dim * seq_len * bits / 8
    val_quant_bytes = batch * hidden_dim * seq_len * bits / 8

    # Per-channel key metadata (calibration-derived, batch-independent)
    key_lut_bytes = hidden_dim * (2 ** bits) * 4   # FP32 per LUT entry
    key_threshold_bytes = hidden_dim * 3 * 4       # zero point + upper/lower threshold

    # Per-token value LUT (scale and zero point baked in)
    val_lut_bytes = batch * seq_len * (2 ** bits) * 4

    # Dense-and-sparse outliers for both K and V
    num_outliers_per_token = 2 * (int(outlier_ratio / 2 * hidden_dim) + 1)
    outlier_bytes = 2 * batch * seq_len * \
        num_outliers_per_token * (4 + 4)  # FP32 value + INT32 index

    quant_bytes = key_quant_bytes + val_quant_bytes
    meta_bytes = (key_lut_bytes + key_threshold_bytes
                  + val_lut_bytes + outlier_bytes)
    return quant_bytes, meta_bytes


def memory_comparison_fig(proportion=False):
    """
    Plot grouped stacked bar charts comparing KIVI, MXFP4, ADKV, KVQuant memory usage.

    Each subplot varies one parameter (seq_len / batch / hidden_dim).
    Top row: quant_bytes and meta_bytes as percentage of total (proportion).
    Bottom row: quant_bytes and meta_bytes as absolute values (KB).
    """
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    # Colour palette
    METHODS = ['KIVI', 'KIVI-128g', 'MXFP4', 'KVQuant', 'ADKV']
    C_QUANT = ['#4C72B0', "#B0994C", '#8172B2',
               '#C44E52', '#55A868']   # solid colours for quant
    C_META = ['#A0B8D8', "#D7D8A0", '#C4B8E0',
              '#E4A8A8', '#A8D4B4']   # lighter shades for meta

    # Fixed default parameters
    DEF_BATCH = 1
    DEF_HIDDEN_DIM = 4096
    DEF_SEQ_LEN = 1024
    DEF_GROUP_SIZE = 32
    DEF_BITS = 4
    DEF_LAYER = 32

    # Sweep ranges
    seq_lens = [128, 256, 512, 1024, 2048, 4096]
    batches = [1, 2, 4, 8, 16, 32]
    hidden_dims = [128, 256, 512, 1024, 2048, 4096]

    # Helper: compute (quant, meta) for all four methods
    def _compute(batch, hidden_dim, seq_len):
        kivi = KIVI_memory(batch, hidden_dim, seq_len,
                           DEF_GROUP_SIZE, DEF_BITS)
        kivi_128 = KIVI_memory(batch, hidden_dim, seq_len, 128, DEF_BITS)
        mxfp4 = MXFP4_memory(batch, hidden_dim, seq_len, DEF_GROUP_SIZE)
        adkv = ADKV_memory(batch, hidden_dim, seq_len, DEF_BITS)
        kvquant = KVQuant_memory(batch, hidden_dim, seq_len, DEF_BITS)
        return {'KIVI': kivi, 'KIVI-128g': kivi_128, 'MXFP4': mxfp4, 'ADKV': adkv, 'KVQuant': kvquant}

    if proportion:
        res = _compute(32, 4096, 4096)
        # KIVI
        KIVI_metadata = res["KIVI"][1]
        KIVI_total = res["KIVI"][0] + res["KIVI"][1]
        KIVI_128g_metadata = res["KIVI-128g"][1]
        KIVI_128g_total = res["KIVI-128g"][0] + res["KIVI-128g"][1]
        # KVQuant
        KVQuant_metadata = res["KVQuant"][1]
        KVQuant_total = res["KVQuant"][0] + res["KVQuant"][1]
        # ADKV
        ADKV_metadata = res["ADKV"][1]
        ADKV_total = res["ADKV"][0] + res["ADKV"][1]
        logger.info("Reference Metrics: (metadata ratio, total reduction)")
        logger.info(
            f"  KIVI {(KIVI_metadata - ADKV_metadata) / KIVI_metadata * 100:.2f}% {(KIVI_total - ADKV_total) / KIVI_total * 100:.2f}%")
        logger.info(
            f"  KIVI-128g {(KIVI_128g_metadata - ADKV_metadata) / KIVI_128g_metadata * 100:.2f}% {(KIVI_128g_total - ADKV_total) / KIVI_128g_total * 100:.2f}%")

        logger.info(
            "Reference Metrics: (total memory, metadata memory, metadata ratio)")
        logger.info(
            f"  KIVI {KIVI_total * DEF_LAYER / 1024 / 1024 / 1024:.2f}GB {KIVI_metadata * DEF_LAYER / 1024 / 1024 / 1024:.2f}GB {KIVI_metadata / KIVI_total * 100:.2f}%")
        logger.info(
            f"  KVQuant {KVQuant_total * DEF_LAYER / 1024 / 1024 / 1024:.2f}GB {KVQuant_metadata * DEF_LAYER / 1024 / 1024 / 1024:.2f}GB {KVQuant_metadata / KVQuant_total * 100:.2f}%")

    # Helper: draw one subplot
    def _draw_bars(ax, x_values, compute_fn, x_label, y_label, title, as_proportion):
        """
        For each value in x_values, draw four side-by-side stacked bars (one per method).
        Stack bottom = quant_bytes, top = meta_bytes.

        as_proportion=True -> y-axis in % of total
        as_proportion=False -> y-axis in KB
        """
        n_x = len(x_values)
        n_meth = len(METHODS)
        bar_w = 0.18
        x = np.arange(n_x)

        for i, method in enumerate(METHODS):
            quant_vals, meta_vals = [], []
            for xv in x_values:
                q, m = compute_fn(xv)[method]
                total = q + m
                if as_proportion:
                    quant_vals.append(q / total * 100)
                    meta_vals.append(m / total * 100)
                else:
                    quant_vals.append(q / 1024 / 1024 / 1024 *
                                      DEF_LAYER)       # bytes -> KB
                    meta_vals.append(m / 1024 / 1024 / 1024 * DEF_LAYER)

            quant_vals = np.array(quant_vals)
            meta_vals = np.array(meta_vals)
            offset = (i - (n_meth - 1) / 2) * bar_w

            ax.bar(x + offset, quant_vals, bar_w,
                   color=C_QUANT[i], label=f'{method} quant')
            ax.bar(x + offset, meta_vals, bar_w, bottom=quant_vals,
                   color=C_META[i],  label=f'{method} meta')

            # Annotate meta proportion on the top of each bar (proportion plot only)
            if as_proportion:
                for j, (qv, mv) in enumerate(zip(quant_vals, meta_vals)):
                    if mv > 2:   # skip tiny slivers
                        ax.text(x[j] + offset, qv + mv / 2,
                                f'{mv:.0f}%', ha='center', va='center',
                                fontsize=6, color='black')

        ax.set_xticks(x)
        ax.set_xticklabels(x_values, fontsize=12)
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_title(title, fontsize=16)

        ax.set_ylabel(y_label, fontsize=16)
        ax.yaxis.set_tick_params(labelsize=12)

        if as_proportion:
            ax.set_ylim(0, 110)
            ax.axhline(100, color='grey', linewidth=0.5, linestyle='--')
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Build figure: 2 rows (proportion) / 1 row × 3 cols
    fig, axes = plt.subplots(2 if proportion else 1, 3,
                             figsize=(18, 10 if proportion else 5))

    if proportion:
        # Row 0: proportion (%)
        _draw_bars(axes[0, 0],
                   seq_lens,
                   lambda sl: _compute(DEF_BATCH, DEF_HIDDEN_DIM, sl),
                   'Sequence Length',
                   'Proportion of Total (%)',
                   f'Memory Proportion vs. Sequence Length\n(Batch Size = {DEF_BATCH}, Hidden Dimension = {DEF_HIDDEN_DIM})',
                   as_proportion=True)

        _draw_bars(axes[0, 1],
                   batches,
                   lambda bs: _compute(bs, DEF_HIDDEN_DIM, DEF_SEQ_LEN),
                   'Batch Size',
                   '',
                   f'Memory Proportion vs. Batch Size\n(Sequence Length = {DEF_SEQ_LEN}, Hidden Dimension = {DEF_HIDDEN_DIM})',
                   as_proportion=True)

        _draw_bars(axes[0, 2],
                   hidden_dims,
                   lambda hd: _compute(DEF_BATCH, hd, DEF_SEQ_LEN),
                   'Hidden Dimension',
                   '',
                   f'Memory Proportion vs. Hidden Dimension\n(Batch Size = {DEF_BATCH}, Sequence Length = {DEF_SEQ_LEN})',
                   as_proportion=True)

    # Row 1: absolute (KB)
    _draw_bars(axes[1, 0] if proportion else axes[0],
               seq_lens,
               lambda sl: _compute(DEF_BATCH, DEF_HIDDEN_DIM, sl),
               'Sequence Length',
               'Memory (GB)',
               f'Memory Footprint vs. Sequence Length\n(Batch Size = {DEF_BATCH}, Hidden Dimension = {DEF_HIDDEN_DIM})',
               as_proportion=False)

    _draw_bars(axes[1, 1] if proportion else axes[1],
               batches,
               lambda bs: _compute(bs, DEF_HIDDEN_DIM, DEF_SEQ_LEN),
               'Batch Size',
               '',
               f'Memory Footprint vs. Batch Size\n(Sequence Length = {DEF_SEQ_LEN}, Hidden Dimension = {DEF_HIDDEN_DIM})',
               as_proportion=False)

    _draw_bars(axes[1, 2] if proportion else axes[2],
               hidden_dims,
               lambda hd: _compute(DEF_BATCH, hd, DEF_SEQ_LEN),
               'Hidden Dimension',
               '',
               f'Memory Footprint vs. Hidden Dimension\n(Batch Size = {DEF_BATCH}, Sequence Length = {DEF_SEQ_LEN})',
               as_proportion=False)

    # Shared legend
    legend_handles = []
    for i, method in enumerate(METHODS):
        legend_handles.append(mpatches.Patch(
            color=C_QUANT[i], label=f'{method} — Quantized Bytes'))
        legend_handles.append(mpatches.Patch(
            color=C_META[i],  label=f'{method} — Metadata Bytes'))

    if proportion:
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.legend(handles=legend_handles,
                   loc='lower center', bbox_to_anchor=(0.5, 0.90), ncol=5, frameon=False, fontsize=12)
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.85])
        fig.legend(handles=legend_handles,
                   loc='lower center', bbox_to_anchor=(0.5, 0.85), ncol=5, frameon=False, fontsize=12)

    return fig


def parse_args():
    p = argparse.ArgumentParser(
        description="Memory Comparison."
    )

    p.add_argument("--out_dir", type=str, default="figures")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[Figure] Memory Comparison Figure (Figure 3)...")
    fig = memory_comparison_fig(proportion=False)
    fig.savefig(out_dir / "memory_comparison_fig.pdf")

    logger.info("[Figure] Memory Comparison Full Figure (Figure 6)...")
    fig = memory_comparison_fig(proportion=True)
    fig.savefig(out_dir / "memory_comparison_full_fig.pdf")


if __name__ == '__main__':
    main()
