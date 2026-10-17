import argparse
from pathlib import Path
import logging
import math
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.stats import gaussian_kde

from adkv.train_model import ADKVTrainModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PARAM_META = {
    # key: (display_name, transform_fn, colormap, symmetric_norm)
    "init_zp": (
        "$ZP_0$\n(Initial Zero Point)",
        lambda p: p.init_zp.detach().cpu().float().numpy(),
        "RdBu_r", True,
    ),
    "init_sc": (
        "$S_0$\n(Initial Scaling Factor)",
        lambda p: p.get_sc().clamp(min=math.exp(-4.0), max=math.exp(4.0)
                                   ).detach().cpu().float().numpy(),
        "YlOrRd", False,
    ),
    "alpha": (
        "$\\alpha$\n(Zero Point Tracking Rate)",
        lambda p: p.get_alpha().detach().cpu().float().numpy(),
        "Blues", False,
    ),
    "beta": (
        "$\\beta$\n(Scaling Factor Tracking Rate)",
        lambda p: p.get_beta().detach().cpu().float().numpy(),
        "Greens", False,
    ),
    "target": (
        "$T$\n(Quantization Target)",
        lambda p: p.target.detach().cpu().float().numpy(),
        "PuOr", True,
    ),
}


def load_params(ckpt_dir: str, num_layers: int, num_hidden_states: int):
    """
    Load ADKV parameters.

    Parameters:
      ckpt_dir: Path to ckpts.

    Returns:
      data["k"][param_key] -> [num_layers, C]
      data["v"][param_key] -> [num_layers, C]
      data["layers"] -> list[int]
    """

    data = {"k": {pk: [] for pk in PARAM_META},
            "v": {pk: [] for pk in PARAM_META},
            "layers": []}

    ckpt_dir = Path(ckpt_dir)

    for layer in range(0, num_layers):
        ckpt_file_k = ckpt_dir / f"ckpt_k_{layer}.pt"
        ckpt_file_v = ckpt_dir / f"ckpt_v_{layer}.pt"

        if not ckpt_file_k.exists():
            raise FileNotFoundError(
                f"ckpt_k_{layer}.pt not found, please calibrate the ADKV parameters first")
        if not ckpt_file_v.exists():
            raise FileNotFoundError(
                f"ckpt_v_{layer}.pt not found, please calibrate the ADKV parameters first")

        model_k = ADKVTrainModel(num_hidden_states)
        model_k.load_state_dict(torch.load(ckpt_file_k, map_location="cpu"))
        model_v = ADKVTrainModel(num_hidden_states)
        model_v.load_state_dict(torch.load(ckpt_file_v, map_location="cpu"))

        for pk, (_, tfn, _, _) in PARAM_META.items():
            data["k"][pk].append(tfn(model_k))
            data["v"][pk].append(tfn(model_v))

        data["layers"].append(layer)

    for kv in ("k", "v"):
        for pk in PARAM_META:
            data[kv][pk] = np.stack(data[kv][pk], axis=0)  # [L, C]

    return data


def adkv_param_distribution_fig(data: dict):
    """
    All visualizations on one page.

    GridSpec: 4 rows x (5 param-cols + 5 thin cbar-cols) = 10 cols total.
    Rows 0-1 : heatmaps (K / V), shared colorbar in the cbar column.
    Row 2 : per-layer mean ± std line chart (spans param+cbar cols).
    Row 3 : global KDE density (spans param+cbar cols).
    """
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    COLOR_K = "#2166ac"   # blue - Key
    COLOR_V = "#d6604d"   # red - Value

    param_keys = list(PARAM_META.keys())
    N = len(param_keys)  # 5
    layers = data["layers"]
    L = len(layers)
    C = data["k"]["init_zp"].shape[1]
    layer_arr = np.array(layers)

    # Build GridSpec
    # Alternating [main_col, cbar_col, pad_col] x N params
    col_widths = []
    for _ in range(N):
        col_widths += [1.0, 0.05, 0.25]

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(
        nrows=4, ncols=N * 3,
        width_ratios=col_widths,
        wspace=0,
        figure=fig
    )

    ytick_step = max(1, L // 6)
    xtick_step = max(1, C // 4)

    for col_idx, pk in enumerate(param_keys):
        label, _, cmap, symmetric = PARAM_META[pk]
        mat_k = data["k"][pk]
        mat_v = data["v"][pk]

        # Shared norm for K and V (same color scale)
        combined = np.concatenate([mat_k.ravel(), mat_v.ravel()])
        vmin, vmax = combined.min(), combined.max()
        if symmetric:
            abs_max = max(abs(vmin), abs(vmax))
            norm = TwoSlopeNorm(vcenter=0, vmin=-abs_max, vmax=abs_max)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        # Row 0: Key heatmap
        ax_k = fig.add_subplot(gs[0, col_idx * 3])
        im = ax_k.imshow(mat_k, aspect="auto", cmap=cmap, norm=norm,
                         interpolation="nearest", origin="upper")
        cax = fig.add_subplot(gs[0, col_idx * 3 + 1])
        fig.colorbar(im, cax=cax)
        ax_k.set_box_aspect(1)
        ax_k.set_title(label, fontsize=16)
        ax_k.set_xticks([])
        if col_idx == 0:
            ax_k.set_yticks(range(0, L, ytick_step))
            ax_k.set_yticklabels(
                [str(layers[i]) for i in range(0, L, ytick_step)])
            ax_k.tick_params(axis="y", labelsize=12)
            ax_k.set_ylabel("Layer Index", fontsize=16)
        else:
            ax_k.set_yticks([])

        # Row 1: Value heatmap
        ax_v = fig.add_subplot(gs[1, col_idx * 3])
        ax_v.imshow(mat_v, aspect="auto", cmap=cmap, norm=norm,
                    interpolation="nearest", origin="upper")
        ax_v.set_box_aspect(1)
        ax_v.set_xticks(range(0, C, xtick_step))
        ax_v.set_xticklabels([str(i) for i in range(0, C, xtick_step)])
        ax_v.tick_params(axis="x", labelsize=12)
        ax_v.set_xlabel("Channel Index", fontsize=16)
        if col_idx == 0:
            ax_v.set_yticks(range(0, L, ytick_step))
            ax_v.set_yticklabels(
                [str(layers[i]) for i in range(0, L, ytick_step)])
            ax_v.tick_params(axis="y", labelsize=12)
            ax_v.set_ylabel("Layer Index", fontsize=16)
        else:
            ax_v.set_yticks([])

        # Row 2: Per-layer mean ± std
        ax_line = fig.add_subplot(gs[2, col_idx * 3])
        for kv, color, ls, tag in [
            ("k", COLOR_K, "-",  "Key"),
            ("v", COLOR_V, "--", "Value"),
        ]:
            mu = data[kv][pk].mean(axis=1)
            sd = data[kv][pk].std(axis=1)
            ax_line.plot(layer_arr, mu, color=color, ls=ls, lw=1.0, label=tag)
            ax_line.set_box_aspect(1)
            ax_line.fill_between(layer_arr, mu - sd, mu + sd,
                                 color=color, alpha=0.18)
        ax_line.set_xlim(layer_arr[0], layer_arr[-1])
        ax_line.set_xlabel("Layer Index", fontsize=16)
        ax_line.tick_params(labelsize=12)
        ax_line.grid(True, lw=0.3, alpha=0.45)
        ax_line.spines[["top", "right"]].set_visible(False)
        if col_idx == 0:
            ax_line.set_ylabel("Mean ± Std", fontsize=16)
        if col_idx == N - 1:
            ax_line.legend(frameon=False, loc="upper right",
                           handlelength=1.2, handletextpad=0.3)

        # Row 3: KDE density
        ax_kde = fig.add_subplot(gs[3, col_idx * 3])
        for kv, color, ls, tag in [
            ("k", COLOR_K, "-",  "Key"),
            ("v", COLOR_V, "--", "Val"),
        ]:
            vals = data[kv][pk].ravel()
            lo, hi = np.percentile(vals, [0.5, 99.5])
            xs = np.linspace(lo, hi, 300)
            ys = gaussian_kde(vals, bw_method="silverman")(xs)
            ax_kde.plot(xs, ys, color=color, ls=ls, lw=1.0, label=tag)
            ax_kde.set_box_aspect(1)
            ax_kde.fill_between(xs, 0, ys, color=color, alpha=0.12)
        ax_kde.set_xlabel("Value", fontsize=16)
        ax_kde.set_yticks([])
        ax_kde.tick_params(labelsize=12)
        ax_kde.spines[["top", "right", "left"]].set_visible(False)
        ax_kde.grid(axis="x", lw=0.3, alpha=0.45)
        if col_idx == 0:
            ax_kde.set_ylabel("Density", fontsize=16)

    fig.tight_layout(rect=[0.01, 0, 1, 1])

    # Row section labels in left margin
    for r, txt in enumerate(["Heatmap (Key)", "Heatmap (Value)", "Layer mean", "KDE"]):
        pos = gs[r, 0].get_position(fig)
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(0.01, y_center, txt, ha="left", va="center",
                 fontsize=16, rotation=90, linespacing=1.3)

    return fig


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize ADKV trained parameter distributions."
    )

    p.add_argument("--num_layers", type=int, default=32)
    p.add_argument("--num_hidden_states", type=int, default=4096)

    p.add_argument("--calib_dir", type=str, default="calib")
    p.add_argument("--out_dir", type=str, default="figures")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.calib_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Loading K/V cache parameters from {args.num_layers} layers...")
    data = load_params(args.calib_dir, args.num_layers, args.num_hidden_states)

    logger.info("[Figure] ADKV Param Distribution Figure (Figure 6)...")
    fig = adkv_param_distribution_fig(data)

    fig.savefig(out_dir / "adkv_param_distribution_fig.pdf")
    logger.info(f"Saved: {out_dir}/adkv_param_distribution_fig.pdf")


if __name__ == "__main__":
    main()
