import argparse
import logging
from pathlib import Path
import sys
import subprocess
from tqdm import tqdm
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sliding_window_mean(
    data: np.ndarray,
    window: int,
    stride: int,
):
    """
    Non-causal sliding window mean along the token (time) axis.

    Parameters:
        data: [N, T, C] float64
        window: window width W
        stride: window stride S

    Returns:
        mu: [N, n_windows, C], where n_windows = len(range(0, T-W+1, S))

    Notes:
        mu[n, w, c] = mean(data[n, wS : wS+W, c])
        This is the local mean for channel c inside window w.
    """
    N, T, C = data.shape
    starts = list(range(0, T - window + 1, stride))
    n_windows = len(starts)
    mu = np.empty((N, n_windows, C), dtype=np.float64)
    for i, s in enumerate(starts):
        mu[:, i, :] = data[:, s: s + window, :].mean(axis=1)
    return mu  # [N, n_windows, C]


def sliding_window_scale(
    data: np.ndarray,
    window: int,
    stride: int,
) -> np.ndarray:
    """
    Sliding window mean-absolute-deviation (MAD) along the token axis.

    Parameters:
        data: [N, T, C] float64
        window: window width W
        stride: window stride S

    Returns:
        env: [N, n_windows, C]

    Notes:
        mu_w = mean(data[wS : wS+W, :]) [N, C]
        env[w, c] = mean(|data[wS : wS+W, c] - mu_w[c]|)
    """
    N, T, C = data.shape
    starts = list(range(0, T - window + 1, stride))
    n_windows = len(starts)
    env = np.empty((N, n_windows, C), dtype=np.float64)
    for i, s in enumerate(starts):
        chunk = data[:, s: s + window, :]  # [N, W, C]
        mu_w = chunk.mean(axis=1, keepdims=True)  # [N, 1, C]
        env[:, i, :] = np.abs(chunk - mu_w).mean(axis=1)
    return env  # [N, n_windows, C]


def ema_trackability_seq(
    seq: np.ndarray,
    alpha_grid: np.ndarray,
):
    """
    Grid-search EMA-trackability on a coarse sequence.

    For each channel c, maximise over alpha:
        P_c(alpha) = 1 - Var_w(r_w) / Var_w(s_w) (averaged over N sequences)
    where s_w = seq[:, w, c] and r_w is the causal EMA residual.

    Parameters:
        seq: [N, n_windows, C] the coarse signal (mu or env)
        alpha_grid: 1-D array of alpha values to search

    Returns:
        P_star: [C] optimal trackability per channel
        alpha_star: [C] optimal EMA rate per channel
    """
    N, W, C = seq.shape
    s = seq.astype(np.float64)

    var_signal = s.var(axis=1)  # [N, C]
    valid = var_signal > 1e-10  # channels with real variance

    A = len(alpha_grid)
    P_matrix = np.full((A, C), np.nan)

    for i, alpha in enumerate(tqdm(alpha_grid)):
        dc = s[:, 0, :].copy()  # [N, C]
        residuals = np.empty((N, W - 1, C), dtype=np.float64)
        for w in range(1, W):
            r = s[:, w, :] - dc  # [N, C]
            residuals[:, w - 1, :] = r
            dc = dc + alpha * r

        var_resid = residuals.var(axis=1)  # [N, C]
        ratio = np.where(valid, var_resid / (var_signal + 1e-12), np.nan)
        with np.errstate(all="ignore"):
            P_matrix[i] = 1.0 - np.nanmean(ratio, axis=0)

    best_idx = np.nanargmax(P_matrix, axis=0)  # [C]
    P_star = P_matrix[best_idx, np.arange(C)]
    alpha_star = alpha_grid[best_idx]

    return P_star, alpha_star


def window_trackability(
    data: np.ndarray,
    window: int,
    stride: int,
    alpha_grid: np.ndarray,
):
    """
    Compute WTZP and WTS for a single KV tensor.

    Parameters:
        data: [N, T, C]
        window: window width W
        stride: stride S
        alpha_grid: alpha search grid

    Returns:
        WTZP: [C] zero point trackability
        WTS: [C] scaling factor trackability
        alpha_zp: [C] optimal alpha for zero point tracking
        alpha_sc [C] optimal alpha for scaling factor tracking
        n_windows: number of windows
    """
    data = data.astype(np.float64)
    mu = sliding_window_mean(data, window, stride)  # [N, n_w, C]
    env = sliding_window_scale(data, window, stride)  # [N, n_w, C]

    WTZP, alpha_zp = ema_trackability_seq(mu, alpha_grid)
    WTS, alpha_sc = ema_trackability_seq(env, alpha_grid)

    return {
        "WTZP": WTZP,
        "WTS": WTS,
        "alpha_zp": alpha_zp,
        "alpha_sc": alpha_sc,
        "n_windows":  mu.shape[1],
    }


def shuffle_tokens(data: np.ndarray, seed: int = 42):
    """
    Shuffle the token (time) dimension independently per sample.

    This destroys all temporal auto-correlation while preserving the marginal per-channel distribution.
    The resulting WTZP / WTS values serve as a null-hypothesis baseline: if the original data had no temporal structure,
    its trackability scores would look like those of the shuffled version (typically P* = 0).

    Parameters:
        data: [N, T, C] float64
        seed: RNG seed for reproducibility

    Returns:
        out: [N, T, C] copy with T permuted independently per sample n
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(data)
    N, T, _ = data.shape
    for n in range(N):
        idx = rng.permutation(T)
        out[n] = data[n, idx, :]
    return out


def build_alpha_grid(n_alpha: int):
    return np.unique(np.concatenate([
        np.linspace(0.01, 0.10, n_alpha // 2),
        np.linspace(0.10, 0.99, n_alpha - n_alpha // 2),
    ]))


def wtzp_wts_fig(
    tp_all: dict[int, dict],
    layers: list[int],
    tp_shuf: dict[int, dict] | None = None,
    show_std: bool = False
):
    """
    Grouped bar chart: mean WTZP and WTS per layer for K and V.

    Raw values are always plotted on a linear axis.
    If tp_shuf is provided, the shuffled-token control group is overlaid on the same bars using hatched (diagonal-line) fill
    so both groups are directly comparable without normalisation.
    """
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    def _stats(tp_dict, key, metric):
        means = np.array([tp_dict[L][key][metric].mean() for L in layers])
        stds = np.array([tp_dict[L][key][metric].std() for L in layers])
        return means, stds

    x = np.arange(len(layers))
    width = 0.20
    offsets = [-1.5, -0.5, 0.5, 1.5]

    colours = ["#4393c3", "#2166ac", "#f4a582", "#d6604d"]
    labels = [
        "K WTZP",
        "K WTS",
        "V WTZP",
        "V WTS",
    ]
    keys_metrics = [("k", "WTZP"), ("k", "WTS"),
                    ("v", "WTZP"), ("v", "WTS")]

    fig, ax = plt.subplots(figsize=(5, 5))

    for (kv, metric), off, col, lbl in zip(keys_metrics, offsets, colours, labels):
        means, stds = _stats(tp_all, kv, metric)

        # Primary bars (original data)
        ax.bar(x + off * width, means, width,
               yerr=stds if show_std else None, capsize=2.5,
               color=col, alpha=0.80,
               error_kw={"lw": 0.8},
               label=lbl)

        # Shuffled-control overlay: same position/size, hatched, no fill
        if tp_shuf is not None:
            s_means, s_stds = _stats(tp_shuf, kv, metric)
            ax.bar(x + off * width, s_means, width,
                   yerr=s_stds if show_std else None, capsize=2.5,
                   color="none", edgecolor=col, linewidth=0.8,
                   hatch="//\\\\", alpha=0.90,
                   error_kw={"lw": 0.8},
                   label=f"{lbl} (shuffled)" if off == offsets[0] else "_nolegend_")

    # Build a clean legend: solid patches for original, hatched for shuffled
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, alpha=0.80, label=l)
                      for c, l in zip(colours, labels)]
    if tp_shuf is not None:
        legend_handles.append(
            Patch(facecolor="none", edgecolor="gray", hatch="//\\\\",
                  linewidth=0.8, label="Shuffled")
        )
    ax.legend(handles=legend_handles,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=12,
              handlelength=1.2, columnspacing=0.8, handletextpad=0.3, borderpad=0.2)

    ax.set_ylabel("Sliding-Window WTZP and WTS", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer Index", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)

    ax.grid(True, axis="y", which="major",
            linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    return fig


def load_layer_data(
    calib_dir: str,
    layer: int,
    model_path: str | None,
    dataset_path: str | None,
    n_samples: int = 256,
    clip_len: int = 100,
    max_length: int = 516,
    batch_size: int = 16,
):
    pt_file = Path(calib_dir) / f"calib_llama_{layer}.pt"

    if pt_file.exists():
        logger.info(f"[layer {layer}] Loading {pt_file}")
        return torch.load(pt_file, map_location="cpu")

    logger.info(
        f"[layer {layer}] Calibrating (this may take a few minutes)...")
    cmd = [
        sys.executable, "-m", "adkv.calib_collector",
        f"--layers={layer}",
        f"--shuffle",
        f"--clip_len={clip_len}",
        f"--n_samples={n_samples}",
        f"--max_length={max_length}",
        f"--batch_size={batch_size}",
        f"--output={Path(calib_dir) / "calib_llama.pt"}",
    ]
    if model_path:
        cmd.append(f"--model={model_path}")
    if dataset_path:
        cmd.append(f"--calib_data={dataset_path}")
    subprocess.run(cmd, check=True)
    return torch.load(pt_file, map_location="cpu")


def parse_args():
    p = argparse.ArgumentParser(
        description="Sliding-window zero point and scaling factor trackability of KV cache."
    )

    p.add_argument("--layers", type=int, nargs="+", required=True)

    p.add_argument("--model", type=str)
    p.add_argument("--calib_data", type=str)
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--clip_len", type=int, default=100)
    p.add_argument("--max_length", type=int, default=516)
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument("--window", type=int, default=8,
                   help="Sliding window width W (tokens)")
    p.add_argument("--stride", type=int,
                   help="Sliding window stride S; defaults to W//2 if not set")

    p.add_argument("--n_alpha", type=int, default=50,
                   help="Number of alpha values in grid search")

    p.add_argument("--shuffle_seed", type=int, default=42,
                   help="RNG seed for the shuffled-token control group; set to -1 to disable the control group entirely")

    p.add_argument("--calib_dir", type=str, default="calib")
    p.add_argument("--out_dir", type=str, default="figures")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.calib_dir).mkdir(parents=True, exist_ok=True)

    layers = sorted(args.layers)
    stride = max(
        1, args.stride if args.stride is not None else args.window // 2)
    alpha_grid = build_alpha_grid(args.n_alpha)

    tp_all = {}  # tp_all[L]["k"]["WTZP"] = [C]
    tp_shuf = {} if args.shuffle_seed != -1 else None

    # Use cache
    if (out_dir / "wtzp_wts_cache.npy").exists():
        logger.info("Detected cache")
        cache = np.load(out_dir / "wtzp_wts_cache.npy",
                        allow_pickle=True).item()
        tp_all, tp_shuf, layers = cache["tp_all"], cache["tp_shuf"], cache["layers"]
    else:
        for L in layers:
            calib_dataset = load_layer_data(
                calib_dir=args.calib_dir,
                layer=L,
                model_path=args.model,
                dataset_path=args.calib_data,
                n_samples=args.n_samples,
                clip_len=args.clip_len,
                max_length=args.max_length,
                batch_size=args.batch_size
            )
            k_np = calib_dataset["calib_k"].numpy().astype(
                np.float64)  # [N, T, C]
            v_np = calib_dataset["calib_v"].numpy().astype(np.float64)

            _, T, _ = k_np.shape
            n_windows = max(1, (T - args.window) // stride + 1)
            logger.info(
                f"[layer {L}] data shape: {k_np.shape}, {n_windows} windows (W={args.window}, S={stride})")

            res_k = window_trackability(k_np, args.window, stride, alpha_grid)
            res_v = window_trackability(v_np, args.window, stride, alpha_grid)

            tp_all[L] = {"k": res_k, "v": res_v}

            logger.info(f"K WTZP mean={res_k['WTZP'].mean():.4f} std={res_k['WTZP'].std():.4f} | "
                        f"WTS mean={res_k['WTS'].mean():.4f} std={res_k['WTS'].std():.4f}")
            logger.info(f"V WTZP mean={res_v['WTZP'].mean():.4f} std={res_v['WTZP'].std():.4f} | "
                        f"WTS mean={res_v['WTS'].mean():.4f} std={res_v['WTS'].std():.4f}")

            # Shuffled-token control group
            if tp_shuf is not None:
                logger.info(
                    f"[layer {L}] computing shuffled-token control (seed={args.shuffle_seed}) ...")
                k_shuf = shuffle_tokens(k_np, seed=args.shuffle_seed)
                v_shuf = shuffle_tokens(v_np, seed=args.shuffle_seed)
                shuf_k = window_trackability(
                    k_shuf, args.window, stride, alpha_grid)
                shuf_v = window_trackability(
                    v_shuf, args.window, stride, alpha_grid)
                tp_shuf[L] = {"k": shuf_k, "v": shuf_v}

                logger.info(f"K WTZP mean={shuf_k['WTZP'].mean():.4f} std={shuf_k['WTZP'].std():.4f} | "
                            f"WTS mean={shuf_k['WTS'].mean():.4f} std={shuf_k['WTS'].std():.4f}")
                logger.info(f"V WTZP mean={shuf_v['WTZP'].mean():.4f} std={shuf_v['WTZP'].std():.4f} | "
                            f"WTS mean={shuf_v['WTS'].mean():.4f} std={shuf_v['WTS'].std():.4f}")

            del calib_dataset, k_np, v_np

        # Save cache
        np.save(out_dir / "wtzp_wts_cache.npy",
                {"tp_all": tp_all, "tp_shuf": tp_shuf, "layers": layers})

    logger.info("[Figure] WTZP / WTS Figure (Figure 2 Left)...")

    fig = wtzp_wts_fig(tp_all, layers, tp_shuf=tp_shuf)
    fig.savefig(out_dir / "wtzp_wts_fig.pdf")
    logger.info(f"Saved: {out_dir}/wtzp_wts_fig.pdf")


if __name__ == "__main__":
    main()
