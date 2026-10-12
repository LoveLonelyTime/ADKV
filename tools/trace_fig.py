import argparse
from pathlib import Path
import logging
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Trace of KV cache."
    )

    p.add_argument("--calib_dir", type=str, default="calib")
    p.add_argument("--out_dir", type=str, default="figures")
    return p.parse_args()


def trace_fig(key: np.ndarray):
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_red", ["white", "red"]
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    data = key.T  # (dim, token)

    vmin = np.percentile(data, 10)  # < 90% -> White
    vmax = np.percentile(data, 99.5)  # > 99.5 -> Red (Outliers)

    ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )

    ax.set_xlabel("Token Index", fontsize=12)
    ax.set_ylabel("Channel Index", fontsize=12)

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Llama-2 7B Layer #12
    if not (Path(args.calib_dir) / "calib_llama_11.pt").exists():
        raise FileNotFoundError(
            "calib_llama_11.pt not found, please collect calibration dataset first")

    calib_dataset = torch.load(Path(args.calib_dir) / "calib_llama_11.pt")
    key = calib_dataset["calib_k"].numpy().astype(np.float32)
    logger.log(f"key shape: {key.shape}")

    key_abs = np.abs(key[0, :, :])  # (batch, token, dim)

    logger.info("[Figure] WTZP / WTS Figure (Figure 2 Left)...")
    fig = trace_fig(key_abs)

    fig.savefig(out_dir / "trace_fig.pdf")
    logger.info(f"Saved: {out_dir}/trace_fig.pdf")


if __name__ == "__main__":
    main()
