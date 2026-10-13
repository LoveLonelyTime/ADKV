import argparse
from pathlib import Path
import logging
import sys
import subprocess
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_layer_data(
    calib_dir: str,
    layer: int,
    model_path: str | None,
    dataset_path: str | None,
    n_samples: int = 12,
    clip_len:int = 16000,
    max_length: int = 4096,
    batch_size: int = 2,
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


def trace_fig(key: np.ndarray):
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_red", ["white", "red"]
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    data = key.T  # (dim, token)

    vmin = np.percentile(data, 90)  # < 90% -> White
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Trace of KV cache."
    )

    p.add_argument("--layer", type=int, default=11)

    p.add_argument("--model", type=str)
    p.add_argument("--calib_data", type=str)
    p.add_argument("--n_samples", type=int, default=12)
    p.add_argument("--clip_len", type=int, default=16000)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=2)

    p.add_argument("--calib_dir", type=str, default="calib")
    p.add_argument("--out_dir", type=str, default="figures")
    return p.parse_args()

def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.calib_dir).mkdir(parents=True, exist_ok=True)

    calib_dataset = load_layer_data(
        calib_dir=args.calib_dir,
        layer=args.layer,
        model_path=args.model,
        dataset_path=args.calib_data,
        n_samples=args.n_samples,
        clip_len=args.clip_len,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    key = calib_dataset["calib_k"].numpy().astype(np.float32)
    logger.info(f"key shape: {key.shape}")

    key_abs = np.abs(key[0, :, :])  # (batch, token, dim)

    logger.info("[Figure] Trace Figure (Figure 2 Middle)...")
    fig = trace_fig(key_abs)

    fig.savefig(out_dir / "trace_fig.pdf")
    logger.info(f"Saved: {out_dir}/trace_fig.pdf")


if __name__ == "__main__":
    main()
