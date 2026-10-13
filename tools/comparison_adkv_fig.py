import argparse
from pathlib import Path
import logging
import sys
import subprocess
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from adkv.train_model import ADKVTrainModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rtn_quant(x: np.ndarray, g_size=32):
    zp = np.zeros_like(x)
    l = len(x)
    for i in range(0, l, g_size):
        g = x[i: min(i + g_size, l)]
        zp[i: min(i + g_size, l)] = np.min(g)
    return zp


def load_layer_data(
    calib_dir: str,
    layer: int,
    model_path: str | None,
    dataset_path: str | None,
    n_samples: int = 12,
    clip_len: int = 16000,
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


def comparison_adkv_fig(
    key: np.ndarray,
    rtn_zp: np.ndarray,
    adkv_zp: np.ndarray,
):
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(key, label='Key')
    ax.plot(rtn_zp, label='RTN Zero Point')
    ax.plot(adkv_zp, label='ADKV Zero Point')

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=12,
              handlelength=1.2, columnspacing=0.8, handletextpad=0.3, borderpad=0.2)

    ax.set_xlabel("Token Index", fontsize=16)
    ax.set_ylabel("Magnitude", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    return fig


def parse_args():
    p = argparse.ArgumentParser(
        description="Trace of KV cache."
    )

    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--dim", type=int, default=24)
    p.add_argument("--limit", type=int, default=256)

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

    # Load ADKV parameters
    if not (Path(args.calib_dir) / f"ckpt_k_{args.layer}.pt").exists():
        raise FileNotFoundError(
            f"ckpt_k_{args.layer}.pt not found, please calibrate the ADKV parameters first")

    k_model = ADKVTrainModel(4096, truncation=32).to('cpu')
    k_model.load_state_dict(torch.load(
        Path(args.calib_dir) / f"ckpt_k_{args.layer}.pt"))

    logger.info("Running ADKV...")
    with torch.no_grad():
        _, zp_outputs, _ = k_model(
            torch.from_numpy(key[0:1]), output_states=True)
        adkv_zp = zp_outputs.numpy()

    logger.info("Running RTN...")
    rtn_zp = rtn_quant(key[0, :args.limit, args.dim])

    logger.info("[Figure] Comparison ADKV Figure (Figure 2 Right)...")
    fig = comparison_adkv_fig(
        key[0, :args.limit, args.dim], rtn_zp, adkv_zp[0, :args.limit, args.dim])

    fig.savefig(out_dir / "comparison_adkv_fig.pdf")
    logger.info(f"Saved: {out_dir}/comparison_adkv_fig.pdf")


if __name__ == "__main__":
    main()
