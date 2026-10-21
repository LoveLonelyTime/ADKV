"""
Ray-based parallel ADKV training launcher.

Each Ray task trains one layer and claims 1 GPU.
Ray automatically queues tasks when all GPUs are occupied.
"""

import argparse
import logging
from pathlib import Path
import ray

from .adkv_train_ray_worker import train_layer_task, ALL_ATTENTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_layers(spec: str):
    """
    Parse layer spec: '0-31' -> [0..31], '0,1,5' -> [0,1,5].
    """
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end)))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def create_parser():
    p = argparse.ArgumentParser(
        description="Ray-parallel ADKV training: one task per layer, 1 GPU each."
    )
    # Layer selection
    p.add_argument("--layers", type=str, default="0-31",
                   help="Layers to train, e.g. '0-31' or '0,1,5,11'")
    p.add_argument("--calib_dir", type=str, default="calib")

    # Training hyperparams
    p.add_argument("--num_hidden_states", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--truncation", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--scale_min", type=float, default=-4.0)
    p.add_argument("--scale_max", type=float, default=4.0)
    p.add_argument("--atten", type=str, default="llama",
                   choices=list(ALL_ATTENTIONS.keys()),
                   help="Attention function to use for loss computation")
    p.add_argument("--checkpoint", action='store_true',
                   help="Enable checkpoint loading")

    # Ray options
    p.add_argument("--ray_address", type=str, default=None,
                   help="Ray cluster address, e.g. 'auto' or 'ray://host:10001'. "
                        "Omit to start a local Ray instance.")
    p.add_argument("--num_gpus_per_task", type=float, default=1.0,
                   help="GPUs requested per layer task (fractional allowed, e.g. 0.5)")
    return p.parse_args()


def main():
    args = create_parser()
    layers = parse_layers(args.layers)

    Path(args.calib_dir).mkdir(parents=True, exist_ok=True)

    # Initialize Ray (local cluster if no address given)
    ray_kwargs = {}
    if args.ray_address:
        ray_kwargs["address"] = args.ray_address
    ray.init(**ray_kwargs)

    logger.info(
        f"Ray initialized. Available resources: {ray.available_resources()}")
    logger.info(f"Submitting {len(layers)} layer tasks: {layers}")

    # Allow fractional GPU via a decorated variant when needed
    if args.num_gpus_per_task != 1.0:
        task_fn = train_layer_task.options(num_gpus=args.num_gpus_per_task)
    else:
        task_fn = train_layer_task

    # Submit all tasks
    # Ray queues them automatically when GPUs are full
    futures = {}
    for layer in layers:
        dataset_path = Path(args.calib_dir) / f"calib_llama_{layer}.pt"
        ckpt_k = Path(args.calib_dir) / f"ckpt_k_{layer}.pt"
        ckpt_v = Path(args.calib_dir) / f"ckpt_v_{layer}.pt"

        ref = task_fn.remote(
            layer_idx=layer,
            dataset_path=dataset_path,
            ckpt_k=ckpt_k,
            ckpt_v=ckpt_v,
            num_hidden_states=args.num_hidden_states,
            batch_size=args.batch_size,
            truncation=args.truncation,
            epochs=args.epochs,
            lr=args.learning_rate,
            grad_clip=args.grad_clip,
            bits=args.bits,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            atten=args.atten,
            checkpoint=args.checkpoint
        )
        futures[ref] = layer
        logger.info(f"Submitted layer {layer:2d} -> {dataset_path}")

    # Wait for all tasks, reporting results as they finish
    logger.info(f"Waiting for {len(futures)} tasks to complete...")
    done_refs = list(futures.keys())
    completed = 0
    failed = []

    while done_refs:
        finished, done_refs = ray.wait(done_refs, num_returns=1)
        ref = finished[0]
        layer = futures[ref]
        try:
            result = ray.get(ref)
            completed += 1
            logger.info(
                f"[DONE] Layer {layer:2d} done [{completed}/{len(futures)}] ckpt_k={result['ckpt_k']} ckpt_v={result['ckpt_v']}")
        except Exception as e:
            failed.append(layer)
            logger.info(f"[FAIL] Layer {layer:2d} failed: {e}")

    logger.info(f"All tasks finished. Completed: {completed}, Failed: {len(failed)}")
    if failed:
        logger.info(f"Failed layers: {failed}")

    ray.shutdown()


if __name__ == "__main__":
    main()
