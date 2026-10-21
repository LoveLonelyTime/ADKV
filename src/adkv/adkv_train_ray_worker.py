"""
Worker module for Ray-based parallel ADKV training.

This module contains the @ray.remote task and all supporting model/training code.
"""

import os
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import ray

from .train_model import ADKVTrainModel
from .llama_attention import llama_calib_attention

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_ATTENTIONS = {
    "llama":    llama_calib_attention
}


def _train_step(model_k, model_v, optimizer_k, optimizer_v,
                batch_q, batch_k, batch_v,
                atten, device, grad_clip=1.0):
    """
    ADKV Training Logic
    """
    # 1) Teacher attention
    with torch.no_grad():
        attn_output, attn_weights = atten(batch_q, batch_k, batch_v, None, 128)

    # 2) model_k / model_v forward
    output_k = model_k(batch_k)
    output_v = model_v(batch_v)

    # 3) Student attention (same kernel as teacher)
    model_attn_output, model_attn_weights = atten(
        batch_q, output_k, output_v, None, 128)

    # Per-batch MSE
    total_loss_output = ((model_attn_output - attn_output) ** 2).mean()
    # total_loss_weights = ((model_attn_weights - attn_weights) ** 2).mean()
    total_loss_weights = torch.tensor(0.0)
    total_loss_k = ((batch_k - output_k) ** 2).mean()
    total_loss_v = ((batch_v - output_v) ** 2).mean()
    # Alternative: total_loss = total_loss_output + total_loss_weights + total_loss_v + total_loss_k
    total_loss = total_loss_output

    # 4) Backward (+ clip + step)
    optimizer_k.zero_grad()
    optimizer_v.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model_k.parameters(), grad_clip)
    torch.nn.utils.clip_grad_norm_(model_v.parameters(), grad_clip)
    optimizer_k.step()
    optimizer_v.step()

    return (total_loss_output.item(), total_loss_weights.item(),
            total_loss_k.item(), total_loss_v.item())


def _train(layer_idx, model_k, model_v, dataloader, atten, device,
           epochs=10, lr=1e-3, grad_clip=1.0):
    try:
        from aim import Run
        run = Run(experiment=f"layer_{layer_idx:02d}")
        use_aim = True
    except Exception:
        run = None
        use_aim = False

    optimizer_k = torch.optim.Adam(model_k.parameters(), lr=lr)
    optimizer_v = torch.optim.Adam(model_v.parameters(), lr=lr)

    pbar = tqdm(range(epochs), desc=f"[Layer {layer_idx}] training",
                position=layer_idx % 8, leave=True)
    for epoch in pbar:
        epoch_loss_output = 0.0
        epoch_loss_weights = 0.0
        epoch_loss_k = 0.0
        epoch_loss_v = 0.0
        for batch_q, batch_k, batch_v in dataloader:
            batch_q = batch_q.to(dtype=torch.float)
            batch_k = batch_k.to(dtype=torch.float)
            batch_v = batch_v.to(dtype=torch.float)

            loss_output, loss_weights, loss_k, loss_v = _train_step(
                model_k, model_v, optimizer_k, optimizer_v,
                batch_q, batch_k, batch_v,
                atten, device, grad_clip)
            epoch_loss_output += loss_output
            epoch_loss_weights += loss_weights
            epoch_loss_k += loss_k
            epoch_loss_v += loss_v

        n = len(dataloader)
        if use_aim:
            run.track(epoch_loss_output / n, name="loss_output",
                      epoch=epoch, context={"subset": "train"})
            run.track(epoch_loss_weights / n, name="loss_weights",
                      epoch=epoch, context={"subset": "train"})
            run.track(epoch_loss_k / n, name="loss_k",
                      epoch=epoch, context={"subset": "train"})
            run.track(epoch_loss_v / n, name="loss_v",
                      epoch=epoch, context={"subset": "train"})

            run.track(model_k.init_zp.mean().item(), name="dc_k",
                      epoch=epoch, context={"subset": "train"})
            run.track(model_k.get_sc().mean().item(), name="scale_k",
                      epoch=epoch, context={"subset": "train"})
            run.track(model_k.get_alpha().mean().item(), name="alpha_k",
                      epoch=epoch, context={"subset": "train"})
            run.track(model_k.get_beta().mean().item(), name="beta_k",
                      epoch=epoch, context={"subset": "train"})
            run.track(model_k.target.mean().item(), name="target_k",
                      epoch=epoch, context={"subset": "train"})
            run.track(model_v.init_zp.mean().item(), name="dc_v",
                      epoch=epoch, context={"subset": "train"})
            run.track(model_v.get_sc().mean().item(), name="scale_v",
                      epoch=epoch, context={"subset": "train"})
            run.track(model_v.get_alpha().mean().item(), name="alpha_v",
                      epoch=epoch, context={"subset": "train"})
            run.track(model_v.get_beta().mean().item(), name="beta_v",
                      epoch=epoch, context={"subset": "train"})
            run.track(model_v.target.mean().item(), name="target_v",
                      epoch=epoch, context={"subset": "train"})

        pbar.set_postfix({
            "loss_out": f"{epoch_loss_output / n:.6f}",
            "loss_wgt": f"{epoch_loss_weights / n:.6f}",
            "loss_k":   f"{epoch_loss_k / n:.6f}",
            "loss_v":   f"{epoch_loss_v / n:.6f}"
        })
    if use_aim:
        run.close()


# Ray remote task, requests 1 GPU
# Ray queues it when GPUs are saturated
@ray.remote(num_gpus=1)
def train_layer_task(layer_idx: int,
                     dataset_path: str,
                     ckpt_k: str,
                     ckpt_v: str,
                     num_hidden_states: int = 4096,
                     batch_size: int = 4,
                     truncation: int = 256,
                     epochs: int = 100,
                     lr: float = 1e-2,
                     grad_clip: float = 1.0,
                     bits: int = 4,
                     scale_min: float = -4.0,
                     scale_max: float = 4.0,
                     atten: str = "llama",
                     checkpoint: bool = False):
    """
    Train ADKV for one layer on the GPU assigned by Ray.
    """
    # Ray sets CUDA_VISIBLE_DEVICES to the assigned GPU; use cuda:0 here
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Calibration file not found: {dataset_path}\n"
            f"Run calibration first."
        )

    ds = torch.load(dataset_path, map_location="cpu",
                    mmap=True, weights_only=True)
    query_dataset = ds["calib_q"].to(
        device=device, dtype=torch.float16, non_blocking=True).contiguous()
    key_dataset = ds["calib_k"].to(
        device=device, dtype=torch.float16, non_blocking=True).contiguous()
    value_dataset = ds["calib_v"].to(
        device=device, dtype=torch.float16, non_blocking=True).contiguous()
    torch.cuda.synchronize(device)

    N, T, C = key_dataset.shape
    if T % truncation != 0:
        logger.warning(
            f"[Layer {layer_idx}] T={T} not divisible by truncation={truncation}; "
            "padding may cause minor numerical error."
        )
        
    logger.info(
        f"[Layer {layer_idx}] Dataset: N={N}, T={T}, C={C}, K mean={key_dataset.mean():.4f} V mean={value_dataset.mean():.4f}")

    dataset = TensorDataset(query_dataset, key_dataset, value_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    model_k = ADKVTrainModel(num_hidden_states, truncation=truncation,
                             bits=bits, log_sc_min=scale_min,
                             log_sc_max=scale_max).to(device)
    model_v = ADKVTrainModel(num_hidden_states, truncation=truncation,
                             bits=bits, log_sc_min=scale_min,
                             log_sc_max=scale_max).to(device)

    if checkpoint:
        model_k.load_state_dict(torch.load(ckpt_k))
        model_v.load_state_dict(torch.load(ckpt_v))

    logger.info(
        f"[Layer {layer_idx}] Starting training: epochs={epochs}, batch_size={batch_size}, device={device}")

    _train(layer_idx, model_k, model_v, dataloader, ALL_ATTENTIONS[atten], device,
           epochs=epochs, lr=lr, grad_clip=grad_clip)

    os.makedirs(os.path.dirname(ckpt_k) if os.path.dirname(
        ckpt_k) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_v) if os.path.dirname(
        ckpt_v) else ".", exist_ok=True)
    torch.save(model_k.state_dict(), ckpt_k)
    torch.save(model_v.state_dict(), ckpt_v)
    logger.info(f"[Layer {layer_idx}] Saved -> {ckpt_k}, {ckpt_v}")

    return {"layer": layer_idx, "ckpt_k": ckpt_k, "ckpt_v": ckpt_v, "status": "done"}
