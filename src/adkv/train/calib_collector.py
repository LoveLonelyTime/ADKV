import logging
import argparse
import os
import threading
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from datasets import load_from_disk, load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def patch_layers(model, layer_indices):
    """
    Register forward hooks on each attention layer in layer_indices.

    Returns:
        captured: dict {layer_idx: {"q": [], "k": [], "v": []}} populated by the hooks.
        handles: list of hook handles to remove() when done.
    """
    captured = {i: {"q": [], "k": [], "v": []} for i in layer_indices}

    # DataParallel replicates the module to each GPU and runs hooks on each replica independently (in separate threads).
    # Use threading.Lock to serialise list.append() across replica threads
    lock = threading.Lock()
    handles = []

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    def make_hook(layer_idx, n_q, n_kv, head_dim):
        def _hook(module, args_in, kwargs_in, output):
            hidden_states = kwargs_in.get("hidden_states")
            position_embeddings = kwargs_in.get("position_embeddings")

            with torch.no_grad():
                B, T, _ = hidden_states.shape

                q_flat = module.q_proj(hidden_states)
                k_flat = module.k_proj(hidden_states)
                v_flat = module.v_proj(hidden_states)

                query_states = q_flat.view(B, T, n_q, head_dim).transpose(1, 2)
                key_states = k_flat.view(B, T, n_kv, head_dim).transpose(1, 2)
                val_states = v_flat.view(B, T, n_kv, head_dim).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin)

                q_out = query_states.transpose(
                    1, 2).reshape(B, T, n_q * head_dim)
                k_out = key_states.transpose(
                    1, 2).reshape(B, T, n_kv * head_dim)
                v_out = val_states.transpose(
                    1, 2).reshape(B, T, n_kv * head_dim)

                q_cpu = q_out.detach().to(dtype=torch.float16).cpu()
                k_cpu = k_out.detach().to(dtype=torch.float16).cpu()
                v_cpu = v_out.detach().to(dtype=torch.float16).cpu()
            with lock:
                captured[layer_idx]["q"].append(q_cpu)
                captured[layer_idx]["k"].append(k_cpu)
                captured[layer_idx]["v"].append(v_cpu)
        return _hook

    for layer_idx in layer_indices:
        self_attn = base_model.model.layers[layer_idx].self_attn
        head_dim = self_attn.head_dim
        n_kv = self_attn.k_proj.out_features // head_dim
        n_q = self_attn.q_proj.out_features // head_dim

        # with_kwargs=True, so the hook receives position_embeddings / attention_mask
        h = self_attn.register_forward_hook(
            make_hook(layer_idx, n_q, n_kv, head_dim), with_kwargs=True)
        handles.append(h)

    q_dim = base_model.model.layers[layer_indices[0]
                                    ].self_attn.q_proj.out_features
    kv_dim = base_model.model.layers[layer_indices[0]
                                     ].self_attn.k_proj.out_features
    logger.info(
        f"Patched layers {layer_indices}, q_dim = {q_dim}, kv_dim = {kv_dim}")

    return captured, handles


def run_chunk(args, model, tokenizer, calib_data, device, chunk):
    """
    Run all calibration batches with hooks on `chunk` layers, save each layer to disk.
    """
    captured, handles = patch_layers(model, chunk)

    # Per-layer accumulators across batches
    all_layers = {i: {"q": [], "k": [], "v": []} for i in chunk}
    n_batches = (len(calib_data) + args.batch_size - 1) // args.batch_size

    try:
        with torch.no_grad():
            for start in range(0, len(calib_data), args.batch_size):
                batch_texts = calib_data[start: start + args.batch_size]

                enc = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                # Clear hook buffers for every layer before each batch
                for i in chunk:
                    captured[i]["q"].clear()
                    captured[i]["k"].clear()
                    captured[i]["v"].clear()

                model(input_ids=input_ids, attention_mask=attention_mask)

                # DataParallel fires the hook once per replica
                # Concat fragments
                for i in chunk:
                    batch_q = torch.cat(captured[i]["q"], dim=0)
                    batch_k = torch.cat(captured[i]["k"], dim=0)
                    batch_v = torch.cat(captured[i]["v"], dim=0)
                    all_layers[i]["q"].append(batch_q)
                    all_layers[i]["k"].append(batch_k)
                    all_layers[i]["v"].append(batch_v)

                batch_idx = start // args.batch_size + 1
                first = chunk[0]
                logger.info(f"batch {batch_idx}/{n_batches}"
                            f" q shape: {all_layers[first]['q'][0].shape}"
                            f" kv shape: {all_layers[first]['k'][0].shape}")
    finally:
        for h in handles:
            h.remove()

    base, ext = os.path.splitext(args.output)
    if not ext:
        ext = ".pt"

    for i in chunk:
        calib_q = torch.cat(all_layers[i]["q"], dim=0)  # [N, T, q_dim]
        calib_k = torch.cat(all_layers[i]["k"], dim=0)  # [N, T, kv_dim]
        calib_v = torch.cat(all_layers[i]["v"], dim=0)

        out_path = f"{base}_{i}{ext}"
        logger.info(f"Layer {i}:")
        logger.info(
            f"calib_k: {calib_k.shape} (min={calib_k.min():.4f}, max={calib_k.max():.4f}, mean={calib_k.mean():.4f}, std={calib_k.std():.4f})")
        logger.info(
            f"calib_v: {calib_v.shape} (min={calib_v.min():.4f}, max={calib_v.max():.4f}, mean={calib_v.mean():.4f}, std={calib_v.std():.4f})")
        torch.save({"calib_q": calib_q, "calib_k": calib_k,
                   "calib_v": calib_v}, out_path)
        logger.info(f"Saved -> {out_path}")

        # Free as we go
        all_layers[i]["q"].clear()
        all_layers[i]["k"].clear()
        all_layers[i]["v"].clear()


def parse_layers_arg(s: str):
    """
    Parse '0-31' (exclusive range) or '5' into a list of layer indices.
    """
    s = s.strip()
    if '-' in s:
        a, b = s.split('-', 1)
        return list(range(int(a), int(b)))
    return [int(s)]


def load_tokenizer(args):
    logger.info("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Llama tokenizer has no dedicated pad token, reuse eos_token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(
            f"pad_token set to eos_token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")

    tokenizer.padding_side = "right"

    return tokenizer


def load_model(args, device):
    logger.info("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager"
    ).to(device).eval()

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        logger.info(f"Using DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)

    return model


def load_calib_data(args):
    logger.info("Loading dataset...")
    if args.calib_data is None:
        # Default wikimedia
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    else:
        ds = load_from_disk(args.calib_data)

    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)

    texts = []
    for example in ds:
        text = example["text"].strip()
        # Skip very short stubs
        if len(text) > args.clip_len:
            texts.append(text)
        if len(texts) >= args.n_samples:
            break

    logger.info(f"Collected {len(texts)} text samples")
    return texts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect calibration data for ADKV."
    )

    parser.add_argument('--model', type=str,
                        default='meta-llama/Llama-2-7b-hf')

    parser.add_argument('--calib_data', type=str, default=None,
                        help="Path to calibration dataset from local disk.")
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clip_len', type=int, default=4096)

    parser.add_argument('--layers', type=str, default='0',
                        help="Collect from a single layer index (e.g. '5') or range (e.g. '0-31', exclusive).")
    parser.add_argument('--layers_per_chunk', type=int, default=4,
                        help="Max number of layers captured simultaneously per forward pass.")

    parser.add_argument('--n_samples', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=128)

    parser.add_argument('--output', type=str, default='calib_llama.pt',
                        help="Output file prefix; each layer is saved as <prefix>_{layer}.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    layer_indices = parse_layers_arg(args.layers)
    if args.layers_per_chunk < 1:
        raise ValueError("layers_per_chunk must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args)
    model = load_model(args, device)
    calib_data = load_calib_data(args)

    chunks = [layer_indices[i:i + args.layers_per_chunk]
              for i in range(0, len(layer_indices), args.layers_per_chunk)]
    logger.info(
        f"Processing {len(layer_indices)} layer(s) in {len(chunks)} chunk(s)")

    for chunk_idx, chunk in enumerate(chunks):
        logger.info(
            f"\n=== Chunk {chunk_idx+1}/{len(chunks)}: layers {chunk} ===")
        run_chunk(args, model, tokenizer, calib_data, device, chunk)


if __name__ == "__main__":
    main()
