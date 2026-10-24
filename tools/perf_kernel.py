import math
import logging
import torch

from adkv.kernel.gemv import forward_qk, forward_ov

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NH = 32
DIM = 128
TILE = 4

BATCHES = [1, 4, 16, 32, 64, 128, 256, 512]
SEQ_LENS = [128, 256, 512, 1024, 2048]

SCALE_MIN = math.exp(-4.0)
SCALE_MAX = math.exp(4.0)
WARMUP = 3


def _supergroups(T: int) -> int:
    """Supergroups needed for T cached tokens plus the one appended token."""
    t_packed = (T + 1) // 2
    return (t_packed + TILE - 1) // TILE + 1


def _bench(B: int, T: int) -> float:
    """Average latency (ms) of one decode step: QK -> softmax -> OV."""
    dev = "cuda"
    nsc = _supergroups(T)

    # Decode-step KV cache, already holding T tokens.
    # The cache is filled with a fixed byte pattern (0x5A -> q_int {-3, 2}) so the setup is cheap and
    # the fp16 filter state stays bounded over long sequences.
    k_cache = torch.full((B, NH, nsc, DIM // 4, 4, 4), 0x5A,
                         dtype=torch.uint8, device=dev)
    v_cache = torch.full((B, NH, nsc, DIM // 4, 4, 4), 0x5A,
                         dtype=torch.uint8, device=dev)

    q = torch.randn(B, NH, 1, DIM, dtype=torch.float16, device=dev) * 0.5
    k_new = torch.randn(B, NH, 1, DIM, dtype=torch.float16, device=dev) * 0.5
    v_new = torch.randn(B, NH, 1, DIM, dtype=torch.float16, device=dev) * 0.5

    zp = torch.zeros(NH, DIM, dtype=torch.float16, device=dev)
    sc = torch.ones(NH, DIM, dtype=torch.float16, device=dev)
    alpha = torch.full((NH, DIM), 0.05, dtype=torch.float16, device=dev)
    beta = torch.zeros(NH, DIM, dtype=torch.float16, device=dev)
    tgt = torch.full((NH, DIM), 3.5, dtype=torch.float16, device=dev)

    def step():
        logits = forward_qk(q, k_cache, k_new, zp, sc, alpha, beta, tgt,
                            T, SCALE_MIN, SCALE_MAX)
        attn = torch.softmax(logits, dim=-1)
        forward_ov(attn, v_cache, v_new, zp, sc, alpha, beta, tgt,
                   SCALE_MIN, SCALE_MAX)

    # forward_qk / forward_ov append k_new / v_new at position T, overwriting
    # the same slot every iteration, so the timed workload is identical.
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    iters = max(5, min(100, 2_000_000 // (B * T)))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        step()
        end.record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    ms = times[len(times) // 2]  # median is robust to occasional hiccups
    del k_cache, v_cache, q, k_new, v_new, zp, sc, alpha, beta, tgt
    return ms


def main():
    torch.manual_seed(0)

    # Push the GPU out of its idle power state before timing small batches.
    burn = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    burn_dst = torch.empty_like(burn)
    for _ in range(20):
        burn_dst.copy_(burn)
    torch.cuda.synchronize()
    del burn, burn_dst
    torch.cuda.empty_cache()

    logger.info(
        "Request throughput at 32 layers (req/s): B / (single-layer step latency * 32)")
    logger.info(
        f"Step = forward_qk + softmax + forward_ov;  nh={NH}, DIM={DIM}, fp16")
    header = "B\\T".ljust(8) + "".join(f"{t:>10}" for t in SEQ_LENS)
    logger.info(header)
    logger.info("-" * len(header))

    for B in BATCHES:
        row = []
        for T in SEQ_LENS:
            try:
                ms = _bench(B, T)
                row.append(B / (ms / 1000.0 * 32.0))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row.append(float("nan"))
            torch.cuda.empty_cache()
        logger.info(f"{B:<8}" + "".join(
            f"{v:10.1f}" if v == v else f"{'OOM':>10}" for v in row))

    logger.info("-" * len(header))
    logger.info("requests/s, 32 layers (higher is better)")


if __name__ == "__main__":
    main()
