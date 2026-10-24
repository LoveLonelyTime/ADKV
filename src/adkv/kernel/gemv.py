import os

import torch
from torch.utils.cpp_extension import load

# Build extension
_dir = os.path.dirname(os.path.abspath(__file__))
adkv_gemv = load(
    name="adkv_gemv",
    sources=[os.path.join(_dir, "gemv.cpp"), os.path.join(_dir, "gemv.cu")],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)


def forward_qk(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    k_new: torch.Tensor,
    k_zp: torch.Tensor,
    k_sc: torch.Tensor,
    k_alpha: torch.Tensor,
    k_beta: torch.Tensor,
    k_t: torch.Tensor,
    T: int,
    scale_min: float,
    scale_max: float,
) -> torch.Tensor:
    """Decode-phase QK^T over the nibble-packed ADKV cache.

    Computes ``o[b, h, t] = sum_c q[b, h, c] * k_deq[b, h, t, c]`` for the
    tokens already cached, then quantises ``k_new`` with the filter state after
    token ``T-1`` and appends it at position ``T`` (in place).  The logit of the
    appended token against the same query is computed too and returned as the
    last element.

    Args:
        q:       ``[B, nh, 1, DIM]`` float16 query.
        k_cache: ``[B, nh, n_super_cap, DIM/4, 4, 4]`` uint8 uint4-tiled cache.
                 Must be contiguous; token ``T`` is written in place.
        k_new:   ``[B, nh, 1, DIM]`` float16 token to append.
        k_zp, k_sc, k_alpha, k_beta, k_t:
                 ``[nh, DIM]`` float16 ADKV parameters.
        T:       current sequence length (tokens ``0 .. T-1`` are cached).
        scale_min, scale_max: clamp bounds (kept for API compatibility).

    Returns:
        ``[B, nh, T + 1]`` float16 attention logits: the ``T`` cached tokens
        followed by the self logit of the appended token.
    """
    return adkv_gemv.forward_qk(
        q, k_cache, k_new,
        k_zp, k_sc, k_alpha, k_beta, k_t,
        T, scale_min, scale_max,
    )


def forward_ov(
    attn: torch.Tensor,
    v_cache: torch.Tensor,
    v_new: torch.Tensor,
    v_zp: torch.Tensor,
    v_sc: torch.Tensor,
    v_alpha: torch.Tensor,
    v_beta: torch.Tensor,
    v_t: torch.Tensor,
    scale_min: float,
    scale_max: float,
) -> torch.Tensor:
    """Decode-phase OV over the nibble-packed ADKV cache.

    Computes ``o[b, h, c] = sum_t attn[b, h, t] * v_deq[b, h, t, c]`` for the
    tokens already cached, then quantises ``v_new`` with the filter state after
    token ``T-1`` and appends it at position ``T`` (in place).  ``attn`` holds
    the ``T`` cached weights plus the self weight of the appended token, i.e.
    ``attn.shape[2] == T + 1``; the self weight is folded in as
    ``attn[T] * v_deq_new``.

    Args:
        attn:    ``[B, nh, T + 1]`` float16 attention weights (last one is the
                 appended token's self weight).
        v_cache: ``[B, nh, n_super_cap, DIM/4, 4, 4]`` uint8 uint4-tiled cache.
                 Must be contiguous; token ``T`` is written in place.
        v_new:   ``[B, nh, 1, DIM]`` float16 token to append.
        v_zp, v_sc, v_alpha, v_beta, v_t:
                 ``[nh, DIM]`` float16 ADKV parameters.
        scale_min, scale_max: clamp bounds (kept for API compatibility).

    Returns:
        ``[B, nh, DIM]`` float16 attention output.
    """
    out = adkv_gemv.forward_ov(
        attn, v_cache, v_new,
        v_zp, v_sc, v_alpha, v_beta, v_t,
        scale_min, scale_max,
    )
    return out.view(attn.shape[0], attn.shape[1], -1)


def encode_kv(
    kv_cache: torch.Tensor,
    kv_new: torch.Tensor,
    kv_zp: torch.Tensor,
    kv_sc: torch.Tensor,
    kv_alpha: torch.Tensor,
    kv_beta: torch.Tensor,
    kv_t: torch.Tensor,
    scale_min: float,
    scale_max: float,
) -> torch.Tensor:
    """Prefill: quantise tokens ``[0, T)`` into the uint4-tiled KV cache.

    Replays the ADKV filter from ``(kv_zp, kv_sc)`` and writes every token
    ``t`` to byte_row ``t/2``, low nibble for even ``t`` and high nibble for odd
    ``t``.  Purely in place.

    Args:
        kv_cache: ``[B, nh, n_super_cap, DIM/4, 4, 4]`` uint8 uint4-tiled cache.
                  Must be contiguous; it is overwritten for tokens ``[0, T)``.
        kv_new:   ``[B, nh, T, DIM]`` float16 raw tokens.
        kv_zp, kv_sc, kv_alpha, kv_beta, kv_t:
                  ``[nh, DIM]`` float16 ADKV parameters.
        scale_min, scale_max: clamp bounds (kept for API compatibility).

    Returns:
        The same ``kv_cache`` tensor (modified in place).
    """
    return adkv_gemv.encode_kv(
        kv_cache, kv_new,
        kv_zp, kv_sc, kv_alpha, kv_beta, kv_t,
        scale_min, scale_max,
    )
