from collections.abc import Callable
import math
from dataclasses import dataclass

import torch
from torch import nn
import triton
import triton.language as tl

from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

SCALE_MIN = math.exp(-4.0)   # = 0.0183
SCALE_MAX = math.exp(4.0)    # = 54.6
BITS = 4
QUANT_UPPER = 2 ** (BITS - 1) - 1
QUANT_LOWER = -2 ** (BITS - 1)

# Per-(head, channel) ADKV parameters.  The same five fields exist for K and V.
ADKV_PARAM_FIELDS = ("zp_init", "scale_init", "alpha", "beta", "quant_target")


@triton.jit
def _adkv_quant_kernel(
    key_buf_ptr, val_buf_ptr,
    key_dc_ptr, key_scale_ptr,
    val_dc_ptr, val_scale_ptr,
    key_alpha_ptr, key_beta_ptr, key_qt_ptr,
    val_alpha_ptr, val_beta_ptr, val_qt_ptr,
    start,
    stride_b, stride_t,
    scale_min: tl.constexpr, scale_max: tl.constexpr,
    C, BLOCK_C: tl.constexpr, N_STEPS: tl.constexpr,
    QUANT_UPPER: tl.constexpr, QUANT_LOWER: tl.constexpr,
):
    """
    grid = (B, cdiv(C, BLOCK_C)).
    Each program owns one (batch, channel-tile) pair and serially iterates
    over N_STEPS tokens.

    Each per-step load/store accesses BLOCK_C contiguous elements — fully
    coalesced.  State tensors (dc, scale) live in registers across all steps.
    tl.static_range unrolls the loop at compile time, enabling the compiler to
    pipeline/schedule the N_STEPS iterations efficiently.
    """
    b_id = tl.program_id(0)
    c_id = tl.program_id(1)
    offs_c = tl.arange(0, BLOCK_C) + c_id * BLOCK_C   # [BLOCK_C]
    mask_c = offs_c < C

    # --- load per-(batch, channel) states into registers ---
    bc_base = b_id * C + offs_c
    k_dc = tl.load(key_dc_ptr + bc_base, mask=mask_c, other=0.0)
    k_scale = tl.load(key_scale_ptr + bc_base, mask=mask_c, other=1.0)
    k_alpha = tl.load(key_alpha_ptr + offs_c,  mask=mask_c, other=0.0)
    k_beta = tl.load(key_beta_ptr + offs_c,  mask=mask_c, other=0.0)
    k_qt = tl.load(key_qt_ptr + offs_c,  mask=mask_c, other=0.0)

    v_dc = tl.load(val_dc_ptr + bc_base, mask=mask_c, other=0.0)
    v_scale = tl.load(val_scale_ptr + bc_base, mask=mask_c, other=1.0)
    v_alpha = tl.load(val_alpha_ptr + offs_c,  mask=mask_c, other=0.0)
    v_beta = tl.load(val_beta_ptr + offs_c,  mask=mask_c, other=0.0)
    v_qt = tl.load(val_qt_ptr + offs_c,  mask=mask_c, other=0.0)

    eps = 1e-8

    # --- statically-unrolled loop over tokens ---
    # tl.static_range unrolls at compile time so the compiler can pipeline
    # iterations; each load/store is BLOCK_C contiguous elements (coalesced).
    for step in tl.static_range(N_STEPS):
        buf_offs = b_id * stride_b + \
            (start + step) * stride_t + offs_c  # [BLOCK_C]

        # ---- Key ----
        k = tl.load(key_buf_ptr + buf_offs, mask=mask_c, other=0.0)
        residual = k - k_dc
        scale_safe = tl.maximum(k_scale, eps)
        q = tl.extra.cuda.libdevice.rint(tl.div_rn(residual, scale_safe))
        q = tl.clamp(q, QUANT_LOWER, QUANT_UPPER)
        residual_hat = q * scale_safe
        tl.store(key_buf_ptr + buf_offs, residual_hat + k_dc, mask=mask_c)
        k_dc = k_dc + k_alpha * residual_hat
        error = tl.abs(q) - k_qt
        k_scale = tl.clamp(k_scale + k_scale * k_beta *
                           error, scale_min, scale_max)

        # ---- Value ----
        v = tl.load(val_buf_ptr + buf_offs, mask=mask_c, other=0.0)
        residual = v - v_dc
        scale_safe = tl.maximum(v_scale, eps)
        q = tl.extra.cuda.libdevice.rint(tl.div_rn(residual, scale_safe))
        q = tl.clamp(q, QUANT_LOWER, QUANT_UPPER)
        residual_hat = q * scale_safe
        tl.store(val_buf_ptr + buf_offs, residual_hat + v_dc, mask=mask_c)
        v_dc = v_dc + v_alpha * residual_hat
        error = tl.abs(q) - v_qt
        v_scale = tl.clamp(v_scale + v_scale * v_beta *
                           error, scale_min, scale_max)

    # --- write back updated states ---
    tl.store(key_dc_ptr + bc_base, k_dc, mask=mask_c)
    tl.store(key_scale_ptr + bc_base, k_scale, mask=mask_c)
    tl.store(val_dc_ptr + bc_base, v_dc, mask=mask_c)
    tl.store(val_scale_ptr + bc_base, v_scale, mask=mask_c)


@dataclass
class ADKVConfig:
    zp_init: torch.Tensor
    scale_init: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    quant_target: torch.Tensor


# Params
INIT_SEQ_LEN = 128
CAP_LEN = 128
GROUP_SIZE = 32


class KVCacheLayer_ADKV:
    def __init__(self):
        self.key_dc = None
        self.key_scale = None
        self.value_dc = None
        self.value_scale = None
        self._key_buf = None
        self._value_buf = None
        self._quant_len = 0
        self._recent_len = 0

    def get_seq_length(self) -> int:
        return self._recent_len

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def _init_buffers(self, B: int, C: int, device: torch.device, dtype: torch.dtype):
        self._key_buf = torch.empty(
            B, INIT_SEQ_LEN, C, device=device, dtype=dtype)
        self._value_buf = torch.empty(
            B, INIT_SEQ_LEN, C, device=device, dtype=dtype)
        self._recent_len = 0
        self._quant_len = 0

    def _cap_buffers(self, T: int):
        B, max_len, C = self._key_buf.shape
        req_len = T - (max_len - self._recent_len)
        if (req_len <= 0):
            return
        incr_len = max(req_len, CAP_LEN)
        new_key_buf = torch.empty(
            B, max_len + incr_len, C, device=self._key_buf.device, dtype=self._key_buf.dtype)
        new_value_buf = torch.empty(
            B, max_len + incr_len, C, device=self._value_buf.device, dtype=self._value_buf.dtype)
        # Copy
        new_key_buf[:, 0:self._recent_len,
                    :] = self._key_buf[:, 0:self._recent_len, :]
        new_value_buf[:, 0:self._recent_len,
                      :] = self._value_buf[:, 0:self._recent_len, :]
        self._key_buf = new_key_buf
        self._value_buf = new_value_buf

    def _append_tokens(self, key_states: torch.Tensor, value_states: torch.Tensor):
        # You can enable Attention Sink to improve ADKV
        # self._quant_len = 2
        T = key_states.shape[1]
        self._key_buf[:, self._recent_len:self._recent_len + T, :] = key_states
        self._value_buf[:, self._recent_len:self._recent_len +
                        T, :] = value_states
        self._recent_len += T

    def _quant(self, start: int, key_alpha: torch.Tensor, key_beta: torch.Tensor, key_quant_target: torch.Tensor, value_alpha: torch.Tensor, value_beta: torch.Tensor, value_quant_target: torch.Tensor, n_step: int = 32):
        B, _, C = self._key_buf.shape

        # key_buf / value_buf strides: (stride_b, stride_t, stride_c==1)
        stride_b = self._key_buf.stride(0)
        stride_t = self._key_buf.stride(1)

        BLOCK_C = 32
        def grid(meta): return (B, triton.cdiv(C, meta['BLOCK_C']))

        _adkv_quant_kernel[grid](
            self._key_buf, self._value_buf,
            self.key_dc, self.key_scale,
            self.value_dc, self.value_scale,
            key_alpha, key_beta, key_quant_target,
            value_alpha, value_beta, value_quant_target,
            start,
            stride_b, stride_t,
            float(SCALE_MIN), float(SCALE_MAX),
            C, BLOCK_C=BLOCK_C, N_STEPS=n_step,
            QUANT_LOWER=QUANT_LOWER, QUANT_UPPER=QUANT_UPPER,
            enable_fp_fusion=False
        )

    def _quantize_new_tokens(self, key_alpha: torch.Tensor, key_beta: torch.Tensor, key_quant_target: torch.Tensor, value_alpha: torch.Tensor, value_beta: torch.Tensor, value_quant_target: torch.Tensor):
        # Quantize on write, exactly like the ADKV kernels: every appended
        # token is quantized immediately, with no full-precision tail.
        # Tokens are processed in GROUP_SIZE tiles only to bound the kernel's
        # statically-unrolled (compile-time) loop length.
        while self._recent_len - self._quant_len > GROUP_SIZE:
            self._quant(self._quant_len, key_alpha, key_beta,
                        key_quant_target, value_alpha, value_beta, value_quant_target)
            self._quant_len += GROUP_SIZE

        residual = self._recent_len - self._quant_len
        if residual:
            self._quant(self._quant_len, key_alpha, key_beta,
                        key_quant_target, value_alpha, value_beta, value_quant_target, residual)
            self._quant_len += residual

    @property
    def key_states(self) -> torch.Tensor | None:
        if self._recent_len == 0:
            return None
        return self._key_buf[:, :self._recent_len, :]

    @property
    def value_states(self) -> torch.Tensor | None:
        if self._recent_len == 0:
            return None
        return self._value_buf[:, :self._recent_len, :]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               key_zp_init: torch.Tensor,
               key_scale_init: torch.Tensor,
               key_alpha: torch.Tensor,
               key_beta: torch.Tensor,
               key_quant_target: torch.Tensor,
               value_zp_init: torch.Tensor,
               value_scale_init: torch.Tensor,
               value_alpha: torch.Tensor,
               value_beta: torch.Tensor,
               value_quant_target: torch.Tensor,
               ) -> tuple[torch.Tensor, torch.Tensor]:
        # key_states / value_states: [B, T, C]
        B, T, C = key_states.shape

        # Allocate buffers on first call
        if self._key_buf is None:
            self._init_buffers(B, C, key_states.device, key_states.dtype)

        # Init adaptive filter states on first call
        if self.key_dc is None:
            self.key_dc = key_zp_init.repeat(B, 1)
            self.key_scale = key_scale_init.repeat(B, 1)
            self.value_dc = value_zp_init.repeat(B, 1)
            self.value_scale = value_scale_init.repeat(B, 1)

        # Check capacity
        self._cap_buffers(T)

        # Append
        self._append_tokens(key_states, value_states)

        # Quantize everything appended in this call (prefill or decode).
        self._quantize_new_tokens(
            key_alpha, key_beta, key_quant_target, value_alpha, value_beta, value_quant_target)

        return self.key_states, self.value_states


class KVCache_ADKV(list):
    def __init__(self, num_layers):
        for _ in range(num_layers):
            self.append(KVCacheLayer_ADKV())

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self[layer_idx].get_seq_length()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        return self[layer_idx].get_mask_sizes(cache_position)


class LlamaAttention_ADKV(LlamaAttention):
    """Llama attention backed by the ADKV-quantized KV cache.

    ``prefill_full_precision`` selects the evaluation contract:

    * ``True``  -- fused prefill: attention uses the full-precision K/V just
      computed, while the quantized cache is only written.  Prompt-only
      workloads are insensitive to quantization under this contract.
    * ``False`` -- cache-read prefill: attention reads K/V back from the
      quantized cache, so quantization error also affects prefill and
      propagates across layers.

    Decode always reads the quantized cache.
    """

    # Buffers that must stay FP32 even when the model is cast to FP16/BF16.
    _FP32_BUFFERS = frozenset(
        f"{prefix}_{field}"
        for prefix in ("key", "value")
        for field in ADKV_PARAM_FIELDS
    )

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        # [num_kv_heads * head_dim] parameters, one entry per (head, channel).
        n = config.num_key_value_heads * self.head_dim
        for name in self._FP32_BUFFERS:
            # NOTE: passed directly to the Triton kernel, which expects FP32.
            self.register_buffer(
                name, torch.empty(n, dtype=torch.float32), persistent=False)

    def _apply(self, fn, recurse=True):
        # Temporarily remove FP32 buffers so the parent _apply (e.g. model.half(),
        # model.to(device)) cannot change their dtype.  Device/layout changes are
        # still applied; only dtype promotion back to FP32 is enforced.
        saved = {name: self._buffers.pop(name) for name in self._FP32_BUFFERS}
        super()._apply(fn, recurse=recurse)
        for name, buf in saved.items():
            if buf is not None:
                buf = fn(buf)                      # allow device move
                if buf.dtype != torch.float32:
                    buf = buf.to(torch.float32)    # revert any dtype cast
            self._buffers[name] = buf
        return self

    def _adkv_params(self, prefix: str) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, f"{prefix}_{field}")
                     for field in ADKV_PARAM_FIELDS)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = True,
        prefill_full_precision: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(
            hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        if past_key_values is not None:
            if not isinstance(past_key_values, KVCache_ADKV):
                raise TypeError(
                    "ADKV attention requires a KVCache_ADKV, got "
                    f"{type(past_key_values).__name__}")
            layer_cache = past_key_values[self.layer_idx]

            # [B, H, T, C] -> [B, T, H * C], quantized into the cache on write.
            B, H, T, C = key_states.shape
            ori_dtype = key_states.dtype
            hist_len = layer_cache.get_seq_length()
            layer_cache.update(
                key_states.transpose(1, 2).reshape(B, T, H * C).float(),
                value_states.transpose(1, 2).reshape(B, T, H * C).float(),
                *self._adkv_params("key"), *self._adkv_params("value"))

            if T > 1 and prefill_full_precision:
                # Prefill (A): attend with the full-precision K/V we just
                # computed.  For chunked prefill, prepend the cached history
                # so attention still covers the whole prefix.
                if hist_len:
                    key_states = torch.cat([
                        layer_cache.key_states[:, :hist_len].view(
                            B, hist_len, H, C).transpose(1, 2).to(ori_dtype),
                        key_states], dim=2)
                    value_states = torch.cat([
                        layer_cache.value_states[:, :hist_len].view(
                            B, hist_len, H, C).transpose(1, 2).to(ori_dtype),
                        value_states], dim=2)
            else:
                # Prefill (B) / decode: read the whole cached sequence back,
                # including the quantized history.
                T_cached = layer_cache.get_seq_length()
                key_states = layer_cache.key_states.view(
                    B, T_cached, H, C).transpose(1, 2).to(ori_dtype)
                value_states = layer_cache.value_states.view(
                    B, T_cached, H, C).transpose(1, 2).to(ori_dtype)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer_ADKV(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LlamaAttention_ADKV(
            config=config, layer_idx=layer_idx)


class LlamaModel_ADKV(LlamaModel):
    """LlamaModel whose self-attention layers use the ADKV cache."""

    _no_split_modules = ["LlamaDecoderLayer_ADKV"]

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_ADKV(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )
        # The base layers built above were discarded; re-initialize so the new
        # ones follow the config's initializer_range instead of nn.Linear's.
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Resolve use_cache here (before the base forward's check_model_inputs
        # decorator) so the ADKV cache class is chosen consistently and no
        # DynamicCache is ever created.
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", True)
        if use_cache and past_key_values is None:
            past_key_values = KVCache_ADKV(self.config.num_hidden_layers)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )


class LlamaForCausalLM_ADKV(LlamaForCausalLM):
    _no_split_modules = ["LlamaDecoderLayer_ADKV"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel_ADKV(config)
        # `super().__init__` tied lm_head to the discarded model's embedding.
        self.tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        prefill_full_precision: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Forward pass.

        Args:
            prefill_full_precision:
                * ``True``  -- fused-prefill contract: prefill attention uses
                  the full-precision K/V; only decode reads the quantized cache.
                * ``False`` -- cache-read contract: prefill attention also
                  reads the quantized cache, so prefill-only benchmarks are
                  sensitive to quantization.
                If omitted, ``config.prefill_full_precision`` is used (default
                ``True``); set that attribute to switch contracts for callers
                that cannot pass keyword arguments (e.g. lm-eval).
        """
        if prefill_full_precision is None:
            prefill_full_precision = getattr(
                self.config, "prefill_full_precision", True)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            prefill_full_precision=prefill_full_precision,
            **kwargs,
        )

    def load_adkv_config(self, adkv_config: list[tuple[ADKVConfig, ADKVConfig]]):
        """Fill the per-layer K/V ADKV parameters from ``ADKVConfig`` pairs."""
        for layer_idx, configs in enumerate(adkv_config):
            attn = self.model.layers[layer_idx].self_attn
            for prefix, cfg in zip(("key", "value"), configs):
                for field in ADKV_PARAM_FIELDS:
                    getattr(attn, f"{prefix}_{field}").copy_(
                        getattr(cfg, field))

    def _prepare_cache_for_generation(
        self,
        *vargs,
        **kwargs,
    ):
        # Leave past_key_values unset: LlamaModel_ADKV.forward creates the ADKV
        # cache lazily, and HF's default DynamicCache is incompatible with it.
        return None
