from collections.abc import Callable
import math
from dataclasses import dataclass
import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from adkv.kernel.gemv import encode_kv, forward_ov, forward_qk

SCALE_MIN = math.exp(-4.0)  # = 0.0183
SCALE_MAX = math.exp(4.0)  # = 54.6

# Per-(head, channel) ADKV parameters.  The same five fields exist for K and V.
ADKV_PARAM_FIELDS = ("zp_init", "scale_init", "alpha", "beta", "quant_target")


@dataclass
class ADKVConfig:
    zp_init: torch.Tensor
    scale_init: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    quant_target: torch.Tensor


# Params
INIT_SEQ_LEN = 64
CAP_LEN = 64


class KVCacheLayer_ADKV:
    """Per-layer ADKV KV cache.

    K/V tokens are stored quantized in the uint4-tiled layout consumed by the
    ADKV kernels::

        [B, H, n_super, C // 4, 4, 4]  uint8

    One supergroup holds 4 byte_rows = 8 tokens.  The cache starts with
    ``INIT_SEQ_LEN`` token slots and grows by ``CAP_LEN`` slots whenever a new
    token would not fit.
    """

    # Buffers
    _key_buf: torch.Tensor = None
    _value_buf: torch.Tensor = None
    _quant_len: int = 0
    _cap_len: int = 0

    def __init__(self):
        self._key_buf = None
        self._value_buf = None
        self._quant_len = 0
        self._cap_len = 0
        self._B = 0
        self._H = 0
        self._C = 0
        self._device = None

    def get_seq_length(self) -> int:
        return self._quant_len

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    # Storage management
    @staticmethod
    def _n_super(tokens: int) -> int:
        return (tokens + 7) // 8

    def _alloc(self, cap_tokens: int):
        n_super = self._n_super(cap_tokens)
        self._cap_len = n_super * 8
        shape = (self._B, self._H, n_super, self._C // 4, 4, 4)
        self._key_buf = torch.zeros(
            shape, dtype=torch.uint8, device=self._device)
        self._value_buf = torch.zeros(
            shape, dtype=torch.uint8, device=self._device)

    def _init_buffers(self, B: int, H: int, C: int, device: torch.device):
        if C not in (128, 256, 512):
            raise ValueError(
                f"ADKV kernels support head_dim in (128, 256, 512), got {C}")
        self._B, self._H, self._C, self._device = B, H, C, device
        self._quant_len = 0
        self._alloc(INIT_SEQ_LEN)

    def _cap_buffers(self, T: int):
        """
        Grow by CAP_LEN slots until at least ``T`` tokens fit.
        """
        if T <= self._cap_len:
            return
        new_cap = self._cap_len
        while new_cap < T:
            new_cap += CAP_LEN
        n_super = self._n_super(new_cap)
        new_cap = n_super * 8

        shape = (self._B, self._H, n_super, self._C // 4, 4, 4)
        key = torch.zeros(shape, dtype=torch.uint8, device=self._device)
        value = torch.zeros(shape, dtype=torch.uint8, device=self._device)
        old_super = self._key_buf.shape[2]
        key[:, :, :old_super].copy_(self._key_buf)
        value[:, :, :old_super].copy_(self._value_buf)
        self._key_buf, self._value_buf = key, value
        self._cap_len = new_cap

    # Quantization
    def _append_tokens(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       key_zp_init: torch.Tensor,
                       key_scale_init: torch.Tensor,
                       key_alpha: torch.Tensor,
                       key_beta: torch.Tensor,
                       key_quant_target: torch.Tensor,
                       value_zp_init: torch.Tensor,
                       value_scale_init: torch.Tensor,
                       value_alpha: torch.Tensor,
                       value_beta: torch.Tensor,
                       value_quant_target: torch.Tensor):
        """
        Quantize ``[B, H, T, C]`` key/value tokens into the packed caches.
        """
        H, C = self._H, self._C
        key = key_states.to(torch.float16).contiguous()
        value = value_states.to(torch.float16).contiguous()

        encode_kv(self._key_buf, key,
                  key_zp_init.view(H, C), key_scale_init.view(H, C),
                  key_alpha.view(H, C), key_beta.view(H, C),
                  key_quant_target.view(H, C),
                  SCALE_MIN, SCALE_MAX)
        encode_kv(self._value_buf, value,
                  value_zp_init.view(H, C), value_scale_init.view(H, C),
                  value_alpha.view(H, C), value_beta.view(H, C),
                  value_quant_target.view(H, C),
                  SCALE_MIN, SCALE_MAX)
        self._quant_len += key.shape[2]

    @property
    def key_states(self) -> torch.Tensor | None:
        """Packed uint8 K cache (uint4-tiled), or None before allocation."""
        return self._key_buf

    @property
    def value_states(self) -> torch.Tensor | None:
        """Packed uint8 V cache (uint4-tiled), or None before allocation."""
        return self._value_buf

    # Prefill
    def prefill(self,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                key_zp_init: torch.Tensor,
                key_scale_init: torch.Tensor,
                key_alpha: torch.Tensor,
                key_beta: torch.Tensor,
                key_quant_target: torch.Tensor,
                value_zp_init: torch.Tensor,
                value_scale_init: torch.Tensor,
                value_alpha: torch.Tensor,
                value_beta: torch.Tensor,
                value_quant_target: torch.Tensor):
        """
        Quantize the prompt into the cache.
        No attention is computed here.
        """
        B, H, T, C = key_states.shape
        if value_states.shape != key_states.shape:
            raise ValueError("key/value shapes differ")

        if self._key_buf is None:
            self._init_buffers(B, H, C, key_states.device)
        elif (B, H, C) != (self._B, self._H, self._C):
            raise ValueError("cache was created for a different (B, H, C)")

        if T == 0:
            return
        if self._quant_len != 0:
            raise NotImplementedError(
                "ADKV prefill encodes from token 0; chunked prefill is not supported by the kernel")
        self._cap_buffers(T)
        self._append_tokens(
            key_states, value_states,
            key_zp_init, key_scale_init, key_alpha, key_beta, key_quant_target,
            value_zp_init, value_scale_init, value_alpha, value_beta,
            value_quant_target)

    # Decode
    def decode(self,
               query_states: torch.Tensor,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
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
               attention_mask: torch.Tensor | None = None,
               scaling: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Attend over the cached tokens, append the new K/V, return the output.

        ``query_states`` / ``key_states`` / ``value_states`` are ``[B, H, 1, C]``.
        The QK/OV kernels attend to the ``ctx_len`` cached tokens and also fold in
        the self logit/value of the appended token, so the returned weights have
        length ``ctx_len + 1`` and this is a standard causal decode step.
        """
        B, Hq, Tq, C = query_states.shape
        _, H, Tk, _ = key_states.shape
        if Tq != 1 or Tk != 1:
            raise ValueError(
                "decode expects exactly one query/key/value token")
        if Hq != H:
            raise ValueError(
                "ADKV decode supports MHA "
                "(num_attention_heads == num_key_value_heads)")

        if self._key_buf is None:
            self._init_buffers(B, H, C, query_states.device)
        elif (B, H, C) != (self._B, self._H, self._C):
            raise ValueError("cache was created for a different (B, H, C)")

        self._cap_buffers(self._quant_len + 1)
        ctx_len = self._quant_len

        q = query_states.to(torch.float16).contiguous()
        k = key_states.to(torch.float16).contiguous()
        v = value_states.to(torch.float16).contiguous()

        k_zp = key_zp_init.view(H, C)
        k_sc = key_scale_init.view(H, C)
        k_a = key_alpha.view(H, C)
        k_b = key_beta.view(H, C)
        k_t = key_quant_target.view(H, C)
        v_zp = value_zp_init.view(H, C)
        v_sc = value_scale_init.view(H, C)
        v_a = value_alpha.view(H, C)
        v_b = value_beta.view(H, C)
        v_t = value_quant_target.view(H, C)

        # ctx_len cached tokens + the self logit of the token being appended.
        logits = forward_qk(q, self._key_buf, k, k_zp, k_sc, k_a, k_b, k_t,
                            ctx_len, SCALE_MIN, SCALE_MAX)
        logits = logits.float() * scaling
        if attention_mask is not None:
            mask = (attention_mask[:, :, 0, :]
                    if attention_mask.dim() == 4 else attention_mask)
            logits = logits + mask.float()
        weights = torch.softmax(logits, dim=-1).to(torch.float16)

        out = forward_ov(weights, self._value_buf, v,
                         v_zp, v_sc, v_a, v_b, v_t, SCALE_MIN, SCALE_MAX)
        self._quant_len = ctx_len + 1
        return out.to(query_states.dtype), weights


class KVCache_ADKV(list):
    def __init__(self, num_layers):
        for _ in range(num_layers):
            self.append(KVCacheLayer_ADKV())

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self[layer_idx].get_seq_length()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        return self[layer_idx].get_mask_sizes(cache_position)


class LlamaAttention_ADKV(LlamaAttention):
    """Llama attention with ADKV-quantized decoding.

    Prefill follows the regular (eager or backend) attention path while also
    quantizing K/V into the ADKV cache.  Decode (one query token) is fused into
    the ADKV QK/OV kernels and returns the attention output directly.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        # [num_kv_heads * head_dim] fp16 zero-point / scale / EMA parameters,
        # one entry per (head, channel).  Populated by load_adkv_config().
        n = config.num_key_value_heads * self.head_dim
        for prefix in ("key", "value"):
            for field in ADKV_PARAM_FIELDS:
                self.register_buffer(
                    f"{prefix}_{field}", torch.empty(n, dtype=torch.float16),
                    persistent=False)

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

        layer_cache = None
        if use_cache and past_key_values is not None:
            if not isinstance(past_key_values, KVCache_ADKV):
                raise TypeError(
                    "ADKV attention requires a KVCache_ADKV, got "
                    f"{type(past_key_values).__name__}")
            layer_cache = past_key_values[self.layer_idx]

        if layer_cache is not None and query_states.shape[2] == 1:
            # Decode: attend over the quantized cache, append the new K/V and
            # return the attention output.
            attn_output, attn_weights = layer_cache.decode(
                query_states, key_states, value_states,
                *self._adkv_params("key"), *self._adkv_params("value"),
                attention_mask=attention_mask, scaling=self.scaling)
            attn_output = self.o_proj(
                attn_output.reshape(*input_shape, -1).contiguous())
            return attn_output, attn_weights

        # Prompt (or no cache): dense attention over the incoming tokens.  When
        # a cache is present, store the prompt quantized for later decoding.
        if layer_cache is not None:
            layer_cache.prefill(
                key_states, value_states,
                *self._adkv_params("key"), *self._adkv_params("value"))

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

        attn_output = self.o_proj(
            attn_output.reshape(*input_shape, -1).contiguous())
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
        # HF's check_model_inputs decorator (applied by the base forward) would
        # fill None from config.use_cache; resolve it here first so the cache
        # class is chosen consistently and no DynamicCache is ever created.
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
