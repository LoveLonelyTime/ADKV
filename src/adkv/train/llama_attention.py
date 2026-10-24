import torch
import torch.nn.functional as F

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def llama_calib_attention(query_states: torch.Tensor,
                          key_states: torch.Tensor,
                          value_states: torch.Tensor,
                          attention_mask: torch.Tensor = None,
                          head_dim: int = 128):
    # query_states: [B, T, C_q] (also accepts [T, C_q] - auto-unsqueezed below)
    # key_states: [B, T, C_kv]
    # value_states: [B, T, C_kv]

    B, T, C_q = query_states.shape
    num_attention_heads = C_q // head_dim
    num_key_value_heads = key_states.shape[-1] // head_dim
    num_key_value_groups = num_attention_heads // num_key_value_heads

    # [B, T, C] -> [B, H, T, D]
    query_states = query_states.view(
        B, T, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(
        B, T, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(
        B, T, num_key_value_heads, head_dim).transpose(1, 2)

    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states,
        attn_mask=None, is_causal=True,
    )  # [B, H, T, D]

    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, T, H, D]
    attn_output = attn_output.view(B, T, num_attention_heads * head_dim)  # [B, T, C_q]

    return attn_output, None
