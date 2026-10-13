import math
import torch
import torch.nn as nn


class IntQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sc, upper_limit, lower_limit):
        eps = 1e-8
        sc_safe = sc.clamp(min=eps)

        # Quantize and clamp
        q = torch.clamp(torch.round(x / sc_safe),
                        min=lower_limit, max=upper_limit)

        # Dequantize
        x_hat = q * sc_safe

        ctx.save_for_backward(x, sc_safe, q)
        ctx.lower_limit = lower_limit
        ctx.upper_limit = upper_limit
        return x_hat, q

    @staticmethod
    def backward(ctx, grad_x_hat, grad_q):
        x, sc, q = ctx.saved_tensors
        lower_limit = ctx.lower_limit
        upper_limit = ctx.upper_limit

        # LSQ-style gradient for sc
        # See also: https://arxiv.org/pdf/1902.08153
        x_scaled = x / sc
        grad_x = torch.where((x_scaled <= upper_limit) & (
            x_scaled >= lower_limit), grad_x_hat, 0.0)

        lsq_norm = 1.0 / upper_limit
        grad_sc = q - x_scaled
        grad_sc = torch.where(
            x_scaled > upper_limit, upper_limit, grad_sc)
        grad_sc = torch.where(
            x_scaled < lower_limit, lower_limit, grad_sc)

        return grad_x, grad_x_hat * grad_sc * lsq_norm, None, None


@torch.compile(dynamic=False)
def _adkv_chunk_forward(kv_cache_chunk, current_zp, current_sc,
                        alpha, beta, target, upper_limit, lower_limit,
                        sc_min, sc_max, output_states):
    """
    Parameters:
        kv_cache_chunk: [B, chunk, C]
        current_zp: [B, C]
        current_sc: [B, C]
        alpha/beta/target: [C] (broadcast over batch)
    Returns:
        outputs: [B, chunk, C]
        current_zp: [B, C]
        current_sc: [B, C]
    """
    chunk_size = kv_cache_chunk.shape[1]
    outputs = []
    zp_seq = []
    sc_seq = []
    for t in range(chunk_size):
        if output_states:
            zp_seq.append(current_zp.clone().detach())
            sc_seq.append(current_sc.clone().detach())

        residual = kv_cache_chunk[:, t] - current_zp         # [B, C]
        residual_hat, q = IntQuantSTE.apply(
            residual, current_sc, upper_limit, lower_limit)  # [B, C], [B, C]
        output = residual_hat + current_zp                   # [B, C]
        outputs.append(output)

        # Update states (alpha-beta filter)
        current_zp = current_zp + alpha * residual_hat       # [B, C]
        error = torch.abs(q) - target                        # [B, C]
        current_sc = torch.clamp(
            current_sc + current_sc * beta * error,
            min=sc_min, max=sc_max)                          # [B, C]

    if output_states:
        return torch.stack(outputs, dim=1), current_zp, current_sc, torch.stack(zp_seq, dim=1), torch.stack(sc_seq, dim=1)
    else:
        return torch.stack(outputs, dim=1), current_zp, current_sc, None, None


class ADKVTrainModel(nn.Module):
    def __init__(self, num_hidden_states, truncation=32, bits=4, log_sc_min=-4.0, log_sc_max=4.0):
        super().__init__()

        self.num_hidden_states = num_hidden_states
        self.truncation = truncation
        self.bits = bits
        self.upper_limit = 2 ** (bits - 1) - 1
        self.lower_limit = -2 ** (bits - 1)
        self.sc_min = math.exp(log_sc_min)
        self.sc_max = math.exp(log_sc_max)

        # Learnable parameters
        self.init_zp = nn.Parameter(torch.zeros(num_hidden_states))
        self.init_log_sc = nn.Parameter(torch.zeros(num_hidden_states))
        self.alpha = nn.Parameter(torch.zeros(num_hidden_states))
        self.beta = nn.Parameter(torch.zeros(num_hidden_states))
        self.target = nn.Parameter(torch.zeros(num_hidden_states))

        self.reset_parameters()

    def reset_parameters(self):
        # zp: 0
        nn.init.constant_(self.init_zp, 0.0)

        # sc: exp(0) = 1
        nn.init.constant_(self.init_log_sc, 0.0)

        # alpha: sigmoid(2) = 0.5
        nn.init.normal_(self.alpha, mean=0.0, std=0.02)

        # beta: small
        nn.init.normal_(self.beta, mean=0.0, std=0.02)

        # target: upper_limit / 2
        nn.init.constant_(self.target, self.upper_limit / 2)

    # Constrained parameters
    def get_sc(self):
        return torch.exp(self.init_log_sc)

    def get_alpha(self):
        return torch.sigmoid(self.alpha)

    def get_beta(self):
        return 0.1 * torch.sigmoid(self.beta)

    def forward(self, kv_cache, CHUNK=32, output_states=False):
        """
        kv_cache: [B, T, C]
        B: The size of batches
        T: The length of token sequence (will be padded to a multiple of truncation)
        C: The number of channels
        """
        B, T, C = kv_cache.shape
        assert C == self.num_hidden_states
        assert self.truncation % CHUNK == 0

        alpha = self.get_alpha()
        beta = self.get_beta()
        target = self.target

        # Pad T to the nearest multiple of truncation so every chunk has
        # exactly `truncation` timesteps — required for torch.compile to see
        # a static shape and avoid recompilation.
        pad_len = (CHUNK - T % CHUNK) % CHUNK
        if pad_len > 0:
            # Pad with zeros; the extra outputs will be discarded at the end.
            kv_cache = torch.nn.functional.pad(kv_cache, (0, 0, 0, pad_len))
        T_padded = T + pad_len
        num_chunks = self.truncation // CHUNK
        num_truncations = T_padded // self.truncation

        # Expand 1 -> B
        current_zp = self.init_zp.expand(B, -1).contiguous()  # [B, C]
        current_sc = self.get_sc().expand(B, -1).contiguous()  # [B, C]

        chunk_inputs = kv_cache.split(CHUNK, dim=1)
        chunk_outputs = []
        chunk_zp_seq = []
        chunk_sc_seq = []
        for i in range(num_truncations):
            for j in range(num_chunks):
                kv_chunk = chunk_inputs[i * num_chunks + j]  # [B, chunk, C]
                out_chunk, current_zp, current_sc, zp_seq, sc_seq = _adkv_chunk_forward(
                    kv_chunk, current_zp, current_sc,
                    alpha, beta, target,
                    self.upper_limit, self.lower_limit,
                    self.sc_min, self.sc_max, output_states)
                chunk_outputs.append(out_chunk)

                if output_states:
                    chunk_zp_seq.append(zp_seq)
                    chunk_sc_seq.append(sc_seq)
            # Truncated BPTT: detach states between chunks
            current_zp = current_zp.detach()
            current_sc = current_sc.detach()

        outputs = torch.cat(chunk_outputs, dim=1)  # [B, T_padded, C]

        if output_states:
            zp_outputs = torch.cat(chunk_zp_seq, dim=1)
            sc_outputs = torch.cat(chunk_sc_seq, dim=1)
            # [B, T, C]
            return outputs[:, :T], zp_outputs[:, :T], sc_outputs[:, :T]
        else:
            return outputs[:, :T]  # [B, T, C]
