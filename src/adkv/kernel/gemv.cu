#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

#define WARP_SIZE 32

// Warp-level horizontal reduction
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val)
{
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Warp-level horizontal reduction in fp16 (no CVT in/out).
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ __half warp_reduce_sum_f16(__half val)
{
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
    {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// Warp-level horizontal reduction over half2, both halves reduced in parallel
// Using a single SHFL+HADD2 chain
// Effectively halves the SHFL count vs reducing two scalars separately
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ __half2 warp_reduce_sum_f16x2(__half2 val)
{
    uint32_t v = *reinterpret_cast<uint32_t *>(&val);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
    {
        const uint32_t other = __shfl_xor_sync(0xffffffff, v, mask);
        const __half2 a = *reinterpret_cast<const __half2 *>(&v);
        const __half2 b = *reinterpret_cast<const __half2 *>(&other);
        const __half2 s = __hadd2(a, b);
        v = *reinterpret_cast<const uint32_t *>(&s);
    }
    return *reinterpret_cast<__half2 *>(&v);
}

// cp.async 16B copy (SM80+)
__device__ __forceinline__ void cp_async_cg_16B(void *smem_ptr, const void *gmem_ptr)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_int_ptr), "l"(gmem_ptr));
#else
    *reinterpret_cast<uint4 *>(smem_ptr) = *reinterpret_cast<const uint4 *>(gmem_ptr);
#endif
}
__device__ __forceinline__ void cp_async_commit()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// Fast 4-bit nibble pair -> half2 conversion using a bit trick
// fp16 representation of 1024+x (for x in 0..15) is 0x6400 | x
// (exponent=25, mantissa contributes the integer part exactly)
// After subtracting 1032, we recover x − 8 ∈ [−8, 7] in fp16, all without any I2F/F2H instructions
__device__ __forceinline__ __half2 nibbles2_to_half2(uint32_t pair)
{
    const uint32_t nA = pair & 0xF;
    const uint32_t nB = (pair >> 4) & 0xF;
    const uint32_t bits = 0x64006400u | nA | (nB << 16);
    __half2 h;
    *reinterpret_cast<uint32_t *>(&h) = bits;
    const __half2 bias = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));
    return __hsub2(h, bias);
}

// Append one decoded token to the uint4-tiled cache
__device__ __forceinline__ void encode_adkv_token(
    uint4 *__restrict__ cache_bh, // [n_super_cap, n_cgroup] for this (b,h)
    int n_cgroup,
    int sg_new, int br, int parity,
    const __half *__restrict__ k_new_bh, // [DIM]
    int cg,
    const __half2 *zp2,
    const __half2 *scale2)
{
    uint4 *slot = cache_bh + (size_t)sg_new * n_cgroup + cg;
    uint4 vec = *slot;
    uint32_t ch[4] = {vec.x, vec.y, vec.z, vec.w};
    const uint32_t shift = 8u * (uint32_t)br;
    const uint32_t mask = 0xFFu << shift;
    const int c0 = cg * 4;

#pragma unroll
    for (int p = 0; p < 2; ++p)
    {
        const __half2 kn = *reinterpret_cast<const __half2 *>(k_new_bh + c0 + 2 * p);

        // Low half of the pair = channel 2*p, high half = channel 2*p + 1
        int q0 = __float2int_rn(
            (__half2float(__low2half(kn)) - __half2float(__low2half(zp2[p]))) /
            __half2float(__low2half(scale2[p])));
        int q1 = __float2int_rn(
            (__half2float(__high2half(kn)) - __half2float(__high2half(zp2[p]))) /
            __half2float(__high2half(scale2[p])));
        q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
        q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);

        const uint32_t n0 = (uint32_t)(q0 + 8) & 0x0Fu;
        const uint32_t n1 = (uint32_t)(q1 + 8) & 0x0Fu;

        uint32_t b0 = (ch[2 * p] >> shift) & 0xFFu;
        uint32_t b1 = (ch[2 * p + 1] >> shift) & 0xFFu;
        if (parity)
        {
            b0 = (b0 & 0x0Fu) | (n0 << 4);
            b1 = (b1 & 0x0Fu) | (n1 << 4);
        }
        else
        {
            b0 = (b0 & 0xF0u) | n0;
            b1 = (b1 & 0xF0u) | n1;
        }
        ch[2 * p] = (ch[2 * p] & ~mask) | (b0 << shift);
        ch[2 * p + 1] = (ch[2 * p + 1] & ~mask) | (b1 << shift);
    }

    vec.x = ch[0];
    vec.y = ch[1];
    vec.z = ch[2];
    vec.w = ch[3];
    *slot = vec;
}

// Round-to-nearest int4 quantisation of one fp16 value:
//   q = clamp(round((val - zp) / sc), -8, 7)
__device__ __forceinline__ int quantize_f16(__half val, __half zp, __half sc)
{
    const int q = __float2int_rn((__half2float(val) - __half2float(zp)) / __half2float(sc));
    return q < -8 ? -8 : (q > 7 ? 7 : q);
}

// One ADKV step on a half2 channel pair:
//   zp += alpha * q * sc, sc *= 1 + beta * (|q| - target)
__device__ __forceinline__ void adkv_update_f16x2(
    __half2 q_f, __half2 &zp, __half2 &sc,
    __half2 alpha, __half2 beta, __half2 omt)
{
    zp = __hfma2(alpha, __hmul2(q_f, sc), zp);
    sc = __hmul2(sc, __hfma2(beta, __habs2(q_f), omt));
}

// ADKV Prefill Encode:
//   Quantize tokens [0, T) into the uint4-tiled KV cache.
template <int WAVE_PER_THREAD, int WRAPS_NUM, int BH_PER_CTA = 1>
__global__ void encode_adkv_kernel(
    uint4 *__restrict__ kv,              // [B*nh, n_super_cap, DIM/4] kv cache uint4-tiled
    const __half *__restrict__ kv_new,   // [B*nh, T, DIM] raw tokens to encode fp16
    const __half *__restrict__ kv_zp,    // [nh, DIM] zp_init fp16
    const __half *__restrict__ kv_sc,    // [nh, DIM] sc_init fp16
    const __half *__restrict__ kv_alpha, // [nh, DIM] alpha fp16
    const __half *__restrict__ kv_beta,  // [nh, DIM] beta fp16
    const __half *__restrict__ kv_t,     // [nh, DIM] target fp16
    int B /* Batch */, int nh /* Head */, int T /* Token */, int DIM /* Dim */,
    int n_super_cap /* Allocated supergroups per (b,h) */,
    __half scale_min, __half scale_max // [s_min, s_max] clamp
)
{
    static_assert(BH_PER_CTA == 1 || BH_PER_CTA == WRAPS_NUM,
                  "Head-pack requires BH_PER_CTA == WRAPS_NUM");

    const int lane = threadIdx.x;
    const int wrap_id = threadIdx.y;
    const int bx = (BH_PER_CTA > 1) ? (blockIdx.x * BH_PER_CTA + wrap_id)
                                    : blockIdx.x;
    if (bx >= B * nh)
        return;
    const int h = bx % nh;

    constexpr int CHAN_PER_THREAD = 4 * WAVE_PER_THREAD;
    constexpr int H2_PER_THREAD = CHAN_PER_THREAD / 2;

    // Per-channel filter states and fixed parameters, half2-packed
    __half2 zp[H2_PER_THREAD], sc[H2_PER_THREAD];
    __half2 alpha[H2_PER_THREAD], beta[H2_PER_THREAD], omt[H2_PER_THREAD];

#pragma unroll
    for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
    {
#pragma unroll
        for (int ci2 = 0; ci2 < 2; ci2++)
        {
            const int wave_stride = (BH_PER_CTA > 1)
                                        ? (4 * WARP_SIZE)
                                        : (4 * WARP_SIZE * WRAPS_NUM);
            const int warp_off = (BH_PER_CTA > 1) ? 0 : (wrap_id * 4 * WARP_SIZE);
            const int c0 = wi * wave_stride + warp_off + lane * 4 + ci2 * 2;
            const int pi = h * DIM + c0;
            const int ri = wi * 2 + ci2;

            zp[ri] = *reinterpret_cast<const __half2 *>(kv_zp + pi);
            sc[ri] = *reinterpret_cast<const __half2 *>(kv_sc + pi);
            alpha[ri] = *reinterpret_cast<const __half2 *>(kv_alpha + pi);
            const __half2 b_ = *reinterpret_cast<const __half2 *>(kv_beta + pi);
            const __half2 t_ = *reinterpret_cast<const __half2 *>(kv_t + pi);
            beta[ri] = b_;
            const __half2 one2 = __halves2half2(__float2half(1.0f), __float2half(1.0f));
            omt[ri] = __hsub2(one2, __hmul2(b_, t_)); // omt = 1 - beta * target
        }
    }

    constexpr int TILE = 4;
    const int T_packed = (T + 1) / 2;
    const int n_super = (T_packed + TILE - 1) / TILE;
    const int n_cgroup = DIM / TILE;
    uint4 *kv_bh = kv + (size_t)bx * n_super_cap * n_cgroup;
    const __half *kv_new_bh = kv_new + (size_t)bx * T * DIM;

#pragma unroll 1
    for (int sg = 0; sg < n_super; sg++)
    {
#pragma unroll
        for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
        {
            const int wave_stride = (BH_PER_CTA > 1) ? WARP_SIZE : (WARP_SIZE * WRAPS_NUM);
            const int warp_off = (BH_PER_CTA > 1) ? 0 : (wrap_id * WARP_SIZE);
            const int cg = wi * wave_stride + warp_off + lane;
            const int c_base = cg * 4;

            uint32_t ch[4] = {0u, 0u, 0u, 0u};

#pragma unroll
            for (int b = 0; b < TILE; b++)
            {
                const int t_even = (sg * TILE + b) * 2;
                const int t_odd = t_even + 1;

                // ---- Even token (low nibble) ----
                if (t_even < T)
                {
#pragma unroll
                    for (int ci2 = 0; ci2 < 2; ci2++)
                    {
                        const int ri = wi * 2 + ci2;
                        const __half2 kn = *reinterpret_cast<const __half2 *>(
                            kv_new_bh + (size_t)t_even * DIM + c_base + 2 * ci2);
                        const int q0 = quantize_f16(__low2half(kn),
                                                    __low2half(zp[ri]),
                                                    __low2half(sc[ri]));
                        const int q1 = quantize_f16(__high2half(kn),
                                                    __high2half(zp[ri]),
                                                    __high2half(sc[ri]));
                        ch[2 * ci2] |= ((uint32_t)(q0 + 8) & 0x0Fu) << (8 * b);
                        ch[2 * ci2 + 1] |= ((uint32_t)(q1 + 8) & 0x0Fu) << (8 * b);
                        const __half2 q_f = __halves2half2(__int2half_rn(q0),
                                                           __int2half_rn(q1));
                        adkv_update_f16x2(q_f, zp[ri], sc[ri],
                                          alpha[ri], beta[ri], omt[ri]);
                    }
                }

                // ---- Odd token (high nibble) ----
                if (t_odd < T)
                {
#pragma unroll
                    for (int ci2 = 0; ci2 < 2; ci2++)
                    {
                        const int ri = wi * 2 + ci2;
                        const __half2 kn = *reinterpret_cast<const __half2 *>(
                            kv_new_bh + (size_t)t_odd * DIM + c_base + 2 * ci2);
                        const int q0 = quantize_f16(__low2half(kn),
                                                    __low2half(zp[ri]),
                                                    __low2half(sc[ri]));
                        const int q1 = quantize_f16(__high2half(kn),
                                                    __high2half(zp[ri]),
                                                    __high2half(sc[ri]));
                        ch[2 * ci2] |= ((uint32_t)(q0 + 8) & 0x0Fu) << (4 + 8 * b);
                        ch[2 * ci2 + 1] |= ((uint32_t)(q1 + 8) & 0x0Fu) << (4 + 8 * b);
                        const __half2 q_f = __halves2half2(__int2half_rn(q0),
                                                           __int2half_rn(q1));
                        adkv_update_f16x2(q_f, zp[ri], sc[ri],
                                          alpha[ri], beta[ri], omt[ri]);
                    }
                }
            }

            uint4 vec;
            vec.x = ch[0];
            vec.y = ch[1];
            vec.z = ch[2];
            vec.w = ch[3];
            kv_bh[(size_t)sg * n_cgroup + cg] = vec;
        }
    }
}

// Host-side launcher for the prefill encode kernel
void launch_encode_adkv(
    uint4 *kv,
    const __half *kv_new,
    const __half *kv_zp,
    const __half *kv_sc,
    const __half *kv_alpha,
    const __half *kv_beta,
    const __half *kv_t,
    int B, int nh, int T, int DIM, int n_super_cap,
    __half scale_min, __half scale_max,
    cudaStream_t stream = 0)
{
    const int total_bh = B * nh;
    switch (DIM)
    {
    case 128:
    {
        constexpr int BH_PER_CTA = 4;
        const dim3 block(WARP_SIZE, BH_PER_CTA);
        const dim3 grid((total_bh + BH_PER_CTA - 1) / BH_PER_CTA);
        encode_adkv_kernel<1, BH_PER_CTA, BH_PER_CTA><<<grid, block, 0, stream>>>(
            kv, kv_new, kv_zp, kv_sc, kv_alpha, kv_beta, kv_t,
            B, nh, T, DIM, n_super_cap, scale_min, scale_max);
        break;
    }
    case 256:
    {
        const dim3 block(WARP_SIZE, 2);
        const dim3 grid(total_bh);
        encode_adkv_kernel<1, 2, 1><<<grid, block, 0, stream>>>(
            kv, kv_new, kv_zp, kv_sc, kv_alpha, kv_beta, kv_t,
            B, nh, T, DIM, n_super_cap, scale_min, scale_max);
        break;
    }
    case 512:
    {
        const dim3 block(WARP_SIZE, 4);
        const dim3 grid(total_bh);
        encode_adkv_kernel<1, 4, 1><<<grid, block, 0, stream>>>(
            kv, kv_new, kv_zp, kv_sc, kv_alpha, kv_beta, kv_t,
            B, nh, T, DIM, n_super_cap, scale_min, scale_max);
        break;
    }
    default:
        // Unsupported DIM
        // Supported set: {128, 256, 512}.
        break;
    }
}

// PyTorch interface for the prefill encode kernel
// Writes tokens [0, T) into their positions in kv_cache and returns it
//
// Parameters:
//   kv_cache: [B, nh, n_super_cap, DIM/4, 4, 4] uint8 uint4-tiled
//   kv_new: [B, nh, T, DIM] float16 raw tokens
//   kv_*: [nh, DIM] float16 ADKV params
torch::Tensor encode_kv(
    torch::Tensor kv_cache,
    torch::Tensor kv_new,
    torch::Tensor kv_zp_init, torch::Tensor kv_sc_init,
    torch::Tensor kv_alpha, torch::Tensor kv_beta, torch::Tensor kv_t,
    float scale_min, float scale_max)
{
    TORCH_CHECK(kv_new.scalar_type() == at::kHalf, "kv_new must be float16");
    TORCH_CHECK(kv_cache.scalar_type() == at::kByte, "kv_cache must be uint8");
    TORCH_CHECK(kv_cache.is_contiguous(), "kv_cache must be contiguous (written in place)");
    TORCH_CHECK(kv_zp_init.scalar_type() == at::kHalf, "kv_zp must be float16");
    TORCH_CHECK(kv_sc_init.scalar_type() == at::kHalf, "kv_sc must be float16");
    TORCH_CHECK(kv_alpha.scalar_type() == at::kHalf, "kv_alpha must be float16");
    TORCH_CHECK(kv_beta.scalar_type() == at::kHalf, "kv_beta must be float16");
    TORCH_CHECK(kv_t.scalar_type() == at::kHalf, "kv_t must be float16");

    const int B = kv_new.size(0);
    const int nh = kv_new.size(1);
    const int T = kv_new.size(2);
    const int DIM = kv_new.size(3);
    constexpr int TILE = 4;

    const int n_super_cap = kv_cache.size(2);
    TORCH_CHECK(kv_cache.size(3) * TILE == DIM,
                "kv_cache channel groups do not match kv_new head dim");
    const int n_super = ((T + 1) / 2 + TILE - 1) / TILE;
    TORCH_CHECK(n_super <= n_super_cap,
                "kv_cache capacity too small: need ", n_super,
                " supergroups for ", T, " tokens, have ", n_super_cap);
    TORCH_CHECK(DIM == 128 || DIM == 256 || DIM == 512,
                "DIM must be 128, 256 or 512");

    auto kv_new_flat = kv_new.contiguous();

    launch_encode_adkv(
        reinterpret_cast<uint4 *>(kv_cache.data_ptr<uint8_t>()),
        reinterpret_cast<const __half *>(kv_new_flat.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(kv_zp_init.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(kv_sc_init.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(kv_alpha.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(kv_beta.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(kv_t.data_ptr<at::Half>()),
        B, nh, T, DIM, n_super_cap,
        __float2half(scale_min), __float2half(scale_max));

    return kv_cache;
}

// ADKV Fused GEMV:
//   Decode-phase QK^T (16B uint4-tiled K layout, fp16/half2)
//
// Computes attention logits for a single decode step (T_q = 1):
//   o[bx, t] = \sum_c q[bx, c] * k_deq[bx, t, c] for t \in [0, T)
//
// K layout (memory order, outer -> inner):
//   k: [B*nh, T_packed/4, DIM/4, 4, 4] uint8 (bx, sg, cg, ch_in_group, br_in_super)
// Each thread loads one uint4 per (wi) -> 4 channels x 4 byte_rows = 8 tokens
//
// All per-channel states and parameters loads are fp16/half2 (same bit trick and omt fold as the OV pass)
// The final per-token logit is written as fp32 so the downstream softmax stays numerically stable
//
// The kernel supports two modes:
//   1. Head-pack: One wrap onto one BH (BH_PER_CTA > 1, multiple wraps (BHs) per CTA)
//   2. Channel-split: Multiple wraps onto one BH (BH_PER_CTA = 1, channel partitioning, multiple wraps (one BH) per CTA):
//
// Per supergroup (= 4 byte_rows = 8 tokens) we accumulate 8 fp32 partial sums per thread (sum over this thread's channels), warp-reduce,
// then cross-warp reduce via shared memory before writing.
template <int WAVE_PER_THREAD, int WRAPS_NUM, int BH_PER_CTA = 1>
__global__ void gemv_adkv_qk_kernel(
    const __half *__restrict__ q,       // [B*nh, DIM] query fp16
    uint4 *__restrict__ k,              // [B*nh, T_packed/4, DIM/4] k cache uint4-tiled
    const __half *__restrict__ k_new,   // [B*nh, DIM] new k entry to be appended fp16
    __half *__restrict__ o,             // [B*nh, T] logits fp16
    const __half *__restrict__ k_zp,    // [nh, DIM] zp_init fp16
    const __half *__restrict__ k_sc,    // [nh, DIM] sc_init fp16
    const __half *__restrict__ k_alpha, // [nh, DIM] alpha fp16
    const __half *__restrict__ k_beta,  // [nh, DIM] beta fp16
    const __half *__restrict__ k_t,     // [nh, DIM] target fp16
    int B /* Batch */, int nh /* Head */, int T /* Token */, int DIM /* Dim */,
    int n_super_cap /* Allocated supergroups per (b,h) */,
    __half scale_min, __half scale_max // [s_min, s_max] clamp
)
{
    static_assert(BH_PER_CTA == 1 || BH_PER_CTA == WRAPS_NUM,
                  "Head-pack requires BH_PER_CTA == WRAPS_NUM");

    const int lane = threadIdx.x;
    const int wrap_id = threadIdx.y;

    // Head-pack: wrap_id selects which (b,h) this warp owns
    // Channel-split: all warps share the same bx
    const int bx = (BH_PER_CTA > 1) ? (blockIdx.x * BH_PER_CTA + wrap_id)
                                    : blockIdx.x;
    if (bx >= B * nh)
        return;
    const int h = bx % nh;

    constexpr int CHAN_PER_THREAD = 4 * WAVE_PER_THREAD;
    constexpr int H2_PER_THREAD = CHAN_PER_THREAD / 2;

    // Per-channel ADKV states and q values, all half2-packed (2 channels / lane)
    __half2 zp[H2_PER_THREAD], scale[H2_PER_THREAD];
    __half2 alpha[H2_PER_THREAD], beta[H2_PER_THREAD], omt[H2_PER_THREAD];
    __half2 qv[H2_PER_THREAD];

#pragma unroll
    for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
    {
#pragma unroll
        for (int ci2 = 0; ci2 < 2; ci2++)
        {
            // Head-pack: each warp owns full DIM, no wrap_id stride
            // Channel-split: warps subdivide DIM, wrap_id strides 128 channels
            const int wave_stride = (BH_PER_CTA > 1)
                                        ? (4 * WARP_SIZE)
                                        : (4 * WARP_SIZE * WRAPS_NUM);
            const int warp_off = (BH_PER_CTA > 1) ? 0 : (wrap_id * 4 * WARP_SIZE);
            const int c0 = wi * wave_stride + warp_off + lane * 4 + ci2 * 2;
            const int pi = h * DIM + c0;
            const int qi = bx * DIM + c0;
            const int ri = wi * 2 + ci2;

            zp[ri] = *reinterpret_cast<const __half2 *>(k_zp + pi);
            scale[ri] = *reinterpret_cast<const __half2 *>(k_sc + pi);
            alpha[ri] = *reinterpret_cast<const __half2 *>(k_alpha + pi);
            const __half2 b_ = *reinterpret_cast<const __half2 *>(k_beta + pi);
            const __half2 t_ = *reinterpret_cast<const __half2 *>(k_t + pi);
            beta[ri] = b_;
            const __half2 one2 = __halves2half2(__float2half(1.0f), __float2half(1.0f));
            omt[ri] = __hsub2(one2, __hmul2(b_, t_)); // omt = 1 - beta * t
            qv[ri] = *reinterpret_cast<const __half2 *>(q + qi);
        }
    }

    // --- ChannelGroup (4*UINT32=128bits) ---                  -
    // ---  T0 T1   T2 T3   T4 T5   T6 T7  ---                  |
    // ---  C0 C0   C0 C0   C0 C0   C0 C0  --- pq[wi].x UINT32  |
    // ---  C1 C1   C1 C1   C1 C1   C1 C1  --- pq[wi].y UINT32  |
    // ---  C2 C2   C2 C2   C2 C2   C2 C2  --- pq[wi].z UINT32  |  SuperGroup (128 DIM)
    // ---  C3 C3   C3 C3   C3 C3   C3 C3  --- pq[wi].w UINT32  |
    // ---  E  O    E  O    E  O    E  O                        |
    // --- |Tile0| |Tile1| |Tile2| |Tile3| ---                  |
    // ... ...                                                  _
    constexpr int TILE = 4;
    const int T_packed = (T + 1) / 2;
    const int n_super = (T_packed + TILE - 1) / TILE;
    const int n_cgroup = DIM / TILE;
    const uint4 *k_bh = k + (size_t)bx * n_super_cap * n_cgroup;
    __half *o_bh = o + (size_t)bx * (T + 1);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // cp.async: N-stage ring buffer of K tiles
    // Each thread owns one uint4 slot per (wi, stage), keyed by (wrap_id, wi, lane), so warps get disjoint smem ranges.
    // Sized per stage so smem stays small: at DIM=512 / WRAPS_NUM=4 / WAVE_PER_THREAD=1 that's 2KB/stage, 8KB total,
    // well under the default 48KB budget
    constexpr int NSTAGES = 4;
    constexpr int SLOTS_PER_STAGE = WAVE_PER_THREAD * WRAPS_NUM * WARP_SIZE;
    __shared__ uint4 smem_k[NSTAGES][SLOTS_PER_STAGE];

    auto slot_idx = [&](int wi)
    {
        // Each thread owns one slot per wi; different warps get disjoint ranges
        return (wrap_id * WAVE_PER_THREAD + wi) * WARP_SIZE + lane;
    };

    // Load sg -> stage
    auto issue_load = [&](int sg, int stage)
    {
#pragma unroll
        for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
        {
            const int wave_stride = (BH_PER_CTA > 1) ? WARP_SIZE : (WARP_SIZE * WRAPS_NUM);
            const int warp_off = (BH_PER_CTA > 1) ? 0 : (wrap_id * WARP_SIZE);
            const int cg = wi * wave_stride + warp_off + lane;
            cp_async_cg_16B(&smem_k[stage][slot_idx(wi)],
                            &k_bh[sg * n_cgroup + cg]);
        }
        cp_async_commit();
    };

    // Prologue: prefetch first NSTAGES-1 tiles so that when the main loop starts,
    // iteration sg=0 can issue sg+NSTAGES-1 and immediately have NSTAGES groups in flight
#pragma unroll
    for (int p = 0; p < NSTAGES - 1; p++)
    {
        if (p < n_super)
        {
            issue_load(p, p);
        }
        else
        {
            // Empty group keeps pending count stable
            cp_async_commit();
        }
    }
#endif // __CUDA_ARCH__ >= 800

#pragma unroll 1
    for (int sg = 0; sg < n_super; sg++)
    {
        uint4 pq[WAVE_PER_THREAD];
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        const int cur_stage = sg % NSTAGES;
        const int nxt_sg = sg + NSTAGES - 1;
        const int nxt_stage = nxt_sg % NSTAGES;

        // Issue the tile NSTAGES-1 ahead of the current one
        if (nxt_sg < n_super)
        {
            issue_load(nxt_sg, nxt_stage);
        }
        else
        {
            cp_async_commit();
        }
        cp_async_wait_group<NSTAGES - 1>();

        // Consume the tile out of shared memory
#pragma unroll
        for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
        {
            pq[wi] = smem_k[cur_stage][slot_idx(wi)];
        }
#else
        // SM < 80: load the tile synchronously, straight into registers
#pragma unroll
        for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
        {
            const int wave_stride = (BH_PER_CTA > 1) ? WARP_SIZE : (WARP_SIZE * WRAPS_NUM);
            const int warp_off = (BH_PER_CTA > 1) ? 0 : (wrap_id * WARP_SIZE);
            const int cg = wi * wave_stride + warp_off + lane;
            pq[wi] = k_bh[sg * n_cgroup + cg];
        }
#endif

        // Partial sums in fp16
        __half2 part_even[TILE], part_odd[TILE];
#pragma unroll
        for (int b = 0; b < TILE; b++)
        {
            part_even[b] = __halves2half2(__float2half(0.0f), __float2half(0.0f));
            part_odd[b] = __halves2half2(__float2half(0.0f), __float2half(0.0f));
        }

        // Consume: 4 channels x 4 byte_rows x WAVE_PER_THREAD
#pragma unroll
        for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
        {
            uint32_t lanes[4];
            lanes[0] = pq[wi].x;
            lanes[1] = pq[wi].y;
            lanes[2] = pq[wi].z;
            lanes[3] = pq[wi].w;
#pragma unroll
            for (int ci2 = 0; ci2 < 2; ci2++)
            {
                const int ri = wi * 2 + ci2;
                const uint32_t pqA = lanes[ci2 * 2 + 0];
                const uint32_t pqB = lanes[ci2 * 2 + 1];
                const __half2 qval = qv[ri];

#pragma unroll
                for (int b = 0; b < TILE; b++)
                {
                    const uint8_t pkA = (pqA >> (8 * b)) & 0xFF;
                    const uint8_t pkB = (pqB >> (8 * b)) & 0xFF;
                    const uint8_t lo_pair = (pkA & 0x0F) | ((pkB & 0x0F) << 4);
                    const uint8_t hi_pair = (pkA >> 4) | ((pkB & 0xF0));
                    const __half2 qlo = nibbles2_to_half2(lo_pair);
                    const __half2 qhi = nibbles2_to_half2(hi_pair);

                    const int t_even = (sg * TILE + b) * 2;
                    const int t_odd = t_even + 1;

                    // ---- Low nibble (even token) ----
                    {
                        const __half2 qs = __hmul2(qlo, scale[ri]);
                        const __half2 vdeq = __hadd2(qs, zp[ri]);
                        // partial += qval * vdeq
                        part_even[b] = __hfma2(qval, vdeq, part_even[b]);
                        // Padding tokens at the tail of the last supergroup must
                        // not advance the ADKV filter state.
                        if (t_even < T)
                        {
                            adkv_update_f16x2(qlo, zp[ri], scale[ri],
                                              alpha[ri], beta[ri], omt[ri]);
                        }
                    }
                    // ---- High nibble (odd token) ----
                    {
                        const __half2 qs = __hmul2(qhi, scale[ri]);
                        const __half2 vdeq = __hadd2(qs, zp[ri]);
                        part_odd[b] = __hfma2(qval, vdeq, part_odd[b]);
                        if (t_odd < T)
                        {
                            adkv_update_f16x2(qhi, zp[ri], scale[ri],
                                              alpha[ri], beta[ri], omt[ri]);
                        }
                    }
                }
            }
        }

        // Collapse two channels per lane FIRST,
        // then pack (even_token, odd_token) into a single half2 so one warp_reduce_sum_f16x2 produces both token logits at once
        // Halves the SHFL-XOR count (4 reduces vs 8)
        __half2 packed[TILE];
#pragma unroll
        for (int b = 0; b < TILE; b++)
        {
            const __half e = __hadd(__low2half(part_even[b]), __high2half(part_even[b]));
            const __half o = __hadd(__low2half(part_odd[b]), __high2half(part_odd[b]));
            packed[b] = __halves2half2(e, o);
        }
#pragma unroll
        for (int b = 0; b < TILE; b++)
        {
            packed[b] = warp_reduce_sum_f16x2<WARP_SIZE>(packed[b]);
        }

        // Write logits
        if (lane == 0)
        {
            const int te0 = sg * 2 * TILE;
            const bool full_super = (te0 + 2 * TILE) <= T;
            // The output row has T + 1 entries (the last one is the self logit),
            // so the 16B store is aligned only when bx * (T + 1) is a multiple of 8 halves
            const bool aligned = (((bx * (T + 1)) & 7) == 0);
            if ((BH_PER_CTA > 1 || WRAPS_NUM == 1) && full_super && aligned)
            {
                // Fast path
                uint4 v;
                v.x = *reinterpret_cast<const uint32_t *>(&packed[0]);
                v.y = *reinterpret_cast<const uint32_t *>(&packed[1]);
                v.z = *reinterpret_cast<const uint32_t *>(&packed[2]);
                v.w = *reinterpret_cast<const uint32_t *>(&packed[3]);
                *reinterpret_cast<uint4 *>(o_bh + te0) = v;
            }
            else
            {
#pragma unroll
                for (int b = 0; b < TILE; b++)
                {
                    const int te = te0 + 2 * b;
                    const int to = te + 1;
                    const __half ev = __low2half(packed[b]);
                    const __half od = __high2half(packed[b]);
                    if (BH_PER_CTA > 1 || WRAPS_NUM == 1)
                    {
                        // Exclusive
                        if (te < T)
                        {
                            o_bh[te] = ev;
                        }
                        if (to < T)
                        {
                            o_bh[to] = od;
                        }
                    }
                    else
                    {
                        // Atom
                        if (te < T)
                        {
                            atomicAdd(&o_bh[te], ev);
                        }
                        if (to < T)
                        {
                            atomicAdd(&o_bh[to], od);
                        }
                    }
                }
            }
        }
    }

    // Append the new K token using the final filter states (zp/scale as of token T-1)
    // Token T goes to byte_row T/2: low nibble for even T,
    // high nibble for odd T, in supergroup (T/2)/TILE
    {
        const int r = T >> 1;
        const int parity = T & 1;
        const int sg_new = r / TILE;
        const int br_new = r % TILE;
        uint4 *cache_bh = k + (size_t)bx * n_super_cap * n_cgroup;

#pragma unroll
        for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
        {
            const int wave_stride = (BH_PER_CTA > 1) ? WARP_SIZE : (WARP_SIZE * WRAPS_NUM);
            const int warp_off = (BH_PER_CTA > 1) ? 0 : (wrap_id * WARP_SIZE);
            const int cg = wi * wave_stride + warp_off + lane;
            encode_adkv_token(cache_bh, n_cgroup, sg_new, br_new, parity,
                              k_new + (size_t)bx * DIM, cg,
                              &zp[wi * 2], &scale[wi * 2]);
        }
    }

    // Self-attention logit of the token that was just appended
    // zp/scale are still the state after token T-1,
    // i.e. exactly the state used above to quantize k_new, so k_new can be dequantized and dotted with q here
    {
        __half2 self_acc = __halves2half2(__float2half(0.0f), __float2half(0.0f));

#pragma unroll
        for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
        {
            const int wave_stride = (BH_PER_CTA > 1) ? WARP_SIZE : (WARP_SIZE * WRAPS_NUM);
            const int warp_off = (BH_PER_CTA > 1) ? 0 : (wrap_id * WARP_SIZE);
            const int c_base = (wi * wave_stride + warp_off + lane) * 4;

#pragma unroll
            for (int p = 0; p < 2; p++)
            {
                const int ri = wi * 2 + p;
                const __half2 kn = *reinterpret_cast<const __half2 *>(
                    k_new + (size_t)bx * DIM + c_base + 2 * p);

                int q0 = __float2int_rn(
                    (__half2float(__low2half(kn)) - __half2float(__low2half(zp[ri]))) /
                    __half2float(__low2half(scale[ri])));
                int q1 = __float2int_rn(
                    (__half2float(__high2half(kn)) - __half2float(__high2half(zp[ri]))) /
                    __half2float(__high2half(scale[ri])));
                q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
                q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);

                const __half2 q_f = __halves2half2(__int2half_rn(q0), __int2half_rn(q1));
                const __half2 k_deq = __hfma2(q_f, scale[ri], zp[ri]);
                self_acc = __hfma2(qv[ri], k_deq, self_acc);
            }
        }

        const __half self = warp_reduce_sum_f16<WARP_SIZE>(
            __hadd(__low2half(self_acc), __high2half(self_acc)));
        if (lane == 0)
        {
            if (BH_PER_CTA > 1 || WRAPS_NUM == 1)
            {
                o_bh[T] = self;
            }
            else
            {
                atomicAdd(&o_bh[T], self);
            }
        }
    }
}

// Host-side launcher
//
// Parameters:
//   q: [B * nh, DIM] fp16 device pointer
//   k: [B * nh, T_packed/4, DIM/4, 4, 4] uint8_t uint4-tiled
//   o: [B * nh, T + 1] fp16 device pointer (T cached logits + self logit;
//      zero-initialized when the channel-split warps accumulate with atomics)
//   k_new: [B * nh, DIM] fp16 new k entry
//   k_*: [nh, DIM] fp16 device pointers (ADKV params)
//   DIM: head dimension (must be 128, 256, or 512)
void launch_gemv_adkv_qk(
    const __half *q,
    uint8_t *k,
    const __half *k_new,
    __half *o,
    const __half *k_zp,
    const __half *k_sc,
    const __half *k_alpha,
    const __half *k_beta,
    const __half *k_t,
    int B, int nh, int T, int DIM, int n_super_cap,
    float scale_min, float scale_max,
    cudaStream_t stream = 0)
{
    const int total_bh = B * nh;
    switch (DIM)
    {
    case 128:
    {
        // head-pack: 4 warps/CTA, 1 warp owns one (b,h)
        // Trades smaller grid (B*nh/4) for higher per-CTA active warps (4 vs 1) and removes the cross-warp atomicAdd
        // Each warp covers full DIM=128 (1×32×4=128 chs)
        constexpr int BH_PER_CTA = 4;
        const dim3 block(WARP_SIZE, BH_PER_CTA);
        const dim3 grid((total_bh + BH_PER_CTA - 1) / BH_PER_CTA);
        gemv_adkv_qk_kernel<1, BH_PER_CTA, BH_PER_CTA><<<grid, block, 0, stream>>>(
            q, reinterpret_cast<uint4 *>(k), k_new, o, k_zp, k_sc, k_alpha, k_beta, k_t,
            B, nh, T, DIM, n_super_cap, __float2half(scale_min), __float2half(scale_max));
        break;
    }
    case 256:
    {
        const dim3 block(WARP_SIZE, 2);
        const dim3 grid(total_bh);
        gemv_adkv_qk_kernel<1, 2, 1><<<grid, block, 0, stream>>>(
            q, reinterpret_cast<uint4 *>(k), k_new, o, k_zp, k_sc, k_alpha, k_beta, k_t,
            B, nh, T, DIM, n_super_cap, __float2half(scale_min), __float2half(scale_max));
        break;
    }
    case 512:
    {
        const dim3 block(WARP_SIZE, 4);
        const dim3 grid(total_bh);
        gemv_adkv_qk_kernel<1, 4, 1><<<grid, block, 0, stream>>>(
            q, reinterpret_cast<uint4 *>(k), k_new, o, k_zp, k_sc, k_alpha, k_beta, k_t,
            B, nh, T, DIM, n_super_cap, __float2half(scale_min), __float2half(scale_max));
        break;
    }
    default:
        // Unsupported DIM
        // Supported set: {128, 256, 512}
        break;
    }
}

// Pytorch interface
torch::Tensor forward_qk(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor k_new,
    torch::Tensor k_zp_init, torch::Tensor k_sc_init,
    torch::Tensor k_alpha, torch::Tensor k_beta, torch::Tensor k_t,
    int T,
    float scale_min, float scale_max)
{
    TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be float16");
    TORCH_CHECK(k_new.scalar_type() == at::kHalf, "k_new must be float16");
    TORCH_CHECK(k_cache.scalar_type() == at::kByte, "k_cache must be uint8");
    TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous (appended in place)");
    TORCH_CHECK(k_zp_init.scalar_type() == at::kHalf, "k_zp must be float16");
    TORCH_CHECK(k_sc_init.scalar_type() == at::kHalf, "k_sc must be float16");
    TORCH_CHECK(k_alpha.scalar_type() == at::kHalf, "k_alpha must be float16");
    TORCH_CHECK(k_beta.scalar_type() == at::kHalf, "k_beta must be float16");
    TORCH_CHECK(k_t.scalar_type() == at::kHalf, "k_t must be float16");

    int B = q.size(0);
    int nh = q.size(1);
    constexpr int TILE = 4;
    int n_super_cap = k_cache.size(2);
    int n_cgroup = k_cache.size(3);
    int DIM = n_cgroup * TILE;

    TORCH_CHECK(DIM == q.size(3), "q head dimension does not match k_cache");
    TORCH_CHECK(T >= 0, "T must be non-negative");
    TORCH_CHECK((int64_t)(T >> 1) / TILE < (int64_t)n_super_cap,
                "k_cache has no free supergroup to append token ", T);

    // q is [B, nh, 1, DIM] -> [B*nh, DIM]; k_new likewise
    auto q_flat = q.contiguous();
    auto k_new_flat = k_new.reshape({B * nh, DIM}).contiguous();
    // T logits for the cached tokens plus the self logit of the appended token.
    // The channel-split configurations (DIM >= 256) accumulate across warps with atomicAdd,
    // so the buffer must start at zero;
    // the head-pack path (DIM=128) writes every element exactly once and can skip the memset.
    const bool channel_split = (DIM >= 256);
    auto o = (channel_split ? torch::zeros({B * nh, (int64_t)T + 1}, q_flat.options())
                            : torch::empty({B * nh, (int64_t)T + 1}, q_flat.options()));

    launch_gemv_adkv_qk(
        reinterpret_cast<const __half *>(q_flat.data_ptr<at::Half>()),
        k_cache.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(k_new_flat.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(o.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k_zp_init.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k_sc_init.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k_alpha.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k_beta.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k_t.data_ptr<at::Half>()),
        B, nh, T, DIM, n_super_cap,
        scale_min, scale_max);

    return o.reshape({B, nh, T + 1});
}

// ADKV Fused GEMV:
//   Decode-phase OV
//
// Computes the attention-weighted sum of values for a single decode step:
//   o[bx, c] = \sum_t attn[bx, t] * v_deq[bx, t, c] for c \in [0, DIM)
template <int WAVE_PER_THREAD, int WRAPS_NUM, int BH_PER_CTA = 1>
__global__ void gemv_adkv_ov_kernel(
    const __half *__restrict__ attn,    // [B*nh, T+1] attention weights (fp16)
    uint4 *__restrict__ v,              // [B*nh, n_super_cap, DIM/4] uint4-tiled (in/out)
    const __half *__restrict__ v_new,   // [B*nh, DIM] token T to append (fp16)
    __half *__restrict__ o,             // [B*nh, DIM] attention output (fp16)
    const __half *__restrict__ v_zp,    // [nh, DIM] dc_init (fp16)
    const __half *__restrict__ v_sc,    // [nh, DIM] scale_init (fp16)
    const __half *__restrict__ v_alpha, // [nh, DIM] alpha (fp16)
    const __half *__restrict__ v_beta,  // [nh, DIM] beta (fp16)
    const __half *__restrict__ v_t,     // [nh, DIM] target (fp16)
    int B, int nh, int T, int DIM, int n_super_cap,
    float scale_min, float scale_max)
{
    static_assert(BH_PER_CTA == 1 || BH_PER_CTA == WRAPS_NUM,
                  "Head-pack requires BH_PER_CTA == WRAPS_NUM");

    const int lane = threadIdx.x;
    const int wrap_id = threadIdx.y;
    // Head-pack: wrap_id selects which (b,h) this warp owns
    // Channel-split: all warps share the same bx
    const int bx = (BH_PER_CTA > 1) ? (blockIdx.x * BH_PER_CTA + wrap_id)
                                    : blockIdx.x;
    if (bx >= B * nh)
        return;
    const int h = bx % nh;
    constexpr int CHAN_PER_THREAD = 4 * WAVE_PER_THREAD;
    constexpr int H2_PER_THREAD = CHAN_PER_THREAD / 2;

    __half2 zp[H2_PER_THREAD], scale[H2_PER_THREAD];
    __half2 alpha[H2_PER_THREAD], beta[H2_PER_THREAD], omt[H2_PER_THREAD];
    __half2 acc[H2_PER_THREAD];

#pragma unroll
    for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
    {
#pragma unroll
        for (int ci2 = 0; ci2 < 2; ci2++)
        {
            // Head-pack: one warp covers full DIM, no wrap_id stride
            // Channel-split: warps subdivide DIM, wrap_id strides 128 channels
            const int wave_stride = (BH_PER_CTA > 1)
                                        ? (4 * WARP_SIZE)
                                        : (4 * WARP_SIZE * WRAPS_NUM);
            const int warp_off = (BH_PER_CTA > 1) ? 0 : (wrap_id * 4 * WARP_SIZE);
            const int c0 = wi * wave_stride + warp_off + lane * 4 + ci2 * 2;
            const int pi = h * DIM + c0;
            const int ri = wi * 2 + ci2;

            zp[ri] = *reinterpret_cast<const __half2 *>(v_zp + pi);
            scale[ri] = *reinterpret_cast<const __half2 *>(v_sc + pi);
            alpha[ri] = *reinterpret_cast<const __half2 *>(v_alpha + pi);
            const __half2 b_ = *reinterpret_cast<const __half2 *>(v_beta + pi);
            const __half2 t_ = *reinterpret_cast<const __half2 *>(v_t + pi);
            beta[ri] = b_;
            const __half2 one2 = __halves2half2(__float2half(1.0f), __float2half(1.0f));
            omt[ri] = __hsub2(one2, __hmul2(b_, t_));
            acc[ri] = __halves2half2(__float2half(0.0f), __float2half(0.0f));
        }
    }

    constexpr int TILE = 4;
    const int T_packed = (T + 1) / 2;
    const int n_super = (T_packed + TILE - 1) / TILE;
    const int n_cgroup = DIM / TILE;
    const uint4 *v_bh = v + (size_t)bx * n_super_cap * n_cgroup;
    const __half *attn_bh = attn + (size_t)bx * (T + 1);

#pragma unroll 1
    for (int sg = 0; sg < n_super; sg++)
    {
        uint4 pq[WAVE_PER_THREAD];
#pragma unroll
        for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
        {
            const int wave_stride = (BH_PER_CTA > 1) ? WARP_SIZE : (WARP_SIZE * WRAPS_NUM);
            const int warp_off = (BH_PER_CTA > 1) ? 0 : (wrap_id * WARP_SIZE);
            const int cg = wi * wave_stride + warp_off + lane;
            pq[wi] = v_bh[sg * n_cgroup + cg];
        }

        __half2 we2[TILE], wo2[TILE];
        const int t_base = sg * 2 * TILE;
        uint32_t attn_pair_bits = 0;

        if (lane < TILE)
        {
            // Two scalar loads: attn + bx*T is only 2-byte aligned when T is odd, so a half2 load would fault
            // The bound check also covers the partial last supergroup of an arbitrary-length sequence
            const int t0 = t_base + 2 * lane;
            if (t0 < T)
            {
                const __half a0 = attn_bh[t0];
                const __half a1 = (t0 + 1 < T) ? attn_bh[t0 + 1]
                                               : __float2half(0.0f);
                const __half2 pair = __halves2half2(a0, a1);
                attn_pair_bits = *reinterpret_cast<const uint32_t *>(&pair);
            }
        }

#pragma unroll
        for (int b = 0; b < TILE; b++)
        {
            const uint32_t bits = __shfl_sync(0xffffffff, attn_pair_bits, b);
            const __half2 pair = *reinterpret_cast<const __half2 *>(&bits);
            we2[b] = __half2half2(__low2half(pair));
            wo2[b] = __half2half2(__high2half(pair));
        }

#pragma unroll
        for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
        {
            uint32_t lanes[4];
            lanes[0] = pq[wi].x;
            lanes[1] = pq[wi].y;
            lanes[2] = pq[wi].z;
            lanes[3] = pq[wi].w;
#pragma unroll
            for (int ci2 = 0; ci2 < 2; ci2++)
            {
                const int ri = wi * 2 + ci2;
                const uint32_t pqA = lanes[ci2 * 2 + 0];
                const uint32_t pqB = lanes[ci2 * 2 + 1];
#pragma unroll
                for (int b = 0; b < TILE; b++)
                {
                    const uint8_t pkA = (pqA >> (8 * b)) & 0xFF;
                    const uint8_t pkB = (pqB >> (8 * b)) & 0xFF;
                    const uint8_t lo_pair = (pkA & 0x0F) | ((pkB & 0x0F) << 4);
                    const uint8_t hi_pair = (pkA >> 4) | ((pkB & 0xF0));
                    const __half2 qlo = nibbles2_to_half2(lo_pair);
                    const __half2 qhi = nibbles2_to_half2(hi_pair);

                    const int t_even = t_base + 2 * b;
                    const int t_odd = t_even + 1;

                    // ---- Low nibble ----
                    if (t_even < T)
                    {
                        const __half2 qs = __hmul2(qlo, scale[ri]);
                        const __half2 vdeq = __hadd2(qs, zp[ri]);
                        acc[ri] = __hfma2(we2[b], vdeq, acc[ri]);
                        adkv_update_f16x2(qlo, zp[ri], scale[ri],
                                          alpha[ri], beta[ri], omt[ri]);
                    }
                    // ---- High nibble ----
                    if (t_odd < T)
                    {
                        const __half2 qs = __hmul2(qhi, scale[ri]);
                        const __half2 vdeq = __hadd2(qs, zp[ri]);
                        acc[ri] = __hfma2(wo2[b], vdeq, acc[ri]);
                        adkv_update_f16x2(qhi, zp[ri], scale[ri],
                                          alpha[ri], beta[ri], omt[ri]);
                    }
                }
            }
        }
    }

    // Append the new V token using the final filter state (zp/scale as of token T-1)
    // Token T goes to byte_row T/2: low nibble for even T, high nibble for odd T, in supergroup (T/2)/TILE.
    {
        const int r = T >> 1;
        const int parity = T & 1;
        const int sg_new = r / TILE;
        const int br_new = r % TILE;
        uint4 *cache_bh = v + (size_t)bx * n_super_cap * n_cgroup;

#pragma unroll
        for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
        {
            const int wave_stride = (BH_PER_CTA > 1) ? WARP_SIZE : (WARP_SIZE * WRAPS_NUM);
            const int warp_off = (BH_PER_CTA > 1) ? 0 : (wrap_id * WARP_SIZE);
            const int cg = wi * wave_stride + warp_off + lane;
            encode_adkv_token(cache_bh, n_cgroup, sg_new, br_new, parity,
                              v_new + (size_t)bx * DIM, cg,
                              &zp[wi * 2], &scale[wi * 2]);
        }
    }

    // Self-attention contribution of the appended V token: attn[T] times its dequantized value
    // zp/scale are still the post-T-1 state
    {
        const __half2 self_w = __half2half2(attn_bh[T]);

#pragma unroll
        for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
        {
            const int wave_stride = (BH_PER_CTA > 1) ? WARP_SIZE : (WARP_SIZE * WRAPS_NUM);
            const int warp_off = (BH_PER_CTA > 1) ? 0 : (wrap_id * WARP_SIZE);
            const int c_base = (wi * wave_stride + warp_off + lane) * 4;

#pragma unroll
            for (int p = 0; p < 2; p++)
            {
                const int ri = wi * 2 + p;
                const __half2 vn = *reinterpret_cast<const __half2 *>(
                    v_new + (size_t)bx * DIM + c_base + 2 * p);

                int q0 = __float2int_rn(
                    (__half2float(__low2half(vn)) - __half2float(__low2half(zp[ri]))) /
                    __half2float(__low2half(scale[ri])));
                int q1 = __float2int_rn(
                    (__half2float(__high2half(vn)) - __half2float(__high2half(zp[ri]))) /
                    __half2float(__high2half(scale[ri])));
                q0 = q0 < -8 ? -8 : (q0 > 7 ? 7 : q0);
                q1 = q1 < -8 ? -8 : (q1 > 7 ? 7 : q1);

                const __half2 q_f = __halves2half2(__int2half_rn(q0), __int2half_rn(q1));
                const __half2 v_deq = __hfma2(q_f, scale[ri], zp[ri]);
                acc[ri] = __hfma2(self_w, v_deq, acc[ri]);
            }
        }
    }

    // Write output
    __half *o_bh = o + (size_t)bx * DIM;
#pragma unroll
    for (int wi = 0; wi < WAVE_PER_THREAD; wi++)
    {
#pragma unroll
        for (int ci2 = 0; ci2 < 2; ci2++)
        {
            const int wave_stride = (BH_PER_CTA > 1)
                                        ? (4 * WARP_SIZE)
                                        : (4 * WARP_SIZE * WRAPS_NUM);
            const int warp_off = (BH_PER_CTA > 1) ? 0 : (wrap_id * 4 * WARP_SIZE);
            const int c0 = wi * wave_stride + warp_off + lane * 4 + ci2 * 2;
            const int ri = wi * 2 + ci2;
            *reinterpret_cast<__half2 *>(o_bh + c0) = acc[ri];
        }
    }
}

// Host-side launcher for the OV pass
void launch_gemv_adkv_ov(
    const __half *attn,
    uint8_t *v,
    const __half *v_new,
    __half *o,
    const __half *v_zp,
    const __half *v_sc,
    const __half *v_alpha,
    const __half *v_beta,
    const __half *v_t,
    int B, int nh, int T, int DIM, int n_super_cap,
    float scale_min, float scale_max,
    cudaStream_t stream = 0)
{
    const int total_bh = B * nh;
    switch (DIM)
    {
    case 128:
    {
        constexpr int BH_PER_CTA = 4;
        const dim3 block(WARP_SIZE, BH_PER_CTA);
        const dim3 grid((total_bh + BH_PER_CTA - 1) / BH_PER_CTA);
        gemv_adkv_ov_kernel<1, BH_PER_CTA, BH_PER_CTA><<<grid, block, 0, stream>>>(
            attn, reinterpret_cast<uint4 *>(v), v_new, o, v_zp, v_sc, v_alpha, v_beta, v_t,
            B, nh, T, DIM, n_super_cap, scale_min, scale_max);
        break;
    }
    case 256:
    {
        const dim3 block(WARP_SIZE, 2);
        const dim3 grid(total_bh);
        gemv_adkv_ov_kernel<1, 2, 1><<<grid, block, 0, stream>>>(
            attn, reinterpret_cast<uint4 *>(v), v_new, o, v_zp, v_sc, v_alpha, v_beta, v_t,
            B, nh, T, DIM, n_super_cap, scale_min, scale_max);
        break;
    }
    case 512:
    {
        const dim3 block(WARP_SIZE, 4);
        const dim3 grid(total_bh);
        gemv_adkv_ov_kernel<1, 4, 1><<<grid, block, 0, stream>>>(
            attn, reinterpret_cast<uint4 *>(v), v_new, o, v_zp, v_sc, v_alpha, v_beta, v_t,
            B, nh, T, DIM, n_super_cap, scale_min, scale_max);
        break;
    }
    default:
        // Unsupported DIM
        // Supported set: {128, 256, 512}.
        break;
    }
}

torch::Tensor forward_ov(
    torch::Tensor attn,
    torch::Tensor v_cache,
    torch::Tensor v_new,
    torch::Tensor v_zp_init, torch::Tensor v_sc_init,
    torch::Tensor v_alpha, torch::Tensor v_beta, torch::Tensor v_t,
    float scale_min, float scale_max)
{
    TORCH_CHECK(attn.scalar_type() == at::kHalf, "attn must be float16");
    TORCH_CHECK(v_new.scalar_type() == at::kHalf, "v_new must be float16");
    TORCH_CHECK(v_cache.scalar_type() == at::kByte, "v_cache must be uint8");
    TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous (appended in place)");
    TORCH_CHECK(v_zp_init.scalar_type() == at::kHalf, "v_zp must be float16");
    TORCH_CHECK(v_sc_init.scalar_type() == at::kHalf, "v_sc must be float16");
    TORCH_CHECK(v_alpha.scalar_type() == at::kHalf, "v_alpha must be float16");
    TORCH_CHECK(v_beta.scalar_type() == at::kHalf, "v_beta must be float16");
    TORCH_CHECK(v_t.scalar_type() == at::kHalf, "v_t must be float16");

    int B = attn.size(0);
    int nh = attn.size(1);
    TORCH_CHECK(attn.size(2) >= 1,
                "attn must contain the cached weights plus the self weight");
    // The last attention weight belongs to the token that is appended
    // The kernel scans the T cached tokens and adds attn[T] * v_deq_new itself
    int T = attn.size(2) - 1;
    constexpr int TILE = 4;
    int n_super_cap = v_cache.size(2);
    int n_cgroup = v_cache.size(3);
    int DIM = n_cgroup * TILE;

    TORCH_CHECK(DIM == v_new.size(-1), "v_new head dimension does not match v_cache");
    TORCH_CHECK(T >= 0, "T must be non-negative");
    TORCH_CHECK((T >> 1) / TILE < n_super_cap,
                "v_cache has no free supergroup to append token ", T);

    auto attn_flat = attn.contiguous();
    auto v_new_flat = v_new.reshape({B * nh, DIM}).contiguous();

    // Output: [B*nh, DIM] (fp16)
    auto o = torch::empty({B * nh, DIM}, attn_flat.options());

    launch_gemv_adkv_ov(
        reinterpret_cast<const __half *>(attn_flat.data_ptr<at::Half>()),
        v_cache.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(v_new_flat.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(o.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v_zp_init.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v_sc_init.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v_alpha.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v_beta.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v_t.data_ptr<at::Half>()),
        B, nh, T, DIM, n_super_cap,
        scale_min, scale_max);

    return o;
}
