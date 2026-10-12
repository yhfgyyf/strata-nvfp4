// NVFP4 grouped W4A16 for CUDA SM80+: donor Marlin conversion/MMA with Strata's
// original expert blob and device plan. No checkpoint rewrite or FP4 activations.
// Included inside the iq_kernels.cu anonymous namespace after its legacy kernels.
#if !defined(__HIPCC__)
#include "nvfp4_marlin_primitives.cuh"

__device__ __forceinline__ half nvfp4_group_scale(uint8_t x) {
    if (x == 0 || x == 0x7f) return __float2half_rn(0.0f);
    const int exponent = (x >> 3) & 15;
    if (exponent == 0) return __float2half_rn((float) (x & 7) * (1.0f / 512.0f));
    return __ushort_as_half((uint16_t) (((x & 0x7f) << 7) + 0x2000));
}

// One lane obtains k=2*(lane%4), k+1 and k+8, k+9 for one weight row.
// The original blob stores low/high halves of each 16-value NVFP4 group.
__device__ __forceinline__ void nvfp4_mma_row(const block_nvfp4* row, int k16, int k2,
                                            uint32_t (&values)[2]) {
    const block_nvfp4& block = row[k16 / 64];
    const int sub = (k16 % 64) / 16;
    const uint16_t packed = *(const uint16_t*) (block.qs + sub * 8 + k2);
    const uint32_t q = ((uint32_t) (packed & 255) << 8) | ((uint32_t) (packed & 0xff00) << 16);
    half2 fragment[2];
    nvfp4_marlin::dequant_pair(q, fragment);
    const half2 scale = __half2half2(nvfp4_group_scale(block.d[sub]));
    fragment[0] = __hmul2(fragment[0], scale);
    fragment[1] = __hmul2(fragment[1], scale);
    values[0] = *reinterpret_cast<const uint32_t*>(&fragment[0]);
    values[1] = *reinterpret_cast<const uint32_t*>(&fragment[1]);
}

__device__ __forceinline__ void nvfp4_mma_activation(const float* row, int k,
                                                   uint32_t (&values)[2]) {
    const half2 low = row ? __floats2half2_rn(row[k], row[k + 1]) : __float2half2_rn(0.0f);
    const half2 high = row ? __floats2half2_rn(row[k + 8], row[k + 9]) : __float2half2_rn(0.0f);
    values[0] = *reinterpret_cast<const uint32_t*>(&low);
    values[1] = *reinterpret_cast<const uint32_t*>(&high);
}

__global__ void __launch_bounds__(128) nvfp4_fused_gu_swiglu_kernel(
    const unsigned long long* ptr, const int32_t* starts, const int32_t* groups,
    const int32_t* tokens, const float* x, NativeExpertLayout L, float* hidden) {
#if __CUDA_ARCH__ >= 800
    const int group = blockIdx.y;
    if (group >= *groups) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = (blockIdx.x * 4 + warp) * 16 + (lane >> 2), r1 = r0 + 8;
    if (r1 >= L.n_ff) return;  // host dispatch only accepts n_ff divisible by 64
    const int k2 = (lane & 3) * 2;
    const uint8_t* blob = (const uint8_t*) ptr[group];
    const auto* gate0 = (const block_nvfp4*) (blob + r0 * L.gu_row);
    const auto* gate1 = (const block_nvfp4*) (blob + r1 * L.gu_row);
    const auto* up0 = (const block_nvfp4*) (blob + L.up_off + r0 * L.gu_row);
    const auto* up1 = (const block_nvfp4*) (blob + L.up_off + r1 * L.gu_row);
    const float* scales = (const float*) (blob + L.tail_off);
    for (int e0 = starts[group]; e0 < starts[group + 1]; e0 += 8) {
        const int e1 = min(e0 + 8, starts[group + 1]);
        const int read_entry = e0 + (lane >> 2);
        const float* activation = read_entry < e1 ? x + (size_t) tokens[read_entry] * L.n_embd : nullptr;
        float g[4] = {}, u[4] = {};
        for (int k = 0; k < L.n_embd; k += 16) {
            uint32_t ga0[2], ga1[2], ua0[2], ua1[2], b[2];
            nvfp4_mma_row(gate0, k, k2, ga0); nvfp4_mma_row(gate1, k, k2, ga1);
            nvfp4_mma_row(up0, k, k2, ua0); nvfp4_mma_row(up1, k, k2, ua1);
            const uint32_t ga[4] = {ga0[0], ga1[0], ga0[1], ga1[1]};
            const uint32_t ua[4] = {ua0[0], ua1[0], ua0[1], ua1[1]};
            nvfp4_mma_activation(activation, k + k2, b);
            nvfp4_marlin::mma(ga, b, g); nvfp4_marlin::mma(ua, b, u);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int entry = e0 + (lane & 3) * 2 + (i & 1);
            if (entry < e1) {
                const float gate = g[i] * scales[0], up = u[i] * scales[1];
                const float h = (gate / (1.0f + __expf(-gate))) * up;
                hidden[(size_t) entry * L.n_ff + (i < 2 ? r0 : r1)] = __half2float(__float2half_rn(h));
            }
        }
    }
#endif
}

__global__ void __launch_bounds__(128) nvfp4_fused_down_kernel(
    const unsigned long long* ptr, const int32_t* starts, const int32_t* groups,
    const int32_t* destinations, const float* hidden, NativeExpertLayout L, float* out) {
#if __CUDA_ARCH__ >= 800
    const int group = blockIdx.y;
    if (group >= *groups) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = (blockIdx.x * 4 + warp) * 16 + (lane >> 2), r1 = r0 + 8;
    if (r1 >= L.n_embd) return;
    const int k2 = (lane & 3) * 2;
    const uint8_t* blob = (const uint8_t*) ptr[group];
    const auto* row0 = (const block_nvfp4*) (blob + L.down_off + r0 * L.d_row);
    const auto* row1 = (const block_nvfp4*) (blob + L.down_off + r1 * L.d_row);
    const float scale = ((const float*) (blob + L.tail_off))[2];
    for (int e0 = starts[group]; e0 < starts[group + 1]; e0 += 8) {
        const int e1 = min(e0 + 8, starts[group + 1]);
        const int read_entry = e0 + (lane >> 2);
        const float* activation = read_entry < e1 ? hidden + (size_t) read_entry * L.n_ff : nullptr;
        float c[4] = {};
        for (int k = 0; k < L.n_ff; k += 16) {
            uint32_t a0[2], a1[2], b[2];
            nvfp4_mma_row(row0, k, k2, a0); nvfp4_mma_row(row1, k, k2, a1);
            const uint32_t a[4] = {a0[0], a1[0], a0[1], a1[1]};
            nvfp4_mma_activation(activation, k + k2, b);
            nvfp4_marlin::mma(a, b, c);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int entry = e0 + (lane & 3) * 2 + (i & 1);
            if (entry < e1)
                out[(size_t) destinations[entry] * L.n_embd + (i < 2 ? r0 : r1)] = c[i] * scale;
        }
    }
#endif
}

// Small MoE groups do not fill an 8-token MMA tile. Decode four FP4 weights per
// lane with the same Marlin half2 primitive and reuse them for two entries.
// Gate/up and FP16 hidden conversion are fused. The down projection still uses
// the donor's tensor-core primitive.
__global__ void __launch_bounds__(128) nvfp4_fused_gu_small_kernel(
    const unsigned long long* ptr, const int32_t* starts, const int32_t* groups,
    const int32_t* tokens, const float* x, NativeExpertLayout L, float* hidden) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 4 + warp;
    if (row >= L.n_ff) return;
    for (int group = blockIdx.y; group < *groups; group += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) ptr[group];
        const auto* gate_row = (const block_nvfp4*) (blob + row * L.gu_row);
        const auto* up_row = (const block_nvfp4*) (blob + L.up_off + row * L.gu_row);
        const float* scales = (const float*) (blob + L.tail_off);
        for (int e0 = starts[group]; e0 < starts[group + 1]; e0 += 2) {
            const int count = min(2, starts[group + 1] - e0);
            const float* a0 = x + (size_t) tokens[e0] * L.n_embd;
            const float* a1 = count > 1 ? x + (size_t) tokens[e0 + 1] * L.n_embd : a0;
            float g[2] = {}, u[2] = {};
            for (int tile = 0; tile < L.n_embd; tile += 128) {
                const int k = tile + (lane >> 2) * 16, k2 = (lane & 3) * 2;
                uint32_t gw[2], uw[2];
                nvfp4_mma_row(gate_row, k, k2, gw); nvfp4_mma_row(up_row, k, k2, uw);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const float2 gate = __half22float2(*reinterpret_cast<const half2*>(&gw[half]));
                    const float2 up = __half22float2(*reinterpret_cast<const half2*>(&uw[half]));
                    const int offset = k + k2 + half * 8;
                    const float2 ax0 = __half22float2(__floats2half2_rn(a0[offset], a0[offset + 1]));
                    g[0] = fmaf(gate.x, ax0.x, g[0]); g[0] = fmaf(gate.y, ax0.y, g[0]);
                    u[0] = fmaf(up.x, ax0.x, u[0]); u[0] = fmaf(up.y, ax0.y, u[0]);
                    if (count > 1) {
                        const float2 ax1 = __half22float2(__floats2half2_rn(a1[offset], a1[offset + 1]));
                        g[1] = fmaf(gate.x, ax1.x, g[1]); g[1] = fmaf(gate.y, ax1.y, g[1]);
                        u[1] = fmaf(up.x, ax1.x, u[1]); u[1] = fmaf(up.y, ax1.y, u[1]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                if (i < count) {
                    const float gate = warp_sum(g[i]) * scales[0], up = warp_sum(u[i]) * scales[1];
                    if (lane == 0) {
                        const float h = (gate / (1.0f + __expf(-gate))) * up;
                        hidden[(size_t) (e0 + i) * L.n_ff + row] = __half2float(__float2half_rn(h));
                    }
                }
            }
        }
    }
}
#endif  // CUDA; HIP keeps its existing implementation
