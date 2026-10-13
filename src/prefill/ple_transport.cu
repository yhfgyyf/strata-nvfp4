#include "strata/prefill/kernels.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace strata::prefill {
namespace {
__global__ void ple_raw_fp8_to_f32_kernel(const uint8_t* raw, float* output, int64_t count, float scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned value = raw[i], exponent = (value >> 3) & 15, mantissa = value & 7;
    unsigned bits = 0;
    if (exponent == 0) {
        if (mantissa >= 4) bits = (120u << 23) | ((mantissa - 4) << 21);
        else if (mantissa >= 2) bits = (119u << 23) | ((mantissa - 2) << 22);
        else if (mantissa == 1) bits = 118u << 23;
    } else if (!(exponent == 15 && mantissa == 7)) {
        bits = ((exponent + 120) << 23) | (mantissa << 20);
    }
    bits |= (value >> 7) << 31; // CPU table maps both NaN encodings to signed zero.
    output[i] = __fmul_rn(__uint_as_float(bits), scale);
}
}  // namespace

void ple_raw_fp8_to_f32(const uint8_t* raw, float* output, int64_t count, float scale, void* stream) {
    if (count <= 0) return;
    ple_raw_fp8_to_f32_kernel<<<(unsigned) ((count + 255) / 256), 256, 0, (cudaStream_t) stream>>>(raw, output, count, scale);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(std::string("PLE raw expansion: ") + cudaGetErrorString(status));
}
}  // namespace strata::prefill
