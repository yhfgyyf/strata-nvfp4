/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Extracted/adapted from yhfgyyf/sglang-qwen38-flash-next-sm120 main
// 4cacd4e586743600c35ae6748aa7b2b4fc5b151a:
// kernels/jit/csrc/gemm/marlin/dequant.h (FP4 half2 conversion), and
// kernels/jit/csrc/gemm/marlin_moe/marlin_template.h (FP16/FP32 MMA).
// Removed Torch/TVM/scalar-type dependencies; kept the instruction and FP4
// exponent adjustment. The grouped scheduling/blob mapping is Strata-specific.
#pragma once

namespace nvfp4_marlin {

__device__ __forceinline__ void dequant_pair(uint32_t q, half2 (&frag)[2]) {
    constexpr uint32_t MASK = 0x70007000;
    uint32_t out1 = (q & 0x80008000) | ((q & MASK) >> 3);
    q <<= 4;
    uint32_t out2 = (q & 0x80008000) | ((q & MASK) >> 3);
    // Two low-nibble values, then two high-nibble values. Multiply the raw
    // FP16 representation by 2^(15-1) to restore the FP4 exponent bias.
    const half2 bias = __float2half2_rn(16384.0f);
    frag[1] = __hmul2(*reinterpret_cast<const half2*>(&out1), bias);
    frag[0] = __hmul2(*reinterpret_cast<const half2*>(&out2), bias);
}

__device__ __forceinline__ void mma(const uint32_t (&a)[4], const uint32_t (&b)[2],
                                   float (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

}  // namespace nvfp4_marlin
