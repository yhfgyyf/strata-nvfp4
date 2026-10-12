// A/B the NVFP4 FP16 dequantizer: compare every half bit against a baseline dump and time real expert shapes.
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>

namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) { std::fprintf(stderr, "usage: nvfp4_dequant_parity output.bin [reference.bin]\n"); return 2; }
    std::ofstream output(argv[1], std::ios::binary);
    std::ifstream reference;
    if (argc == 3) reference.open(argv[2], std::ios::binary);
    if (!output || (argc == 3 && !reference)) return 2;
    std::mt19937 rng(1729);
    uint64_t compared = 0;
    for (auto geometry : {std::pair<int, int>{1, 256}, {3, 768}, {640, 2560}}) {
        const int ff = geometry.first, emb = geometry.second;
        const int64_t n = (int64_t) ff * emb;
        const size_t bytes = strata::kernels::iq_row_bytes(40, n);
        std::vector<uint8_t> packed(bytes * 2);
        for (size_t i = 0; i < packed.size(); ++i)
            packed[i] = (uint8_t) (rng() & (i % 36 < 4 ? 127 : 255));
        uint8_t* src; uint16_t* dst; float* tail;
        ck(cudaMalloc(&src, packed.size())); ck(cudaMalloc(&dst, (2 * n + 16) * 2)); ck(cudaMalloc(&tail, 12));
        ck(cudaMemcpy(src, packed.data(), packed.size(), cudaMemcpyHostToDevice));
        const float scales[3] = {0.0037f, 136.5f, 0.000025f};
        ck(cudaMemcpy(tail, scales, 12, cudaMemcpyHostToDevice));
        cudaEvent_t begin, end;
        ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
        for (int scaled = 0; scaled < 2; ++scaled) {
            for (int gu = 0; gu < 2; ++gu) {
                for (int offset : {8, 9}) {  // aligned vector store and the unaligned fallback
                    const int64_t count = gu ? 2 * n : n;
                    auto launch = [&] {
                        if (gu) strata::kernels::iq_dequant_gu_f16(40, src, src + bytes, ff, emb,
                                                                     dst + offset, nullptr, scaled ? tail : nullptr);
                        else strata::kernels::iq_dequant_f16(40, src, n, dst + offset, nullptr, scaled ? tail + 2 : nullptr);
                    };
                    ck(cudaMemset(dst, 0x5A, (2 * n + 16) * 2));
                    launch(); ck(cudaDeviceSynchronize());
                    std::vector<uint16_t> got(count + 16);
                    ck(cudaMemcpy(got.data(), dst, got.size() * 2, cudaMemcpyDeviceToHost));
                    for (int i = 0; i < offset; ++i) if (got[i] != 0x5A5A) return 3;
                    for (int64_t i = offset + count; i < (int64_t) got.size(); ++i) if (got[i] != 0x5A5A) return 3;
                    output.write((char*) got.data(), got.size() * 2);
                    if (reference.is_open()) {
                        std::vector<uint16_t> expected(got.size());
                        if (!reference.read((char*) expected.data(), expected.size() * 2)) return 4;
                        for (size_t i = 0; i < got.size(); ++i) if (got[i] != expected[i]) {
                            std::fprintf(stderr, "mismatch ff=%d emb=%d scaled=%d gu=%d offset=%d i=%zu %04x != %04x\n",
                                         ff, emb, scaled, gu, offset, i, got[i], expected[i]); return 5;
                        }
                        compared += got.size();
                    }
                    for (int i = 0; i < 10; ++i) launch();
                    ck(cudaEventRecord(begin));
                    for (int i = 0; i < 100; ++i) launch();
                    ck(cudaEventRecord(end)); ck(cudaEventSynchronize(end));
                    float ms; ck(cudaEventElapsedTime(&ms, begin, end));
                    std::printf("ff=%d emb=%d scaled=%d gu=%d offset=%d us=%.3f\n", ff, emb, scaled, gu, offset, ms * 10);
                }
            }
        }
        ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end));
        ck(cudaFree(tail)); ck(cudaFree(dst)); ck(cudaFree(src));
    }
    if (reference.is_open() && reference.peek() != std::ifstream::traits_type::eof()) return 4;
    std::printf("PASS compared_half_values=%llu\n", (unsigned long long) compared);
    return output ? 0 : 2;
}
