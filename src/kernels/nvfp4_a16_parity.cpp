// W4A16 GPU expert against a double reference with FP16 input and hidden activations.
// Exercises token/destination permutations, multiple groups, and an inactive group.
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: nvfp4_a16_parity pack/experts.bin [layer expert ...]\n"); return 2; }
    constexpr int64_t N = 2560, FF = 640;
    constexpr int T = 8;
    const k::NativeExpertLayout L = k::native_expert_layout(40, 40, N, FF);
    const size_t layer_bytes = 512 * L.bytes;          // L.bytes counts the 16-byte tail
    std::vector<int> picks;
    for (int i = 2; i + 1 < argc; i += 2) { picks.push_back(std::atoi(argv[i])); picks.push_back(std::atoi(argv[i + 1])); }
    if (picks.empty()) picks = {0, 0, 0, 7, 12, 100, 24, 300, 40, 5, 47, 511};
    std::ifstream f(argv[1], std::ios::binary);
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 1.f);
    const ggml_type_traits* tt = ggml_get_type_traits(GGML_TYPE_NVFP4);
    double worst = 0;
    const int32_t tokens[T] = {7, 5, 3, 1, 0, 2, 4, 6};
    const int32_t destinations[T] = {3, 7, 1, 5, 0, 6, 2, 4};
    for (size_t pi = 0; pi < picks.size(); pi += 2) {
        const int layer = picks[pi], e = picks[pi + 1];
        std::vector<uint8_t> blob(L.bytes);
        f.seekg((std::streamoff) (layer * layer_bytes + (size_t) e * L.bytes));
        if (!f.read((char*) blob.data(), (std::streamsize) blob.size())) { std::fprintf(stderr, "cannot read\n"); return 2; }
        float tail[4];
        std::memcpy(tail, blob.data() + L.tail_off, sizeof tail);
        std::vector<float> wg((size_t) FF * N), wu((size_t) FF * N), wd((size_t) N * FF), x((size_t) T * N);
        tt->to_float(blob.data(), wg.data(), FF * N);
        tt->to_float(blob.data() + L.up_off, wu.data(), FF * N);
        tt->to_float(blob.data() + L.down_off, wd.data(), N * FF);
        for (float& v : x) v = nd(rng);
        std::vector<double> ref((size_t) T * N);
        for (int t = 0; t < T; ++t) {
            std::vector<double> h(FF);
            for (int64_t r = 0; r < FF; ++r) {
                double g = 0, u = 0;
                for (int64_t c = 0; c < N; ++c) {
                    const float a = __half2float(__float2half_rn(x[(size_t) tokens[t] * N + c]));
                    g += (double) wg[(size_t) r * N + c] * a;
                    u += (double) wu[(size_t) r * N + c] * a;
                }
                g *= tail[0];
                u *= tail[1];
                h[r] = __half2float(__float2half_rn((float) (g / (1 + std::exp(-g)) * u)));
            }
            for (int64_t r = 0; r < N; ++r) {
                double s = 0;
                for (int64_t c = 0; c < FF; ++c) s += (double) wd[(size_t) r * FF + c] * h[c];
                ref[(size_t) destinations[t] * N + r] = s * tail[2];
            }
        }
        uint8_t* db; float *dx, *dout; void *scr; unsigned long long* dptr; int32_t *dstart, *dng, *ddst, *dtok;
        ck(cudaMalloc(&db, blob.size()), "blob");
        ck(cudaMemcpy(db, blob.data(), blob.size(), cudaMemcpyHostToDevice), "blob");
        ck(cudaMalloc(&dx, x.size() * 4), "x");
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        ck(cudaMalloc(&dout, (size_t) T * N * 4), "out");
        ck(cudaMalloc(&scr, k::native_expert_scratch_bytes(T, FF)), "scratch");
        const unsigned long long ptr[3] = {(unsigned long long) db, (unsigned long long) db, 0};
        const int32_t start[4] = {0, 3, T, T}, ng = 2;
        ck(cudaMalloc(&dptr, sizeof ptr), "p"); ck(cudaMemcpy(dptr, ptr, sizeof ptr, cudaMemcpyHostToDevice), "p");
        ck(cudaMalloc(&dstart, sizeof start), "s"); ck(cudaMemcpy(dstart, start, sizeof start, cudaMemcpyHostToDevice), "s");
        ck(cudaMalloc(&dng, 4), "n"); ck(cudaMemcpy(dng, &ng, 4, cudaMemcpyHostToDevice), "n");
        ck(cudaMalloc(&ddst, 4 * T), "d"); ck(cudaMemcpy(ddst, destinations, 4 * T, cudaMemcpyHostToDevice), "d");
        ck(cudaMalloc(&dtok, 4 * T), "t"); ck(cudaMemcpy(dtok, tokens, 4 * T, cudaMemcpyHostToDevice), "t");
        k::native_expert_grouped_a16(L, dptr, dstart, dng, ddst, dtok, 3, T, dx, scr, dout, nullptr);
        ck(cudaDeviceSynchronize(), "run");
        std::vector<float> got((size_t) T * N);
        ck(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost), "out");
        std::vector<float> sentinel(got.size(), 13.0f);
        const int32_t zero = 0;
        ck(cudaMemcpy(dng, &zero, 4, cudaMemcpyHostToDevice), "zero groups");
        ck(cudaMemcpy(dout, sentinel.data(), sentinel.size() * 4, cudaMemcpyHostToDevice), "sentinel");
        k::native_expert_grouped_a16(L, dptr, dstart, dng, ddst, dtok, 3, T, dx, scr, dout, nullptr);
        std::vector<float> empty(got.size());
        ck(cudaMemcpy(empty.data(), dout, empty.size() * 4, cudaMemcpyDeviceToHost), "empty groups");
        if (empty != sentinel) { std::fprintf(stderr, "empty groups modified output\n"); return 1; }
        for (void* p : {(void*) db, (void*) dx, (void*) dout, scr, (void*) dptr, (void*) dstart, (void*) dng,
                        (void*) ddst, (void*) dtok})
            cudaFree(p);
        double num = 0, den = 0;
        for (size_t i = 0; i < got.size(); ++i) { num += (got[i] - ref[i]) * (got[i] - ref[i]); den += ref[i] * ref[i]; }
        const double rel = std::sqrt(num / den);
        if (!std::isfinite(rel)) { std::fprintf(stderr, "non-finite expert output or reference\n"); return 1; }
        worst = std::fmax(worst, rel);
        std::printf("layer %2d expert %3d: s_gate %.3g s_up %.3g s_down %.3g -> expert output rel. error %.3f%%\n", layer,
                    e, tail[0], tail[1], tail[2], 100 * rel);
    }
    std::printf("W4A16_RESULT: %s (worst %.6f%%, shuffled rows and zero groups checked)\n",
                worst < 0.001 ? "ok" : "TOO FAR", 100 * worst);
    return worst < 0.001 ? 0 : 1;
}
