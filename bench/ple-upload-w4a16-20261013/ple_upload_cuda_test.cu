#include "strata/prefill/ple_upload.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void delayed_reader(const float* values, int count, float expected, int* errors) {
    const auto start = clock64();
    while (clock64() - start < 5000000) {}
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        if (values[i] != expected) atomicAdd(errors, 1);
}

void sequence(bool pageable, int failure) {
    constexpr int count = 4096, guard = 64;
    cudaStream_t compute{}, copy{};
    cudaEvent_t ready[2]{}, consumed{};
    float* loan = nullptr;
    int* errors = nullptr;
    float* host[2]{};
    std::vector<float> fallback[2];
    checked(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    checked(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    checked(cudaEventCreateWithFlags(&consumed, cudaEventDisableTiming));
    checked(cudaMalloc(&loan, (count + guard * 2) * sizeof(float)));
    checked(cudaMalloc(&errors, sizeof(int)));
    for (int b = 0; b < 2; ++b) {
        checked(cudaEventCreateWithFlags(&ready[b], cudaEventDisableTiming | cudaEventBlockingSync));
        if (pageable) { fallback[b].resize(count); host[b] = fallback[b].data(); }
        else checked(cudaHostAlloc(&host[b], count * sizeof(float), cudaHostAllocDefault));
    }
    for (int request = 0; request < 3; ++request) {
        std::vector<float> initial(count + guard * 2, -17.0f);
        checked(cudaMemcpy(loan, initial.data(), initial.size() * 4, cudaMemcpyHostToDevice));
        checked(cudaMemset(errors, 0, sizeof(int)));
        strata::prefill::PleUploadFence fence;
        std::future<bool> worker;
        auto start = [&](int generation) {
            const int b = (generation - 1) % 2;
            const int rows = generation == 3 ? 129 : count;
            worker = std::async(std::launch::async, [&, generation, b, rows] {
                checked(cudaSetDevice(0));
                try {
                    if (failure == 1 && generation == 2) throw std::runtime_error("injected gather failure");
                    std::fill_n(host[b], rows, float(request * 10 + generation));
                    if (!fence.wait_reusable(generation)) return false;
                    if (generation > 1) checked(cudaStreamWaitEvent(copy, consumed, 0));
                    cudaError_t result = cudaMemcpyAsync(failure == 2 && generation == 2 ? nullptr : loan + guard,
                        host[b], rows * 4, cudaMemcpyHostToDevice, copy);
                    if (result != cudaSuccess) { checked(cudaGetLastError()); throw std::runtime_error("injected copy failure"); }
                    if (failure == 3 && generation == 2) {
                        result = cudaEventRecord(nullptr, copy);
                        assert(result != cudaSuccess);
                        (void) cudaGetLastError();
                        throw std::runtime_error("injected event failure");
                    }
                    checked(cudaEventRecord(ready[b], copy));
                    fence.submitted.store(generation, std::memory_order_release);
                    checked(cudaEventSynchronize(ready[b]));
                    fence.ready.store(generation, std::memory_order_release);
                    return true;
                } catch (const std::exception&) {
                    fence.failed.store(true, std::memory_order_release);
                    return false;
                }
            });
        };
        start(1);
        for (int generation = 1; generation <= 3; ++generation) {
            const int b = (generation - 1) % 2;
            const int rows = generation == 3 ? 129 : count;
            const bool good = worker.get();
            if (!good) { assert(failure && !fence.pending(generation)); break; }
            assert(!fence.pending(generation));
            checked(cudaStreamWaitEvent(compute, ready[b], 0));
            if (generation < 3) {
                checked(cudaEventSynchronize(ready[b ^ 1]));
                start(generation + 1); // reads immediately; destination is still busy on compute
            }
            delayed_reader<<<1, 128, 0, compute>>>(loan + guard, rows, float(request * 10 + generation), errors);
            checked(cudaGetLastError());
            checked(cudaEventRecord(consumed, compute));
            fence.consumed(generation);
        }
        fence.cancel();
        if (worker.valid()) worker.wait();
        checked(cudaStreamSynchronize(copy));
        checked(cudaStreamSynchronize(compute));
        int wrong = 0;
        checked(cudaMemcpy(&wrong, errors, sizeof(int), cudaMemcpyDeviceToHost));
        assert(wrong == 0);
        checked(cudaMemcpy(initial.data(), loan, initial.size() * 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < guard; ++i) assert(initial[i] == -17 && initial[count + guard + i] == -17);
    }
    // A next-chunk worker waiting for an unpublished consumed event must wake on cancellation.
    strata::prefill::PleUploadFence cancel;
    auto blocked = std::async(std::launch::async, [&] { return cancel.wait_reusable(2); });
    assert(blocked.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    cancel.cancel();
    assert(!blocked.get());
    for (int b = 0; b < 2; ++b) { cudaEventDestroy(ready[b]); if (!pageable) cudaFreeHost(host[b]); }
    cudaEventDestroy(consumed); cudaStreamDestroy(copy); cudaStreamDestroy(compute);
    cudaFree(errors); cudaFree(loan);
}

int main() {
    try {
        for (bool pageable : {false, true}) for (int failure = 0; failure <= 3; ++failure) sequence(pageable, failure);
        std::puts("{\"passed\":true,\"pinned_and_pageable\":true,\"three_chunks_and_requests\":true,\"tail\":129,\"failure_modes\":3,\"cancellation\":true,\"canaries\":true}");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
