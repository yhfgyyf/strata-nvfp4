#include "strata/prefill/ple_upload.hpp"
#include <cassert>
#include <chrono>
#include <future>

using strata::prefill::PleUploadFence;
using namespace std::chrono_literals;

int main() {
    PleUploadFence fence;
    assert(fence.wait_reusable(1));
    auto next = std::async(std::launch::async, [&] { return fence.wait_reusable(2); });
    assert(next.wait_for(20ms) == std::future_status::timeout);
    fence.consumed(1);
    assert(next.get());
    fence.submitted.store(2, std::memory_order_release);
    assert(fence.pending(2)); // Queued DMA must not release future-layer credits.
    fence.ready.store(2, std::memory_order_release);
    assert(!fence.pending(2));
    assert(fence.pending(3));
    fence.failed.store(true, std::memory_order_release);
    assert(!fence.pending(3));
    auto cancelled = std::async(std::launch::async, [&] { return fence.wait_reusable(3); });
    assert(cancelled.wait_for(20ms) == std::future_status::timeout);
    fence.cancel();
    assert(!cancelled.get());
    assert(!fence.wait_reusable(1));
    PleUploadFence next_request;
    assert(next_request.wait_reusable(1) && next_request.pending(1));
}
