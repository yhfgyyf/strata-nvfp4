#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace strata::prefill {

// One run, one PLE destination. Publication follows cudaEventRecord(consumed),
// so the next upload never waits on an event that has not yet been recorded.
class PleUploadFence {
public:
    bool wait_reusable(uint64_t generation) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return cancelled_ || generation <= consumed_ + 1; });
        return !cancelled_;
    }
    void consumed(uint64_t generation) {
        { std::lock_guard<std::mutex> lock(mutex_); consumed_ = generation; }
        cv_.notify_all();
    }
    void cancel() {
        { std::lock_guard<std::mutex> lock(mutex_); cancelled_ = true; }
        cv_.notify_all();
    }
    bool pending(uint64_t generation) const {
        return ready.load(std::memory_order_acquire) < generation && !failed.load(std::memory_order_acquire);
    }
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> ready{0};
    std::atomic<bool> failed{false};
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    uint64_t consumed_ = 0;
    bool cancelled_ = false;
};

}  // namespace strata::prefill
