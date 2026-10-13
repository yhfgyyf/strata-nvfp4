#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::ngram {

// Fixed storage, eight ways per set, true LRU within each set. Keys are logical
// rows of the owning table; reset when that table closes or changes generation.
class RawRowCache {
    static constexpr size_t ways = 8, row_bytes = 160;
    struct Entry {
        uint32_t row = UINT32_MAX;
        uint64_t age = 0;
        std::array<uint8_t, row_bytes> data{};
    };
public:
    void reset(size_t budget) {
        const size_t available = budget > sizeof(*this) ? budget - sizeof(*this) : 0;
        const size_t count = (available / sizeof(Entry) / ways) * ways;
        std::vector<Entry>(count).swap(entries_);
        tick_ = 0;
    }
    bool enabled() const { return !entries_.empty(); }
    size_t owned_bytes() const { return enabled() ? sizeof(*this) + entries_.capacity() * sizeof(Entry) : 0; }
    size_t payload_bytes() const { return entries_.size() * row_bytes; }
    bool get(uint32_t row, uint8_t* out) {
        if (!enabled() || row == UINT32_MAX) return false;
        const size_t first = bucket(row);
        for (size_t i = first; i < first + ways; ++i) {
            Entry& e = entries_[i];
            if (e.row != row) continue;
            e.age = ++tick_;
            std::memcpy(out, e.data.data(), row_bytes);
            return true;
        }
        return false;
    }
    void put(uint32_t row, const uint8_t* data) {
        if (!enabled() || row == UINT32_MAX) return;
        const size_t first = bucket(row);
        size_t chosen = first;
        for (size_t i = first; i < first + ways; ++i) {
            if (entries_[i].row == row || entries_[i].row == UINT32_MAX) { chosen = i; break; }
            if (entries_[i].age < entries_[chosen].age) chosen = i;
        }
        Entry& e = entries_[chosen];
        e.row = row;
        e.age = ++tick_;
        std::memcpy(e.data.data(), data, row_bytes);
    }
private:
    size_t bucket(uint32_t row) const {
        uint32_t hash = row * 0x9e3779b1u;
        hash ^= hash >> 16;
        return (hash % (entries_.size() / ways)) * ways;
    }
    std::vector<Entry> entries_;
    uint64_t tick_ = 0;
};

}  // namespace strata::ngram
