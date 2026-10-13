#include "strata/ngram/raw_row_cache.hpp"
#include <cassert>
#include <array>
#include <cstring>

int main() {
    strata::ngram::RawRowCache cache;
    std::array<uint8_t, 160> row{}, out{};
    for (int i = 0; i < 160; ++i) row[i] = uint8_t(i * 19);
    cache.reset(0);
    cache.put(1, row.data());
    assert(!cache.get(1, out.data()));
    cache.reset(1600); // exactly one eight-way set, including metadata/object
    assert(cache.owned_bytes() <= 1600 && cache.payload_bytes() == 8 * 160);
    for (uint32_t id = 0; id < 8; ++id) { row[0] = uint8_t(id); cache.put(id, row.data()); }
    assert(cache.get(0, out.data()) && out[0] == 0);
    row[0] = 8; cache.put(8, row.data());
    assert(!cache.get(1, out.data())); // recently touched0 survives; oldest1 is evicted
    assert(cache.get(0, out.data()) && out[0] == 0);
    assert(cache.get(8, out.data()) && std::memcmp(row.data(), out.data(), row.size()) == 0);
    cache.reset(1 << 20);
    assert(cache.owned_bytes() <= 1 << 20);
    assert(!cache.get(8, out.data())); // new table generation never returns stale bytes
    for (uint32_t id = 0; id < 4096; ++id) { row[0] = uint8_t(id); cache.put(id, row.data()); }
    for (uint32_t id = 0; id < 4096; ++id)
        if (cache.get(id, out.data())) assert(out[0] == uint8_t(id));
    cache.reset(0);
    assert(!cache.get(8, out.data()));
}
