#include "strata/kernels/ngram.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    namespace k = strata::kernels;
    if (argc != 3) return 2;
    std::ifstream f(argv[2], std::ios::binary);
    int32_t n = 0;
    f.read((char*) &n, 4);
    if (!f || n <= 0 || n > 4096 || n % k::PLE_N_HEADS) return 2;
    std::vector<uint32_t> rows((size_t) n);
    std::vector<float> ref((size_t) n * k::PLE_HEAD_DIM), got(ref.size()), batch(ref.size());
    f.read((char*) rows.data(), (std::streamsize) rows.size() * 4);
    f.read((char*) ref.data(), (std::streamsize) ref.size() * 4);
    if (!f) return 2;
    k::PleTable table;
    k::PleIoOptions io;
    io.mode = k::PleIo::Mmap;
    std::string err;
    if (!table.open(argv[1], err, io)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    for (int i = 0; i < n; ++i) table.read_row(rows[(size_t) i], got.data() + (size_t) i * k::PLE_HEAD_DIM);
    if (!table.gather_batch(rows.data(), (size_t) n / k::PLE_N_HEADS, batch.data(), err)) {
        std::fprintf(stderr, "%s\n", err.c_str()); return 1;
    }
    const bool ok = table.rows() == 320001536 && std::strcmp(table.format(), "BF16") == 0 &&
                    std::memcmp(got.data(), ref.data(), ref.size() * 4) == 0 &&
                    std::memcmp(batch.data(), ref.data(), ref.size() * 4) == 0;
    std::printf("RAW_PLE_RESULT: %s (%d rows, scalar and batch reads)\n", ok ? "exact BF16" : "MISMATCH", n);
    return ok ? 0 : 1;
}
