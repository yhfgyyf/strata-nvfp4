#include "strata/kernels/ngram.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/prefill/kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace k = strata::kernels;
void checked(cudaError_t status) { if (status != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(status)); std::exit(1); } }
void compare_gpu(const std::vector<uint8_t>& raw, const std::vector<float>& expected, float scale) {
    uint8_t* device_raw = nullptr; float* device_float = nullptr;
    checked(cudaMalloc(&device_raw, raw.size()));
    checked(cudaMalloc(&device_float, expected.size()*4));
    checked(cudaMemcpy(device_raw,raw.data(),raw.size(),cudaMemcpyHostToDevice));
    strata::prefill::ple_raw_fp8_to_f32(device_raw,device_float,expected.size(),scale,nullptr);
    std::vector<float> actual(expected.size());
    checked(cudaMemcpy(actual.data(),device_float,actual.size()*4,cudaMemcpyDeviceToHost));
    if (std::memcmp(actual.data(),expected.data(),actual.size()*4)) {
        for (size_t i=0;i<actual.size();++i) if (std::memcmp(&actual[i],&expected[i],4)) {
            std::fprintf(stderr,"mismatch at %zu scale %.9g expected %.9g actual %.9g\n",i,scale,expected[i],actual[i]); break;
        }
        std::exit(1);
    }
    cudaFree(device_raw);cudaFree(device_float);
}
int main(int argc,char** argv) {
    if (argc!=2) return 2;
    std::string err;
    k::PleIoOptions io;io.mode=k::PleIo::Mmap;
    setenv("STRATA_PLE_RAW_CACHE_MIB","0",1);
    k::PleTable original;
    if (!original.open(argv[1],err,io)) { std::fprintf(stderr,"%s\n",err.c_str());return 1; }
    assert(std::strcmp(original.format(),"F8_E4M3")==0 && original.rows()<UINT32_MAX);
    std::vector<uint8_t> all_codes(2560);
    for(size_t i=0;i<all_codes.size();++i)all_codes[i]=uint8_t(i);
    for(float scale : {0.0f,1.0f,0.1f,1.75f,std::numeric_limits<float>::min(),std::numeric_limits<float>::denorm_min(),
                      std::numeric_limits<float>::max(),original.fp8_scale()}) {
        std::vector<float> reference(all_codes.size());
        for(size_t i=0;i<all_codes.size();i+=160) k::fp8_e4m3_dequant_row(all_codes.data()+i,scale,reference.data()+i);
        compare_gpu(all_codes,reference,scale);
    }
    strata::GgufFile metadata(argv[1]);
    const auto* shard = metadata.get("strata.ple.rows_per_shard");
    const uint64_t shard_rows = shard ? shard->u : original.rows();
    assert(shard_rows > 0);
    const size_t tokens=512,n=tokens*k::PLE_N_HEADS;
    for(bool repeated : {false,true}) {
        std::vector<uint32_t> rows(n);
        for(size_t i=0;i<n;++i)rows[i]=uint32_t(uint32_t((repeated?i%32:i)*2654435761u)%original.rows());
        rows[0]=0;rows[1]=uint32_t(original.rows()-1);rows[2]=uint32_t(original.rows());rows[3]=UINT32_MAX;
        // Exercise every actual logical-shard boundary, with the original uncached scalar/batch oracle.
        size_t at=4;
        for(uint64_t boundary=shard_rows; boundary<original.rows() && at+1<n; boundary+=shard_rows) {
            rows[at++]=uint32_t(boundary-1);rows[at++]=uint32_t(boundary);
        }
        std::vector<float> reference(n*160);
        if(!original.gather_batch(rows.data(),tokens,reference.data(),err))return 1;
        for(const char* budget : {"0","1"}) {
            setenv("STRATA_PLE_RAW_CACHE_MIB",budget,1);
            k::PleTable table;
            if(!table.open(argv[1],err,io))return 1;
            std::vector<uint8_t> raw(n*160);
            for(int pass=0;pass<2;++pass) {
                const auto start=std::chrono::steady_clock::now();
                if(!table.gather_raw_fp8(rows.data(),tokens,raw.data(),err))return 1;
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                compare_gpu(raw,reference,table.fp8_scale());
                std::printf("{\"case\":\"%s\",\"cache_mib\":%s,\"pass\":%d,\"rows\":%zu,\"raw_bytes\":%zu,\"gpu_decoded_elements\":%zu,\"host_gather_ms\":%.6f,\"bit_equal\":true}\n",
                    repeated?"repeated":"near_unique",budget,pass,n,raw.size(),reference.size(),ms);
            }
            std::vector<float> cached(reference.size());
            if(!table.gather_batch(rows.data(),tokens,cached.data(),err))return 1;
            assert(std::memcmp(cached.data(),reference.data(),cached.size()*4)==0);
            table.close();
        }
    }
    std::puts("{\"all_256_codes\":true,\"scales\":8,\"actual_scale_included\":true,\"raw_and_cached_scalar_batch\":true,\"out_of_range_zero\":true,\"bit_equal\":true}");
}
