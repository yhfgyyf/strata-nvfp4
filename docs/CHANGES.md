# Code changes and defaults

Based on Strata 0.1.40.3, retaining its C++/CUDA engine, Python server and web interface.

| Area | Changes |
|---|---|
| `src/artifact/`, native/PLE loading | External STRI indexes; NVIDIA FP8 PLE scale and shard mapping |
| `src/kernels/cuda/iq_kernels.cu` | NVFP4 W4A16 fused MoE |
| `src/kernels/cpu/nvfp4_*.cpp` | CPU NVFP4 expert computation |
| `src/program/generate.cpp`, `src/core/expert_source.cpp` | Auxiliary CPU selection and NVFP4 runtime support |
| `src/prefill/prefill.cpp` | Configurable prefill ring, optional GU/H compaction and transfer accounting |
| `serve/server.py`, `serve/frontend.py` | Reasoning-loop, long-digit-loop and tool-call parsing/recovery; honor the configured listen port |
| `tools/pack_modelopt_*.py` | NVIDIA expert, dense, PLE and MTP data generation |
| `tools/install.py`, shell entry points | Installation and W4A16 R256 defaults |

Prefill defaults to `STRATA_PREFILL_NVFP4=fp16` and `STRATA_PREFILL_RING=256`; decode uses `STRATA_NVFP4_FUSED_MOE=2`. The chunk size is 8192. Expert-cache allocation is automatic, with a 1536MiB VRAM reserve and INT8 KV. Prefix caching, batch MTP, images and reasoning recovery are enabled.

Buffer compaction and transfer accounting remain available in the source, disabled by default. FP8 GEMM is not part of the default computation path.
