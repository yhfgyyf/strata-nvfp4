# Working on strata-nvfp4

This repository runs NVIDIA Qwen3.8-Flash-Next-NVFP4 on one SM89 GPU.
For installation, follow [docs/AI_SETUP.md](docs/AI_SETUP.md).

- Keep the main expert weights in NVFP4. The default is W4A16 R256, chunk 8192.
- Read the current files before editing. Preserve existing user configuration and model files.
- Never expose an unauthenticated API beyond localhost.
- Use `python -m unittest discover -s tests`, the packing tests and relevant serving tests for changes to installation or APIs.
- Build the native engine on Linux with CUDA, and distinguish CPU checks from GPU/runtime validation.
- Keep documentation factual. Measured throughput belongs to the stated machine and workload.
