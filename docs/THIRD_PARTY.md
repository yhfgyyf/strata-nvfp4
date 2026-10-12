# Licenses and sources

- **Strata**: https://github.com/Niko1221/Strata — MIT. Original copyright notices remain in LICENSE.
- **NVFP4 support**: https://github.com/sergqwer/strata-nvfp4 — MIT, retaining Strata's copyright notices.
- **Fused MoE reference**: https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120 . This repository uses a C++/CUDA SM89 W4A16 implementation; SGLang is not an installation dependency.
- **Marlin primitives**: `src/kernels/cuda/nvfp4_marlin_primitives.cuh` is adapted from Marlin/SGLang, Apache-2.0. Copyright and modification notices remain in the file; the license text is in [LICENSES/Apache-2.0.txt](../LICENSES/Apache-2.0.txt).
- **llama.cpp/ggml**: https://github.com/ggml-org/llama.cpp — MIT. Setup fetches commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` with its original LICENSE. The bundled ggml header license is in third_party/ggml/LICENSE.
- **Outfit web font**: SIL Open Font License, in serve/web/fonts/OFL.txt.
- **NVIDIA Qwen3.8-Flash-Next-NVFP4**: the model page lists the NVIDIA Open Model License and Qwen Community License 1.0. The downloaded licenses apply to model weights and model-derived assets; the code's MIT License does not replace them.

This repository does not redistribute the main model, PLE, MTP or image weights. `data/expert-profile.bin` contains expert rankings; `data/draft_vocab.bin` contains tokenizer IDs. The checkpoint manifest verifies downloads. Conversation text and private request records are not included.
