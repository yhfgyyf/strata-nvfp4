# strata-nvfp4

[简体中文](README.zh-CN.md)

NVIDIA Qwen3.8-Flash-Next-NVFP4 on one RTX 4090 with 48GB VRAM and 64GB system RAM.

Built on [Strata](https://github.com/Niko1221/Strata), with NVFP4 expert storage, W4A16 compute and a 256-slot prefill ring. The server provides a 256K context window, image input, tool calls, OpenAI Chat Completions and Responses, and Anthropic Messages.

## Model

**NVIDIA Qwen3.8-Flash-Next-NVFP4** has 48 layers, 512 experts per layer and 10 active experts per token. Its n-gram/PLE table is stored in FP8 and read from the original checkpoint at runtime. The main model uses native NVFP4 data rather than GGUF weights.

- [Hugging Face: nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
- [ModelScope: nv-community/Qwen3.8-Flash-Next-NVFP4](https://modelscope.cn/models/nv-community/Qwen3.8-Flash-Next-NVFP4)

The installer pins NVIDIA revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`. Downloads from either source are checked against the same NVIDIA file hashes.

| Files | Size |
|---|---:|
| Complete checkpoint download | 132.73GB (123.62GiB) |
| Generated NVFP4 expert weights, `experts.bin` | 67.95GB (63.28GiB) |
| Generated dense weights, `dense.bin` | 6.58GB (6.12GiB) |
| Original FP8 PLE/MTP checkpoint file, retained at runtime | 53.72GB (50.03GiB) |
| Generated MTP and image weights | About 1.73GB |

GB is decimal; GiB is binary. Generated data is additional to the download. Allow about **230GB of free disk space** for both, dependencies and the build.

**Weight size affects speed.** At a fixed VRAM budget, larger expert weights mean fewer experts can stay on the GPU. More cache misses require PCIe transfers or CPU computation, which can sharply reduce both prefill and decode throughput. Download size is not GPU memory usage: PLE is memory-mapped on the host, while dense weights, KV cache and working buffers also share the GPU budget.

## Machine

| Component | Tested configuration |
|---|---|
| GPU | One RTX 4090, 48GB, SM89; driver reports 49,140MiB |
| CPU | Intel Core i7-13700KF, 16 cores / 24 threads |
| RAM | 64GB |
| Storage | 500GB SATA SSD |
| OS | Ubuntu 24.04.4 LTS |
| CUDA / driver | CUDA Toolkit 13.2 / 595.104.02 |

This machine has a 48GB VRAM configuration. The installer defaults and measurements below target this configuration; a standard 24GB RTX 4090 needs a different memory budget.

## Throughput

Measured on October 12, 2026: one request at a time, no reused prefix, six samples per input length, exactly 512 output tokens. The server had a 256K context limit and two request slots. Settings: prefill chunk 8192, W4A16, R256, INT8 KV, MTP and prefix caching. Values are medians; loading and warmup are excluded.

| Input tokens | Prefill (tokens/s) | Decode (tokens/s) | Time to first token (s) |
|---:|---:|---:|---:|
| 8,192 | 3,197 | 102.0 | 2.60 |
| 32,768 | 3,465 | 99.4 | 9.54 |
| 131,072 | 3,524 | 98.8 | 37.44 |

Inputs were synthetic Chinese, English and code prompts. Decode speed varies with the output and MTP acceptance rate. These checks are not a general accuracy benchmark. See [measurement details](docs/PERFORMANCE.md).

## Install

Linux x86-64, Python 3.12+, a C++ compiler and CUDA Toolkit are required. See [INSTALL.md](docs/INSTALL.md) for prerequisites and separate model/data directories.

```bash
git clone https://github.com/yhfgyyf/strata-nvfp4.git
cd strata-nvfp4
./setup.sh --source modelscope --start
```

For Hugging Face:

```bash
./setup.sh --source huggingface --start
```

With an existing checkpoint:

```bash
./setup.sh --skip-download --model-dir /path/to/Qwen3.8-Flash-Next-NVFP4 --start
```

Setup compiles the engine, verifies the checkpoint and generates expert/dense data, indexes, the tokenizer, MTP and image encoder data. Keep the original model directory: PLE reads `model-fp8-mtp-ple.safetensors` directly.

To install through a coding assistant, give it this instruction:

> Install and start https://github.com/yhfgyyf/strata-nvfp4 on this machine. Follow docs/AI_SETUP.md, using NVIDIA Qwen3.8-Flash-Next-NVFP4 and the default W4A16 R256 configuration.

## Run

Start in the foreground; stop with `Ctrl+C`:

```bash
./start.sh
```

Start, inspect and stop a background service:

```bash
./service.sh start
./service.sh status
./service.sh stop
```

The web interface is at `http://127.0.0.1:30001`.

| Client setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:30001/v1` |
| Model name | `Qwen3.8-flash-next` |
| Local API key | `local` when authentication is not configured |
| Chat Completions | `/v1/chat/completions` |
| Responses | `/v1/responses` |

```bash
curl http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-flash-next","messages":[{"role":"user","content":"Explain Python generators in three sentences."}],"max_tokens":256,"reasoning_effort":"none"}'
```

See [API.md](docs/API.md) for Responses, image and tool-call examples. For access from other machines, configure both the listen address and an `api_key` as described in the installation guide.

## Defaults

| Setting | Value |
|---|---|
| Prefill computation | W4A16 |
| Decode computation | W4A16, NVFP4 fused MoE |
| Prefill chunk / ring slots | 8192 / 256 |
| Context limit / request slots | 262144 / 2 |
| KV cache | INT8 |
| GPU expert cache | Automatic allocation |
| Speculative decoding | MTP, `--spec 4`, batch MTP |
| Prefix caching | Enabled |
| Auxiliary CPU selection | Automatic |
| Reasoning / tool parsing | Enabled, with loop recovery |
| Images | Enabled, up to 1024 image tokens |

The installer writes `strata.json`. Data directories, port, context and request slots can be set during installation. Run `./setup.sh --help` for options.

## Changes from Strata

- Pack and load NVIDIA ModelOpt NVFP4 weights; retain the original FP8 PLE storage.
- Add NVFP4 W4A16 fused MoE, CPU expert computation and loader support.
- Include auxiliary CPU selection, prefill memory/transfer accounting and optional buffer compaction.
- Enable reasoning-loop, long-digit-loop and tool-call recovery.
- Add model-specific installation, data generation and W4A16 R256 defaults.

See [CHANGES.md](docs/CHANGES.md) for the source and script entry points.

## License

The code retains Strata's MIT License. Model weights follow the licenses on the NVIDIA model page. Based on work from [Niko1221/Strata](https://github.com/Niko1221/Strata), [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) and [llama.cpp/ggml](https://github.com/ggml-org/llama.cpp). See [THIRD_PARTY.md](docs/THIRD_PARTY.md).
