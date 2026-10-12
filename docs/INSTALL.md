# Installation and data generation

## Requirements

The defaults target Linux x86-64, one 48GB SM89 GPU and 64GB RAM. The tested system uses Ubuntu 24.04 and CUDA 13.2. Check that the driver and Toolkit are installed:

```bash
nvidia-smi
/usr/local/cuda/bin/nvcc --version
```

On Ubuntu 24.04:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3.12 python3.12-venv git
```

`setup.sh` creates a local `.venv` with the Python dependencies, CMake, Ninja and CPU-only PyTorch for export. It does not install or replace GPU drivers. To use another Python 3.12+ interpreter, set `STRATA_PYTHON=/path/to/python`.

The complete checkpoint is 132.73GB. Generated expert weights are 67.95GB, dense weights 6.58GB, and MTP/image weights about 1.73GB. Allow about **230GB free disk space**, including dependencies and build files. Model files and generated data can be stored separately:

```bash
./setup.sh --source modelscope \
  --model-dir /mnt/models/Qwen3.8-Flash-Next-NVFP4 \
  --data-dir /mnt/Strata-data --cuda-home /usr/local/cuda-13.2 --start
```

A read-only hardware/dependency check:

```bash
./setup.sh --check --cuda-home /usr/local/cuda-13.2
```

Before the first installation, this may report missing CMake/Ninja. Normal setup installs them in `.venv`; alternatively, install the system `cmake ninja-build` packages.

## Setup stages

1. Fetch the pinned llama.cpp archive and verify its SHA256.
2. Download the NVIDIA checkpoint, or verify the complete directory supplied with `--skip-download`. The ModelScope mirror uses the same NVIDIA hash manifest.
3. Compile Strata and the image encoder for SM89.
4. Generate native expert weights, dense data and the two `.stidx` indexes.
5. Export the tokenizer, install the expert ranking, and generate MTP and image data.
6. Write `strata.json`. Setup refuses to overwrite a different existing configuration; use `--config` for a separate file.

Repeat the same command after a download interruption. Expert packing saves progress per layer. Completed exports are reused when their checkpoint fingerprint matches. Interrupted dense/MTP staging directories produce an explicit error: move that specific directory aside before retrying. Setup does not automatically delete existing models or runtime data.

## Runtime files

Default location: `runtime/`.

| Path | Contents |
|---|---|
| `packs/nvidia-nvfp4-raw/experts.bin` | NVFP4 codes, block scales and global scales |
| `packs/nvidia-nvfp4-raw/dense.bin` | Q8_0 large matrices, BF16 small matrices, F32 norms, F16 PLE convolution |
| `packs/nvidia-nvfp4-raw/model.stidx` | Model metadata and external dense-data index |
| `packs/nvidia-nvfp4-raw/ple.stidx` | Original FP8 PLE file offsets, scale and shard mapping |
| `packs/nvidia-nvfp4-raw/expert-profile.bin` | GPU resident expert ranking |
| `packs/nvidia-nvfp4-raw/tokenizer/` | Tokenizer and chat template |
| `mtp/` | Q2_0 MTP experts, Q8_0 dense data and draft vocabulary |
| `vision/` | Image encoder/projector and vocabulary-only metadata |

The main model uses external raw data indexes, not GGUF model weights. The auxiliary image encoder uses llama.cpp's mmproj format.

**Keep the original checkpoint directory.** `ple.stidx` references `model-fp8-mtp-ple.safetensors` directly. Regenerate the PLE index after moving that file.

## Separate export commands

With the dependencies already installed in `.venv`:

```bash
.venv/bin/python tools/vendor.py
.venv/bin/python tools/download_model.py --source modelscope --model-dir /path/to/model
.venv/bin/python tools/pack_modelopt_experts.py --model /path/to/model --out /path/to/pack
.venv/bin/python tools/pack_modelopt_dense.py --model /path/to/model --out /path/to/pack --dense-q8
.venv/bin/python tools/prepare_assets.py --model /path/to/model --pack /path/to/pack --vision-out /path/to/vision
cp data/expert-profile.bin /path/to/pack/expert-profile.bin
.venv/bin/python tools/pack_modelopt_mtp.py --model /path/to/model --out /path/to/mtp --workers 2
```

Add `--sample-only` to the expert packer to check 12 matrix samples. Use `pack_modelopt_dense.py --ple-only` to regenerate only the PLE index.

The repository includes the expert ranking and Chinese/English draft vocabulary used for the throughput table. They contain expert coordinates and token IDs, with no conversation text. To generate new files:

```bash
.venv/bin/python tools/prepare_assets.py --model /path/to/model --pack /path/to/pack --draft-out /path/to/draft_vocab.bin
.venv/bin/python tools/make_profile.py --no-base --out /path/to/expert-profile.bin
```

`make_profile.py` can read traces produced by `--dump-routing` and rank experts by access count. Without traces, it creates a uniform ranking. This differs from the supplied ranking; the published measurements use the supplied file.

## Configuration and startup

```bash
./setup.sh --skip-download --model-dir /path/to/model --context 262144 --parallel 2 --port 30001
./service.sh start
curl http://127.0.0.1:30001/health
curl http://127.0.0.1:30001/v1/status
```

First loading can take several minutes. Server output goes to `logs/server.log`; engine output goes to `logs/engine.log`. Wait for `loaded: true` in `/v1/status` before sending requests.

For network access, set both `host: "0.0.0.0"` and a nonempty `api_key` in `strata.json`. Clients must send `Authorization: Bearer <key>`. Keep credentials in the local configuration.

Use `--no-vision` for text-only serving. To reuse existing data, supply `--reuse-pack`, `--reuse-mtp` and `--reuse-vision`. The image directory must contain `mmproj-bf16.gguf` and `tokenizer-only.gguf`.
