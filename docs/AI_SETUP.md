# Installation through a coding assistant

Use this entry point when the user asks to install this repository on their Linux machine. The model is NVIDIA Qwen3.8-Flash-Next-NVFP4; the default is W4A16 R256.

## Check the machine

- Read README.md and docs/INSTALL.md.
- Verify Linux x86-64, one SM89 GPU with 48GB VRAM, 64GB RAM, Python 3.12+, a C++ compiler and CUDA Toolkit.
- Check disk space. A fresh download/export needs about 230GB free. Use `--skip-download --model-dir` for an existing complete checkpoint.
- Inspect GPU workloads and the intended port. Preserve other running models and jobs.
- Preserve existing configuration and data. Do not replace the driver unless the user explicitly requests it.

## Install

```bash
git clone https://github.com/yhfgyyf/strata-nvfp4.git
cd strata-nvfp4
./setup.sh --source modelscope
```

Use `--source huggingface` when requested. Supply `--cuda-home` for another Toolkit location. Dependencies belong in the local `.venv`; use the Ubuntu prerequisite commands in INSTALL.md if the compiler or Python venv support is missing.

Save the PID and log location for long-running stages, and inspect actual progress. Quiet download/build/export output does not establish failure. Read the specific error before retrying; do not delete the entire data directory.

## Start and verify

```bash
./service.sh start
tail -f logs/server.log
```

Poll `http://127.0.0.1:30001/v1/status` until `loaded=true`. Check the 262144 context limit, two request slots and image availability, then run:

```bash
.venv/bin/python tools/smoke.py --url http://127.0.0.1:30001
```

Report the Base URL, model name, start/stop commands and logs. Distinguish format tests, compilation and successful GPU inference. The throughput table applies to the machine listed in README.md.

The default binds only to localhost. For requested network access, configure both the listen address and `api_key`; do not put credentials in Git or public logs.
