# strata-nvfp4

[English](README.md)

在一台 48GB RTX 4090、64GB 内存的 PC 上运行 NVIDIA 的 Qwen3.8-Flash-Next-NVFP4。

引擎基于 [Strata](https://github.com/Niko1221/Strata)。专家权重保留 NVFP4，默认使用 W4A16 计算和 256 槽 prefill ring。提供 256K 上下文、图片输入、工具调用，以及 OpenAI Chat Completions、Responses 和 Anthropic Messages 接口。

## 模型

使用 **NVIDIA Qwen3.8-Flash-Next-NVFP4**，48 层，每层 512 个专家，每个 token 选择 10 个。NVIDIA 权重中的 n-gram／PLE 表以 FP8 存储；运行时直接读取原始文件。主模型不转换为 GGUF。

下载地址：

- [Hugging Face：nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
- [ModelScope：nv-community/Qwen3.8-Flash-Next-NVFP4](https://modelscope.cn/models/nv-community/Qwen3.8-Flash-Next-NVFP4)

安装脚本固定 NVIDIA revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`，从 ModelScope 下载也会校验 NVIDIA 文件哈希。模型权重按模型发布页的许可证使用。

| 文件 | 大小 |
|---|---:|
| 完整模型下载 | 132.73GB（123.62GiB） |
| 生成的 NVFP4 专家权重 `experts.bin` | 67.95GB（63.28GiB） |
| 生成的 dense 权重 `dense.bin` | 6.58GB（6.12GiB） |
| 运行时保留的原始 FP8 PLE／MTP 文件 | 53.72GB（50.03GiB） |
| 生成的 MTP 与图片权重 | 约 1.73GB |

GB 使用十进制，GiB 使用二进制。运行数据需要额外生成，连同原始模型、依赖和编译文件，首次安装预留约 **230GB 空闲磁盘**。

**权重大小会影响推理速度。** 固定显存预算下，单个专家的权重越大，GPU 能常驻的专家越少。缓存未命中会增加 PCIe 搬运或 CPU 计算，可能大幅降低 prefill 和 decode 吞吐。下载大小不等于显存占用：PLE 在主机侧通过 mmap 读取，dense、KV cache 和工作缓冲也会占用显存。

同模型、同机器下，将服务从 2 个请求槽／256K 上下文改为 1 个请求槽／32K 上下文，腾出的显存让 GPU 常驻专家由 9,978 增至 14,361。此前 W4A16 R96、chunk=8192、输出 512 token 的测试中，8,192 token 输入的 prefill 中位数从 2,855 提升到 3,599 token/s（+26.1%），decode 从 102.8 提升到 117.7 token/s（+14.5%）。32,248 token 输入的中位数则为 prefill 3,828、decode 114.0 token/s。测试条件见[单请求槽测量记录](docs/PERFORMANCE.md#single-slot-32k-context)。

## 机器

| 部件 | 实测配置 |
|---|---|
| GPU | 单张 RTX 4090，48GB，SM89；驱动报告 49140 MiB |
| CPU | Intel Core i7-13700KF，16 核／24 线程 |
| 内存 | 64GB |
| 磁盘 | SATA SSD，500GB |
| 系统 | Ubuntu 24.04.4 LTS |
| CUDA／驱动 | CUDA Toolkit 13.2／595.104.02 |

这里使用的是 48GB 显存配置。安装脚本和下表均以这台机器为准；标准 24GB RTX 4090 需要另行调整显存预算。

## 吞吐

2026-10-12 实测，单请求、冷前缀、每个输入长度 6 个样本，输出均为 512 token。服务配置为 256K 上下文、2 个请求槽；prefill chunk=8192、W4A16、R256、INT8 KV、MTP 和前缀缓存。表中为中位数，模型加载和预热不计入。

| 输入 token | Prefill（token/s） | Decode（token/s） | 首 token（秒） |
|---:|---:|---:|---:|
| 8,192 | 3,197 | 102.0 | 2.60 |
| 32,768 | 3,465 | 99.4 | 9.54 |
| 131,072 | 3,524 | 98.8 | 37.44 |

这些是中文、英文和代码合成输入的记录。生成速度会随输出内容及 MTP 接受率变化；有限的质量检查不等于通用准确率评测。测试条件和数值见 [PERFORMANCE.md](docs/PERFORMANCE.md)。

### 256K 上下文对照

2026-10-14 实测：单客户端，输出512 token，前缀复用0，每个输入长度测三次，取算术平均。两种配置均使用262144上下文上限、W4A16、R256、chunk8192和INT8 KV。

| 输入 token | 两槽 Prefill | 两槽 Decode | 单槽 Prefill | 单槽 Decode |
|---:|---:|---:|---:|---:|
| 8,192 | 3,277.8 | 105.67 | 3,905.9 | 115.00 |
| 32,768 | 3,588.4 | 104.80 | 4,286.8 | 111.93 |
| 131,072 | 3,644.4 | 103.13 | 4,276.1 | 112.67 |

单位为token/s。两槽列是当前默认配置，开启RAM交换缓冲区轮换，GPU常驻专家10,295个。单槽列是单独测量的NOWAIT基线，rotation关闭，常驻专家13,377个；少预留一个请求槽，让更多显存用于专家权重。两组测速时都只有一个活动请求，默认服务仍支持两个并发请求。

这两列来自不同配置的独立运行，不是只改变请求槽数的对照。两槽rotation的配对测试中，32K decode相对前后基线均值提升5.5%；8K/128K差异小于基线漂移，prefill基本持平。上方10月12日表是六样本中位数，本表是三次平均值。[测试条件](docs/PERFORMANCE.md#256k-context-comparison)。

## 安装

需要 Linux x86-64、Python 3.12、C++ 编译器和 CUDA Toolkit。首次安装需要约 **230GB 空闲磁盘**：约 133GB 原始模型，加上运行数据、依赖和编译文件。准备命令见 [INSTALL.md](docs/INSTALL.md)。

```bash
git clone https://github.com/yhfgyyf/strata-nvfp4.git
cd strata-nvfp4
./setup.sh --source modelscope --start
```

使用 Hugging Face：

```bash
./setup.sh --source huggingface --start
```

已有模型文件：

```bash
./setup.sh --skip-download --model-dir /path/to/Qwen3.8-Flash-Next-NVFP4 --start
```

安装会编译引擎、校验模型，生成 experts／dense 数据、索引、tokenizer、MTP 和图片编码器。PLE 仍依赖原始 `model-fp8-mtp-ple.safetensors`，保留模型下载目录。

也可以把下面这句话交给编码助手：

> 在这台机器上安装并启动 https://github.com/yhfgyyf/strata-nvfp4 ，按仓库的 docs/AI_SETUP.md 执行，使用 NVIDIA Qwen3.8-Flash-Next-NVFP4 和默认 W4A16 R256 配置。

## 使用

前台启动，按 `Ctrl+C` 关闭：

```bash
./start.sh
```

后台启动、查看状态和关闭：

```bash
./service.sh start
./service.sh status
./service.sh stop
```

浏览器打开 `http://127.0.0.1:30001`。客户端设置：

| 项目 | 值 |
|---|---|
| Base URL | `http://127.0.0.1:30001/v1` |
| 模型名 | `Qwen3.8-flash-next` |
| 本地 API Key | 未配置鉴权时可填 `local` |
| Chat Completions | `/v1/chat/completions` |
| Responses | `/v1/responses` |

```bash
curl http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-flash-next","messages":[{"role":"user","content":"用三句话介绍这台本地模型。"}],"max_tokens":256,"reasoning_effort":"none"}'
```

Responses、图片和工具调用示例见 [API.md](docs/API.md)。对外提供服务时，在配置中同时设置监听地址和 `api_key`，参见安装文档。

## 默认参数

| 参数 | 默认值 |
|---|---|
| Prefill 计算 | W4A16 |
| Decode 计算 | W4A16，NVFP4 fused MoE |
| Prefill chunk／ring | 8192／256 |
| 上下文／请求槽 | 262144／2 |
| KV cache | INT8 |
| GPU 专家缓存 | 自动分配 |
| 推测解码 | MTP，`--spec 4`，batch MTP |
| 前缀缓存 | 开启 |
| aux CPU | 自动选择 |
| PLE 上传／CPU 行缓存 | 提前上传、FP8 原始字节搬运、64 MiB 原始行缓存 |
| RAM 专家交换 | 开启mapped交换缓冲区轮换 |
| 思考／工具解析 | 开启，支持循环恢复 |
| 图片 | 开启，最多 1024 个图片 token |

如需进一步提高推理速度，可按实际负载调整并发数和上下文上限；在满足需求的前提下降低这两项，可释放显存给更多常驻专家。Prefill chunk 可根据实际输入长度和显存余量，在 8192 的基础上增大或减小，并用实际请求对比 prefill 和 decode 速度。

配置由安装器写入 `strata.json`。可通过安装参数指定数据目录、端口、上下文和请求槽；完整选项用 `./setup.sh --help` 查看。

## 相对 Strata 的改动

- 增加 NVIDIA ModelOpt NVFP4 原始权重的打包和读取，保留 FP8 PLE 的原始存储。
- 增加 NVFP4 W4A16 fused MoE、CPU 专家计算及相关加载适配。
- 合入 aux CPU 选择，增加 prefill 内存／搬运记录和可选缓冲紧缩。
- 提前搬运 PLE 原始字节，在 GPU 上展开为 FP32，并启用 CPU 原始行缓存。
- 补充思考循环及长数字串检测、工具调用恢复。
- 增加本模型的一键安装、数据生成和 W4A16 R256 默认配置。

源文件和工具入口见 [CHANGES.md](docs/CHANGES.md)。

## 更新记录

### 2026-10-14

- 安装器默认设置 `STRATA_EXCHANGE_ROTATE=1`。符合条件的 mapped RAM 交换缓冲区在专家交换时转移所有权，省去 host commit 复制；H2D/D2H 搬运仍保留。

## 许可证

代码保留 Strata 的 MIT License。感谢 [Niko1221/Strata](https://github.com/Niko1221/Strata)、[sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) 和 [llama.cpp／ggml](https://github.com/ggml-org/llama.cpp) 的实现。第三方代码和模型的许可见 [THIRD_PARTY.md](docs/THIRD_PARTY.md)。
