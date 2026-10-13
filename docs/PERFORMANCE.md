# Measured performance

October 12, 2026: one RTX 4090 with 48GB VRAM, i7-13700KF, 64GB RAM and SATA SSD, running Ubuntu 24.04 and CUDA 13.2. Model: NVIDIA Qwen3.8-Flash-Next-NVFP4, revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`.

W4A16, R256, prefill chunk 8192, INT8 KV, context limit 262144, two server slots. One client submitted sequential requests. MTP, prefix caching, auxiliary CPU selection, the image encoder and fused MoE were enabled. Formal measurements used no profiler or transfer ledger. Each input length has six cold samples, zero prefix reuse and exactly 512 output tokens. Values below are medians.

| Input tokens | Prefill tokens/s | Decode tokens/s | TTFT seconds |
|---:|---:|---:|---:|
| 8192 | 3196.8928 | 102.00 | 2.6018 |
| 32768 | 3465.1320 | 99.35 | 9.5402 |
| 131072 | 3524.0567 | 98.80 | 37.4397 |

Prefill uses the engine's complete input-processing timer. Decode uses its generation-phase timer and counts final accepted output with MTP. TTFT starts when the client sends the request and ends at the first nonempty string delta; it includes server preparation and phase transitions. These metrics cannot simply be added together.

All 18 records completed 512 output tokens, with no repeated-tail or long-digit loop detected. The synthetic-prompt checks do not establish general model accuracy. Output content, workload and MTP acceptance rate affect decode speed.

Data: [performance.json](performance.json). The supplied expert ranking has SHA256 `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`; the draft vocabulary has SHA256 `0fa45afd44c300f5407b49f5dd13544b0e555200e1a0d2c1de15c54167bd41e3`. Model loading and warmup are excluded.

## Single-slot, 32K context

An earlier test on October 12 used the same model, machine, engine binary and expert ranking, with W4A16, R96, prefill chunk 8192 and INT8 KV. Reducing the server allocation from two request slots / 262144 context tokens to one slot / 32768 context tokens increased startup resident experts from 9,978 to 14,361. Each row below has three cold requests, one client, no prefix reuse and exactly 512 output tokens. The profiler and transfer ledger were disabled during speed measurements.

| Server allocation | Input tokens | Prefill tokens/s | Decode tokens/s | TTFT seconds |
|---|---:|---:|---:|---:|
| Two slots / 256K context | 8192 | 2855.35 | 102.8 | 2.9058 |
| One slot / 32K context | 8192 | 3599.30 | 117.7 | 2.3133 |
| One slot / 32K context | 32248 | 3828.11 | 114.0 | 8.5026 |

For the identical 8192-token prompts, the ratio of medians improved by 26.1% for prefill and 14.5% for decode. The 32768-token window includes input, output and eight safety tokens, so its longest tested input was 32248 tokens. The 3828 prefill result and 117.7 decode result belong to different input lengths.

These are R96 configuration-level observations, separate from the R256 measurements above. Context, request-slot capacity and warmup changed together, and this supplementary test had no returning control run; it does not isolate the speedup caused by expert residency alone. All six single-slot requests retained the baseline's 5/6 synthetic-check score and had no repeated-tail or long-digit loop detected.

## 256K context comparison

October 14, 2026, same NVIDIA model and machine: context limit 262144, W4A16, R256, prefill chunk 8192, INT8 KV, MTP, PLE early/raw upload and its 64 MiB CPU raw-row cache. One client, three warmups excluded, three cold speed requests per size, 512 output tokens and zero prefix reuse. Rates below are arithmetic means; the once-per-request native host timing summary and 1 Hz NVML sampling were uniform. No Nsight or GPU profiler ran during speed tests.

| Input tokens | Two slots, rotation on: prefill | Decode | TTFT s | One slot, rotation off: prefill | Decode | TTFT s |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 3277.8 | 105.67 | 2.5320 | 3905.9 | 115.00 | 2.1366 |
| 32768 | 3588.4 | 104.80 | 9.2099 | 4286.8 | 111.93 | 7.7357 |
| 131072 | 3644.4 | 103.13 | 36.2013 | 4276.1 | 112.67 | 30.8926 |

The two-slot configuration has 10295 resident GPU experts and is the current default. Its rows are `rotation-r1/01-B-rotate`. The one-slot configuration has 13377 resident experts, effective native batch=0, and uses NOWAIT with rotation off; its rows are `lazy-r1/00-A0-single-nowait`. Both execute GEN for the speed requests. The server's slot capacity is different from actual client concurrency.

The table compares complete configurations from separate series, not the isolated effect of slots or rotation. The slot allocation changes the expert-memory budget; rotation also differs. The older October 12 measurements above use medians of six samples, whereas these use means of three. Do not derive a causal speedup from those different dates and statistics.

For rotation alone in the same two-slot A/B/A series, 32K decode is 104.80 tok/s versus the surrounding A mean of 99.33 (+5.50%); signed control drift is -0.87%. At 8K/128K, the observed +3.31%/+5.10% changes are smaller than absolute control drift of 6.40%/6.38%, so gains remain inconclusive. Prefill changes are -0.58%/-0.13%/-0.05%. The 32K decode sample standard deviations for A0/B/A1 are 4.08/1.87/5.68 tok/s; three trials do not establish a statistically guaranteed improvement.

All compared outputs contain 512 tokens, with the same 5/6 synthetic checks and no screened repeat/long-digit loops. The code-result check already fails in the baseline; these checks are not general model accuracy. First-token full-vocabulary logits for the rotation candidate match its control at all three lengths.

Individual values, sample standard deviations, timing/quality receipts and source/build identities are in the [two-slot summary](../bench/pcie-lazy-w4a16-20261013/evidence-public/rotation-r1/summary.json) and [single-slot summary](../bench/pcie-lazy-w4a16-20261013/evidence-public/lazy-r1/summary.json). See the [full report](../bench/pcie-lazy-w4a16-20261013/REPORT.md) for the independent PCIe/lazy comparisons and deployment checks.
