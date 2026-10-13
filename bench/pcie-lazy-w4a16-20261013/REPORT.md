# W4A16 PCIe / lazy residency implementation

Status: complete. RAM exchange-buffer rotation is enabled and verified in production; other PCIe/lazy settings retain their original defaults.

The revised [plan](PLAN.md) applies the single-answer audit's corrections. Experiments run on the RTX 4090 48 GiB / i7-13700KF / 64 GiB server, using the NVIDIA checkpoint and the existing `4ae23d2d…` PLE-enabled binary. Expert storage remains NVFP4 and both GPU resident and demand-offloaded experts use fused MoE backend 2, W4A16. CPU experts retain the original Q8_0 activation path; this is not an all-A16 CPU implementation. Prefill uses chunk 8192 and R256; context is 262144 and the main matrix retains two configured slots with one client. Actual speed-request decode is GEN, verified by its native summary. The one-token numerical requests use BGEN in the two-slot configuration; they are separate diagnostics.

The main matrix is `{kernel,dma} × {0,0.1,0.2}`, with each candidate bracketed by fresh-profile baseline arms. Every arm has three 8K, three 32K and three 128K inputs, each producing 512 tokens. Three warmups and forced-prompt vocabulary diagnostics are excluded. No Nsight or GPU profiler runs during speed measurements. The existing once-per-request native host timing summary is enabled uniformly, and NVML device memory is sampled at 1 Hz.

## Audit changes applied

- Separate configured batch capacity from GEN versus actual batch-mtp execution. WAIT/LAG requires disabling default NOWAIT; `--batch 1` is normalized to effective batch=0. Do not relax async or automatic-cost-model batch guards.
- Keep independent pinned exchange memory, stable GPU slot ownership and reader/completion fences. Never overwrite an incoming RAM expert while CPU/H2D/mapped GPU readers use it. Temporary PCIe staging is not persistent spare storage.
- Jointly compare mode and fraction; retain a bounded short check and owned-process-group teardown for unavailable APIs. A source-recorded DMA driver-lock risk is a test hypothesis, not a claimed failure.
- Do not compare the historical CPU barrier's exclusive 59.64 µs with the full adaptive H2D copy's 171.79 µs as a scheduling threshold. Their populations and overlap definitions differ.
- Restrict SGLang Marlin/CUTLASS statements to the pinned source's conditional dispatch. This task does not verify or modify a SGLang deployment.

## Baseline A0, arithmetic mean of three

| Input / output | Prefill tok/s | Decode tok/s | Quality checks |
|---|---:|---:|---|
| 8192 / 512 | 3303.7 | 99.53 | 5/6 on each case |
| 32768 / 512 | 3589.7 | 99.50 | 5/6 on each case |
| 131072 / 512 | 3647.5 | 99.80 | 5/6 on each case |

The code-result check already fails in the baseline. Retrieval/arithmetic/translation checks pass, outputs are complete, and no periodic-tail or long-digit loop is detected. These synthetic checks do not constitute a general model accuracy evaluation. Individual values, standard deviations, speculation counts, output hashes and actual offload shares are retained in [public evidence](evidence-public/matrix-r1/).

The first baseline return has bit-identical forced-prompt logits at all three lengths. Positive fractions change CPU-versus-GPU rounding; candidate logits and top-1 agreement are recorded separately from task correctness. A nominal fraction applies to distinct cache misses, is converted to a numerator over 256 and rounded down per layer; it is not a fraction of all routed expert entries.

## Memory and source identity

An expert blob is 2,764,816 bytes. The verifier's 16 staging blobs consume 42.19 MiB; this is separate from the prefill R256 ring. Blocking adaptation uses up to 96 pinned exchange blobs (253.13 MiB), while the existing async path requires up to 192 exchange/bounce blobs (506.25 MiB). The captured two-slot arms so far have 10295 resident slots; actual capacity is retained for every arm. Free startup VRAM is not a guaranteed GPU-spare budget.

The server root contains older generate/expert_source snapshots. The retained build links the loaded binary to candidate-b8 generate object `5cca522b…` and the upstream-port engine archive, whose source hashes match this checkout (`41738aef…` / `efb88ac7…`). Root-source hashes are observations, not loaded-build provenance. See [build records](evidence-public/build-provenance.public.json).

Production config and scheduler script are backed up under `/home/tc/Strata-nvfp4/bench/pcie-lazy-w4a16-20261013/matrix-r1/`. Model weights, profile and draft vocabulary remain unchanged. Trial raw outputs/logits stay on the server. The trial runner restores the original owned Strata service independently of candidate API reachability and never restarts a miner.

## Joint matrix result

All 99 cold speed rows are complete, with zero prefix reuse, 512 output tokens, score 5/6 and no screened repetition. All arms have 10295 resident slots. None of the five alternatives gives consistent decode improvement across lengths; keep the existing kernel/0 production setting. DMA passed the bounded checks on this workload, which does not establish safety for every concurrency or fraction.

| Candidate | Prefill 8K / 32K / 128K tok/s | Decode 8K / 32K / 128K tok/s | Decode change vs adjacent A means |
|---|---|---|---|
| kernel/0.1 | 3298.9 / 3593.4 / 3648.4 | 102.60 / 101.63 / 97.17 | +1.42% / +0.11% / -1.70% |
| kernel/0.2 | 3302.8 / 3594.8 / 3650.5 | 100.93 / 99.60 / 100.57 | -2.75% / -2.92% / +2.58% |
| dma/0 | 3290.0 / 3588.4 / 3649.7 | 102.70 / 101.17 / 97.53 | -1.52% / +0.68% / -1.22% |
| dma/0.1 | 3305.5 / 3595.2 / 3651.8 | 102.00 / 99.97 / 98.83 | -1.42% / -0.88% / +0.73% |
| dma/0.2 | 3300.6 / 3590.6 / 3640.1 | 102.30 / 97.23 / 98.17 | +0.20% / -4.34% / +0.86% |

For each nonzero fraction, kernel and DMA forced-prompt logits are byte-identical at all three lengths. Their differences from frac=0 reflect CPU/GPU arithmetic selection. The baseline returns are byte-identical at these diagnostics. These checks do not measure general task accuracy.

## Single-slot lazy mechanisms

The five-arm series is complete: 45 cold requests, all with 512 output tokens, zero prefix reuse, score 5/6 and no screened loops. The production two-slot service was restored after the series. Explicit `--batch 1` is normalized to effective native batch=0; INFO omits `batch_slots`, the API reports serving capacity 1, and numerical diagnostics and speed requests both execute GEN. Context stays 262144. This frees memory allocated for the additional slot and raises resident expert capacity to 13377 in every arm. It is a separate configuration comparison, not evidence that a lazy mechanism alone creates those slots.

| Arm | Prefill 8K / 32K / 128K tok/s | Decode 8K / 32K / 128K tok/s |
|---|---|---|
| A0 NOWAIT | 3905.9 / 4286.8 / 4276.1 | 115.00 / 111.93 / 112.67 |
| B LAG2 | 3897.9 / 4284.3 / 4271.9 | 117.00 / 116.13 / 115.23 |
| A1 NOWAIT | 3902.2 / 4283.6 / 4272.1 | 116.17 / 115.23 / 109.83 |
| B async | 3894.8 / 4283.0 / 4271.9 | 120.03 / 119.17 / 114.73 |
| A2 NOWAIT | 3902.8 / 4282.9 / 4271.3 | 120.00 / 114.03 / 112.97 |

Compared with the mean of the adjacent A arms, LAG2 changes decode by +1.23% / +2.25% / +3.58%; the corresponding signed control drift is +1.01% / +2.95% / -2.51%. Async changes decode by +1.65% / +3.95% / +2.99%, with drift +3.30% / -1.04% / +2.85%. Prefill changes are below 0.3%. Three samples per size do not separate every small gain from noise; these settings remain experimental. In particular, this result does not justify enabling async in a configured two-slot service or lifting its guards.

LAG2 and async forced-prompt logits are byte-identical to A0 at all three lengths: 248320 finite values per prompt, zero maximum error and equal top-1. Each candidate's task checks remain the same 5/6. This validates the sampled first-token distribution and output checks, not every generated token or a general model accuracy score. The async startup receipt confirms that the asynchronous tier actually activated. See [individual rows, standard deviations and receipts](evidence-public/lazy-r1/summary.json).

## Two-slot rotation and LAG2

All five arms and 45 cold requests are complete. Capacity stays at 10295 resident experts. A1 is shared between the two A/B/A comparisons; rotation and LAG2 were not combined.

| Arm | Prefill 8K / 32K / 128K tok/s | Decode 8K / 32K / 128K tok/s |
|---|---|---|
| A0 copy, NOWAIT | 3289.1 / 3593.4 / 3646.8 | 105.67 / 99.77 / 101.37 |
| B rotation, NOWAIT | 3277.8 / 3588.4 / 3644.4 | 105.67 / 104.80 / 103.13 |
| A1 copy, NOWAIT | 3304.6 / 3592.5 / 3645.8 | 98.90 / 98.90 / 94.90 |
| B copy, LAG2 | 3277.3 / 3589.2 / 3649.1 | 100.23 / 99.40 / 95.73 |
| A2 copy, NOWAIT | 3301.9 / 3591.9 / 3646.4 | 103.37 / 102.67 / 98.47 |

| Candidate | Prefill change, 8K / 32K / 128K | Decode change, 8K / 32K / 128K | Decode A-after/A-before drift |
|---|---|---|---|
| Rotation | -0.58% / -0.13% / -0.05% | +3.31% / +5.50% / +5.10% | -6.40% / -0.87% / -6.38% |
| Two-slot LAG2 | -0.78% / -0.08% / +0.08% | -0.89% / -1.37% / -0.98% | +4.52% / +3.81% / +3.76% |

Rotation meets the mean-versus-drift criterion **at 32K**, without a material prefill regression. Its 8K and 128K gains remain inconclusive: the control drift exceeds the observed change. The 32K decode sample standard deviations are A0 4.08, B 1.87 and A1 5.68 tok/s. Three trials are a bounded comparison, not proof of a universal or statistically guaranteed improvement. Rotation was accepted for the observed 32K benefit and retained correctness, then passed production feature checks. Do not claim a confirmed benefit at the other lengths. Keep LAG2 disabled: its two-slot means regress at all lengths.

Both candidates have bit-identical forced-prompt logits at all lengths, the same 5/6 task checks, full outputs and no screened loops. See [all means, standard deviations and drift](evidence-public/rotation-r1/summary.json).

The rotation startup confirms 96 mapped exchange buffers, 2764816 bytes each. The cold interval has 13876 committed exchanges, corresponding to **38364586816 bytes / 35.73 GiB** of skipped host commit memcpy payload. This is an event-counter-derived payload bound for the whole nine-request interval, including commits crossing request boundaries. Logical reads plus writes would be twice the payload; neither quantity is measured physical DRAM traffic. H2D and D2H remain. It does not establish how much critical-path latency came from that memcpy. The source moves ownership and retains stable host/device address pairs; it does not allocate another GPU expert cache. See [activation and exchange snapshots](evidence-public/rotation-r1/exchanges.public.json).

## Validation and attribution limits

Across the three series, 21 arms / 189 cold requests have matched input hashes, three trials per size, 512 output tokens and zero prefix reuse. All use GEN for speed requests. Numerical requests are excluded and use GEN at effective batch=0 or BGEN with two configured slots. The logits-dump environment variable only activates on an original one-token request in the matched loaded generate source; it does not dump a vocabulary row during these 512-token speed requests. GPU profiling is disabled. No new CUDA kernel or arithmetic path was built for this task.

The host exchange-storage regression exercises 12304 exchanges, invalid geometry/commits, incoming-byte preservation, guards, host/device alias pairs, buffer reuse and concurrent residency lookahead. Its large-blob case uses this checkpoint's 2764816-byte NVFP4 size. It passes against the current header. Transfer completion is simulated in that host test; GPU/CPU lifetime behavior is additionally exercised by the service trials. See [test receipt](evidence-public/exchange-storage-test.public.json).

NVML sampling reports peak device use 48318 MiB / minimum free 193 MiB in the original two-slot matrix, versus 48288 / 223 MiB in every single-slot arm. These observations can miss subsecond peaks. The mid-arm A1 native-process RAM snapshot reports VmHWM 47647380 KiB (45.44 GiB), RSS 47019268 KiB and no process swap; it includes mapped pages and is not the system RAM total or every arm's final peak. It cannot quantify DRAM contention. Existing PcieModel batch/memory guards remain intact; new per-layer scheduling is not justified by the inconsistent mode/fraction results or the historical unmatched wait/copy averages.

The measurement runner depends on the retained server request harness and fixtures. [Harness provenance](evidence-public/harness-provenance.public.json) records current hashes and pre-campaign filesystem timestamps; those timestamps are not a cryptographic before-snapshot. Public receipts retain the actual binary/profile/input identities. Raw prompts, generated responses and logits stay on the server.

## Production and rollback

The only semantic change to the live config is `env.STRATA_EXCHANGE_ROTATE: absent → "1"`. `gpu_schedule.sh` already reads `/home/tc/Strata-nvfp4/strata-qwen38-flash-next-nvidia-30001.json`; its bytes and the loaded binary are unchanged. The config SHA256 changes from `a93493f9…` to `f04d9b9d…`. Restarted native PID: 3612665. The deployment checked configuration ownership after the idle wait and before replacement, and retained the original bytes for recovery. The installer now emits the same rotation default.

Chat/Responses on both 30001 and the existing 30000 relay, red-image recognition, forced tool JSON and prefix reuse pass. Both short concurrent requests complete 256 output tokens; live running count reaches two and native admissions show slots 0 and 1. Prefix reuse is 8193 tokens. Capacity remains 10295 GPU experts, two slots, context 262144 and INT8 KV. PLE early/raw/64 MiB CPU-cache controls, W4A16/fused backend 2/R256/chunk8192, MTP, reasoning recovery and other arguments are retained. Miner remains off. See [deployment](evidence-public/accepted-rotation/deployment.public.json) and [live checks](evidence-public/accepted-rotation/service-check.public.json).

The short concurrent check verifies execution and retained features, not concurrent throughput. The unchanged GPU allocation code and previous same-binary two-full-context shared-prefix capacity checks are reused at their original scope; they were not rerun as a new full-context campaign. No new Nsight capture or critical-path attribution is claimed.

Exact byte-verified config and scheduler backups are in `/home/tc/Strata-nvfp4/bench/pcie-lazy-w4a16-20261013/accepted-rotation/`. To remove this setting, first stop the owned service, restore `backup-strata-qwen38-flash-next-nvidia-30001.json` to its original path **only if the current config still matches the recorded deployed hash**, then start with the unchanged `gpu_schedule.sh`. Preserve later operator changes. Model packs, expert profile, draft vocabulary and the PLE-enabled binary require no replacement.

An optional single-slot baseline config is saved on the server at `bench/pcie-lazy-w4a16-20261013/single-nowait-30001.json`. It retains context262144/chunk8192, explicitly normalizes native batch to zero and uses rotation=0, matching the measured single-slot A configuration. Its generation is recorded [here](evidence-public/single-config.public.json); this exact convenience file has not been separately restarted and is not the default. Start it manually only after stopping the existing service:

```bash
bash ~/wildrig/gpu_schedule.sh stop_strata
cd ~/Strata-nvfp4
.venv/bin/python -m serve.server --engine strata \
  --config bench/pcie-lazy-w4a16-20261013/single-nowait-30001.json --port 30001
```

## Reproduction and checks

The remote series used the retained binary/harness and private input fixtures. `matrix-r1` used the archived [runner v1](evidence-public/runner-v1.py); `lazy-r1` and `rotation-r1` used [run_matrix.py](run_matrix.py). Each run's before receipt binds its exact helper SHA256. On the prepared server, use a fresh run name:

```bash
python3 bench/pcie-lazy-w4a16-20261013/next-version/run_matrix.py \
  --plan bench/pcie-lazy-w4a16-20261013/next-version/rotation.json --run rotation-repeat
```

`rotation.json` explicitly sets rotation=0 for A/LAG and rotation=1 for the rotation candidate. The runner restores the then-current production configuration afterward. Other arguments, weights, profile and fixture identities must still match to reproduce these rows; do not overwrite later operator changes to force a match.

Local checks pass: seven orchestration/config/numerical helper tests, six repository tests, the host exchange regression, Python compilation, public means/standard-deviation/input/build/profile/restoration validation, and `git diff --check`. [Offline verification](evidence-public/verification.json) covers all 189 cold requests; it is separate from the live API checks.
