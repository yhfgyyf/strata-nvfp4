# W4A16 decode PCIe and lazy residency implementation

## Scope and acceptance

Apply the five corrections in the single-answer `gpt-6-astra:max` audit of 2026-10-13, then test the existing mechanisms on the NVIDIA checkpoint. Keep NVFP4 storage, fused MoE backend 2, W4A16 prefill, R256, chunk 8192, INT8 KV, context 262144, PLE early/raw upload and its 64 MiB CPU row cache. No INT8/FP8 expert arithmetic experiments, new quantization or SGLang changes.

Success means a measured decision for each tested mechanism, an intact production service with its accepted configuration, and individual measurements plus an arithmetic mean of three in the optimization report. A negative result is retained; it does not justify deploying a regression. GitHub publication and another Humanize loop are outside this request.

## 1. Resolve the executing service

The initial live inspection identified `/home/tc/Strata-nvfp4/strata-qwen38-flash-next-nvidia-30001.json`, **not** the similarly named historical uncensored config. The loaded binary is `engine/strata-nvfp4-w4a16-r256-ple`, SHA256 `4ae23d2d995866b3ddf98525b0de700d3fb48e289135fea39f5643d2cd55af78`. The server has two configured slots; an alone request ordinarily uses GEN. Record native diagnostic command receipts, effective batch settings and actual PCIe offload counts/share; a configured slot count is not proof of batch-mtp execution.

Back up the exact live configuration and `gpu_schedule.sh`, plus hashes of the binary, expert profile, model/PLE indexes and draft vocabulary. Leave model weights and profile unchanged. Check idle service and ownership immediately before stopping it. Run isolated localhost trials; restore production independently of candidate API health, and preserve intervening operator file changes.

The server root's `src/program/generate.cpp` and `src/core/expert_source.cpp` are older snapshots. They are not the source of the loaded port. The retained `step1-v3-r1/build/build.json` links the matching `4ae23d2d…` binary to `candidate-b8/generate.cpp.o` (SHA256 `5cca522b…`) and the upstream-port engine archive. Their recorded source hashes match this checkout: generate `41738aef…`, expert_source `efb88ac7…`. The server Python source differs only in CLI/config port precedence; trials explicitly pass their localhost port. Preserve the root sources and cite build provenance separately from root-source snapshot hashes.

## 2. Joint mode × fraction matrix

Compare `{kernel,dma} × {0,0.1,0.2}`. Keep the zero-fraction modes distinct. Fraction is converted to a numerator over 256 and applied to distinct misses, subject to pinning and staging capacity; verify that positive settings really offload experts.

DMA can submit on the copy stream before resident GPU work completes; kernel fetch executes on the compute stream after resident groups. Source comments record a CUDA driver-lock risk, not a reproduced failure on this GPU. Every combination first gets a short correctness request with an external timeout. A timed-out owned trial is terminated by its exact process group, then production recovery proceeds without requiring that trial's API.

For each candidate use A/B/A, with three matched 8K, 32K and 128K requests per arm and output 512. Use existing input hashes, three warmups, cold prefix reuse, sequential clients, the same expert profile and cache initialization, and unchanged speculation/reasoning settings. Keep `expert-cache auto`, record actual slot capacity, and disclose any allocation difference rather than attributing its effect solely to PCIe scheduling. Separate any diagnostic/profile capture from the service-rate rows. Record native prompt/decode rates, client TTFT, individual values, sample standard deviation, output count, task checks, repetition/long-digit loops and draft acceptance. Compare full-vocabulary first-token logits after identical forced prompts; retain finite/top-1, relative RMSE and cosine measurements without claiming a general model accuracy evaluation.

## 3. Existing lazy mechanisms, tested separately

Establish an effective batch=0 baseline with Python parallel=1 and explicit `--batch 1`, which the engine normalizes to zero. Native INFO omits `batch_slots` for o.batch=0; verify that contract, the actual API serving cap of one, and the GEN diagnostic command. Record the derived effective slot count explicitly. Python parallel=1 alone would also avoid adding native batch slots. Preserve context and chunk, and report the capacity/configuration change separately from mechanism gains. GEN reads NOWAIT/LAG; NOWAIT defaults on. Compare the baseline independently with `STRATA_ADAPT_WAIT=1` plus `STRATA_ADAPT_LAG=2`, and with the existing `--adapt-async 1` state machine. True batch-mtp directly queries pending copies nonblockingly, so single-client GEN evidence cannot establish its concurrency behavior. The production default retains its existing two-slot capacity; a single-slot configuration is a separately measured alternative.

Reuse `stage_exchange`, `commit_exchanges` and asynchronous `RamFence`. D2H writes independent pinned exchange memory; incoming RAM stays immutable until CPU, H2D and mapped GPU readers finish. GPU slots are overwritten only after old readers finish and residency stops planning them. Publish new mappings only after the copy completion event; reclaim RAM only after reader fences. PCIe staging is temporary and is not a stable resident spare. The existing state machine uses controlled retirement rather than inventing free GPU shadow slots.

Budget each 2,764,816-byte expert separately: GPU staging/spares, pinned exchange buffers and bounce buffers. Record exchange capacity and startup/peak sampled VRAM/RAM. The historical 338 MiB startup free space is not a free-spare guarantee.

The current source and loaded binary also contain `STRATA_EXCHANGE_ROTATE=1`: for uniform fully mapped/pinned expert blobs it transfers ownership of the RAM exchange buffer rather than copying the evicted blob into the incoming blob's former address. Test this separately against the same two-slot baseline if the matched swap evidence shows host-copy work. It preserves the same completion contract and existing spare allocation; it is not a speculative GPU shadow cache. Fall back when eligibility fails and retain the startup receipt proving which path ran.

The first single-slot LAG2 comparison gave small positive decode means, pending the baseline return. Since the production two-slot configuration also executes GEN for this workload, add a separate two-slot LAG2 A/B/A to the rotation series, sharing only its unchanged A control. This tests the setting on the actual production capacity without treating true batch-mtp as GEN or relaxing async guards. Do not infer its gain from the single-slot configuration.

## 4. Evidence-gated per-layer scheduling

The existing `PcieModel` already fits CPU work, PCIe-induced DRAM contention and GPU cost. Automatic balancing and async adaptation are restricted by configured batch; do not merely remove guards. `run_slot_rows` lacks ordinary `run()` cost sampling/solving. `STRATA_ADAPT_FETCH` is incompatible with the current resident RAM complement even after batch is removed. Keep it disabled.

Only extend scheduling if the joint matrix and existing lazy tests show a remaining opportunity with matched per-layer/job/transfer evidence and a safe memory budget. The historical 59.64 µs CPU barrier exclusive exposure and 171.79 µs full adaptive H2D copy duration have different populations and overlap definitions; neither sets a PCIe fraction threshold or estimates removable latency. Source-supported Marlin/CUTLASS alternatives likewise do not establish the current SGLang runtime.

## 5. Deployment and verification

Choose a candidate only if full outputs, matched quality checks and numerical diagnostics remain acceptable and mean decode improvement exceeds baseline drift without a material prefill regression. Keep the current defaults when no candidate passes. Apply the minimal accepted config change with a byte-verified backup, restart the owned Strata service, verify loaded binary/env/args, API model/Chat/Responses, and the retained image/tool/prefix features affected by service reconfiguration. Append results and rejected/inconclusive cases to `strata-workspace/STRATA_OPTIMIZATION.md`.

## Execution outcome

- Completed the six-point joint matrix with five bracketed candidates, then the separate single-slot lazy and two-slot rotation/LAG series: 21 arms, 189 cold speed requests, three trials per length and 512 outputs. Decisions and drift are in REPORT.md.
- Retained original PCIe mode/fraction, NOWAIT and configured two-slot capacity. Async batch guards and the existing cost model guards remain unchanged; no new per-layer scheduler was added without matched execution evidence.
- Accepted existing RAM rotation for the observed 32K mean gain above control drift; 8K/128K gains remain inconclusive. Backed up and changed one live config field, restarted and passed ten feature/relay/prefix/two-slot checks. Installer default synchronized.
- Retained private raw outputs on the server, verified public measurements and recovery receipts, and updated the optimization document. No Humanize loop restart or GitHub publication.
