# W4A16 PLE upload and row cache

Status: implementation and the requested comparisons completed.135 cold full512/reuse0 samples were retained across five campaigns. Those campaigns restored the original production W4A16 R256 binary. On October 13, the user subsequently authorized enabling all three PLE controls in production and removing the explicit 1536 MiB reserve; deployment and validation are recorded separately in [DEPLOYMENT.md](DEPLOYMENT.md).

## Scope

NVIDIA Qwen3.8-Flash-Next-NVFP4, one RTX4090 SM89 with49140MiB, chunk8192, context262144, two server slots and one benchmark client. Expert arithmetic stays W4A16; fused MoE decode, INT8 KV, MTP, prefix, vision, reasoning and tool handling are preserved. No INT8 expert or FP8 expert-GEMM campaign runs here. Raw FP8 PLE bytes are expanded to the original FP32 values before the existing projection.

The public source starts at888172d552e5c90dafc8f5f91a3bfe5fbbd660ef. The configured production binary is1b827daf441e701d17b12c77a67cee9b2ead8015288d1392e191f063aa469c39. The server root source is stale; the matching retained `candidate-b8` source/objects/libraries reproduce that executable byte for byte when relinked. Candidate builds replace only the changed objects and retain the original fused MoE libraries. Production configuration, executable, profile, indexes, vocabulary and scheduler remain protected and backed up.

## Measurement rule

For each independent change, A1/B/A2 each run three matched fixed input seeds at8K/32K/128K, actual512 output tokens and zero prefix reuse. Each arm first runs three full-vocabulary first-token numerical checks and three warmups. The requested statistic is the arithmetic mean of the three rates; individual rows, sample standard deviations, mean and paired A/A drift remain in the public summaries. B's change is relative to the mean of A1 and A2 arm means. This is not a confidence interval or a general model accuracy evaluation. Instrumented requests are separate from throughput measurements.

## Implemented controls

| Control | Effect | Native engine default |
|---|---|---|
| `STRATA_PLE_EARLY_UPLOAD` | Dedicated upload stream, ready/consumed generations, bounded future-layer expert submissions until PLE is ready on the device |0|
| `STRATA_PLE_RAW_UPLOAD` | Upload raw E4M3 bytes, exact FP32 expansion/scale on GPU, existing FP32 projection |0|
| `STRATA_PLE_RAW_CACHE_MIB` | Bounded eight-way set-associative raw-row cache, LRU within each set, stable within-batch dedup and restoration before upload |0|

The model-specific installer and current production configuration enable the two upload switches and use a 64 MiB cache. The native engine defaults remain opt-in for other callers. Performance rows below describe their saved configurations; statements about retaining production defaults refer to the campaign decision before the later authorized deployment.

The raw transport requires a mapped E4M3 table with a finite nonnegative scale. With raw upload disabled, Direct IO and other formats retain their original path; explicit raw upload rejects unsupported tables. Cache keys are logical row IDs in the owning table generation; misses retain the physical shard map. Closing/reopening clears cached rows. This cache is separate from the OS page cache.

## Early FP32 upload

Step1 binary71a758dece4752568864b1c1b07b839b700736073a5da401a13af1096e79ebc8. All27 cold requests completed512 output tokens with reuse0. All three numerical lengths match A1 bit for bit,248320 FP32 logits each.

| Input | A1 prefill | B prefill | A2 prefill | B change | A1 decode | B decode | A2 decode | B change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8K|3193.78|3210.47|3200.83|+0.41%|95.70|98.00|100.20|+0.05%|
|32K|3471.57|3482.26|3465.19|+0.40%|95.33|98.43|98.57|+1.53%|
|128K|3524.36|3538.35|3522.31|+0.43%|95.20|98.43|98.87|+1.44%|

Rates are tokens/s, arithmetic means of3. The largest paired decode A/A drift is10.50%; its apparent changes cannot be attributed to the new upload schedule. Prefill gains are below the existing3% acceptance threshold. Keep this control disabled by default.

In the two independent8K Nsight diagnostics, the actual PLE payload is83,814,400B because the batched portion has8185 tokens after scalar seed processing. The nominal full8192 payload remains80MiB.

| Metric, relative to prefill NVTX start | A | B |
|---|---:|---:|
|Host copy submit|66.884ms|35.782ms|
|Device copy start|96.291ms|37.829ms|
|Host submit to device start|29.407ms|2.047ms|
|Copy duration|4.122ms|4.104ms|
|Copy overlap with visible compute|0ms|4.086ms|
|Prefill NVTX window|2318.722ms|2293.030ms|

The critical copy moves earlier and overlaps compute. The remaining copy queue is about2ms in this v1 capture. Later matched diagnostics revealed a different arbitration case; this observation alone was insufficient to validate the submission-based credit release. One profiled request per arm does not replace the unprofiled three-trial result. No layer-level GPU NVTX or CPU sampling was collected; activity overlap is not removable causal latency.

The user's subsequent request to compare against the original R256 capture found an additional critical upload stall:

| Batched prefill, ms | Original R256 | New control | Early PLE v1 |
|---|---:|---:|---:|
|Window|2303.145|2318.722|2293.030|
|H2D only|563.456|578.608|551.663|
|Compute only|60.022|74.991|47.302|
|Compute/H2D overlap|1665.799|1650.975|1680.061|

The same8K payload hash, profile hash,9957/1480/8477 S/L/P and16099 streamed expert copies match. Against the new control the window saves25.693ms; against the old R256 trace it saves10.115ms. The latter is approximately0.44%, consistent in size with the separate unprofiled result, without making cross-run timing a causal proof.

The QSA step records are130960B. Host submission is about0.10ms in all captures. They land at1.843ms originally,37.749ms in the new control and37.820ms with early PLE. Main-stream `gr_norm_rs` starts only after this copy:1.859/37.766/37.834ms. The new traces finish expert256 at37.748/37.819ms, then pause17.421/17.489ms before expert257; the original gap is0.001ms. This is a separate head-of-line stall before layer0 compute, present even with early PLE disabled. The exact reason for the different cross-stream arbitration across retained runs is not established by these captures.

`min(H2D-only, compute-only)` estimates possible additional overlap under unconstrained independent scheduling; it is not a guaranteed minimum latency saving. Here, early PLE fixes one dependency while the step-record copy still blocks the main compute stream. Revisionv2 moves this metadata submission before embeddings under the same opt-in early-upload control. The v2 diagnostic confirms the metadata reorder, but PLE submits at26.786ms and executes only at81.741–85.922ms, with no compute overlap. Future-layer credits were released at host submission rather than device readiness. Revisionv3 instead releases them after the ready event completes, keeping current-layer expert progress and the original consumed-generation protection. Its own timeline and three-trial measurements are complete below; v1/v2 results are retained separately.

## Ready-event revision diagnostic

Revisionv3 binary4ae23d2d995866b3ddf98525b0de700d3fb48e289135fea39f5643d2cd55af78 keeps current-layer expert DMA eligible, uploads QSA step records before embeddings, and holds future-layer credits until the PLE copy/expansion ready event actually completes. The host worker blocks on a blocking-sync event; it does not synchronize the main compute stream. Host-submitted and device-ready generations are distinct.

The CUDA ownership fixture passes pinned/pageable buffers, three chunks per request, three consecutive requests, a129-element tail, guards, cancellation and three injected gather/copy/event failures. Numerical/service validation completed separately below.

| Batched8K prefill diagnostic, ms | Original R256 | v3 control | v3 early/ready |
|---|---:|---:|---:|
|NVTX window|2303.145|2316.154|2283.081|
|H2D only|563.456|576.292|524.136|
|Compute only|60.022|74.621|33.345|
|Compute/H2D overlap|1665.799|1651.097|1711.034|
|QSA step GPU start|1.843|37.753|0.060|
|Gap after expert256|0.001|17.434|0.001|
|PLE host submit|34.564|66.747|36.501|
|PLE device start|81.279|96.260|46.347|
|PLE copy duration|4.092|4.119|4.185|
|PLE copy/compute overlap|0|0|1.980|

Original and revised traces share input/profile and the same9957/1480/8477 S/L/P, nominal chunk8192 and actual8185-token batched segment. They are separate captures, not synchronized layer ranges. In the first160ms, compute-only falls32.922→5.915ms versus original R256, close to the user's roughly30ms initial-region estimate. The total window falls20.064ms rather than the entire27.007ms difference: aggregate kernel/copy durations and overlap elsewhere also differ. H2D union increases7.218ms and compute union18.536ms across those runs; subtracting overlap yields the observed active-union change, rather than treating exclusive-time reduction as a guaranteed latency saving.

The new control/revision window difference is33.073ms. Only the unprofiled repeated service comparison can establish its prefill/decode effect. The original R256 and v1/v2 traces remain preserved; they are not relabeled as revisionv3 evidence.

Revisionv3's unprofiled A/B/A completed27 full512/reuse0 rows. Three complete first-token logit vectors are bit-identical for B and A2 against A1. The six-task/loop screening is unchanged.

| Input tokens | A1 prefill | B prefill | A2 prefill | B change | Prefill A/A drift | Decode A/A drift |
|---|---:|---:|---:|---:|---:|---:|
|8192|3181.78|3232.45|3183.11|+1.57%|+0.04%|+1.16%|
|32768|3457.03|3520.04|3467.04|+1.68%|+0.29%|+3.33%|
|131072|3513.96|3574.98|3515.35|+1.72%|+0.04%|+8.20%|

Rates are arithmetic means of3. This establishes a repeatable prefill improvement against the revised binary with its early flag disabled, while decode changes remain confounded by control drift. A separate direct comparison against the retained production1b827daf binary follows below; matching on/off code alone was insufficient to establish performance against that executable.

## Raw PLE transport and cache

Step2/3 candidate9f6b204ac409ccf3d1b15a184ede0717dc8f0f6d8571542e8481f906acdd3f35. Production defaults are unchanged after the comparisons.

The GPU parity fixture passes all256 encodings,8 scales including the actual model scale, signed-zero/special-code behavior, every actual logical shard boundary sampled, repeated/near-unique rows, OOB zeros and cached scalar/batch equality against the original uncached reader. Its small gather timings are diagnostics with naturally warmed OS pages, not the three-trial service comparison.

At a full8192 chunk, raw H2D is20MiB rather than80MiB. FP32 GPU output still uses80MiB; the raw GPU staging adds20MiB in both `bytes_needed` and `carve`. Host double buffering falls from160MiB to40MiB. These are capacity and byte changes, not guaranteed latency gains. Extra borrowing can reduce the effective resident expert set; actual loan/residency ledgers must be retained.

The64MiB cache budget includes its vector entries and cache object; each entry contains160 payload bytes plus16 bytes of key/age/padding on this build. The mutex/counters, transient dedup map, allocator bookkeeping and OS pages are separate. Transient map rows are bounded by16*T; emitted minimum bytes exclude node links and allocator overhead. VmRSS, lifetime VmHWM and MemAvailable are observed separately. A cache-owned budget is not an RSS cap.

Restore-before-upload keeps16*T*160 raw bytes and the same number of decoded elements. Removing repeated GPU conversion would require a separate unique-row GPU restoration design; it is not implemented here. The raw128K/case2 output screens6/6 versus5/6 for both controls; the other44 matched pairs across the five campaigns are unchanged. This is a generated-trajectory variation, not evidence that raw transport improves model accuracy. All135 cold samples have no detected periodic-tail or long-digit loop.

Raw transport service A/B/A completed27 cold full512/reuse0 rows. All three full-vocabulary numerical lengths match A1 bit for bit for B and A2. Early upload and row cache are disabled in both raw arms.

| Input | A1 prefill | B prefill | A2 prefill | B change | A1 decode | B decode | A2 decode | B change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8K|3180.55|3178.70|3189.65|−0.20%|97.03|98.77|102.03|−0.77%|
|32K|3461.30|3474.38|3456.59|+0.45%|96.20|98.13|95.10|+2.60%|
|128K|3516.73|3521.86|3523.92|+0.04%|94.27|96.90|95.57|+2.09%|

Rates are arithmetic means of3. These prefill changes do not establish a useful gain. Decode control mean drift reaches5.15% at8K; do not attribute the decode differences to raw transport. Numerical equality and the six-task/loop screen are separate from general model accuracy.

The child campaign restored the exact production baseline. Its outer driver then failed a post-success ownership check because it inherited a historical executable path. That failure is retained; a fresh check using the actual R256 executable verified loaded state and all protected hashes in `step2-verified-after-driver-check.public.json`. No speed rows were rewritten.



## CPU row cache0/64MiB

Both cache arms use raw transport and disable early upload. Only the CPU cache budget changes.

| Input | A1 prefill | B prefill | A2 prefill | B change | A1 decode | B decode | A2 decode | B change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8192|3169.90|3202.98|3173.99|+0.98%|101.97|100.97|101.03|-0.53%|
|32768|3460.37|3467.40|3463.13|+0.16%|99.37|98.67|98.27|-0.15%|
|131072|3523.88|3519.16|3520.70|-0.09%|97.40|96.73|99.83|-1.91%|

Arithmetic means of3, tokens/s. Per-case rates, standard deviations and control drift are in `evidence-public/step3-aba-1/means.public.json`.

## Retained production versus revisionv3

A1/A2 run the exact retained production1b827daf binary; B runs4ae23d2d with early upload on and raw/cache off. All27 cold rows are complete. First-token logits match bit for bit at all three lengths; all nine paired quality screens are unchanged.

| Input | A1 prefill | B prefill | A2 prefill | B change | A1 decode | B decode | A2 decode | B change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8192|3178.30|3243.52|3206.28|+1.60%|103.87|102.40|99.40|+0.75%|
|32768|3462.88|3524.07|3470.94|+1.65%|101.57|99.47|99.00|-0.81%|
|131072|3516.92|3576.71|3520.26|+1.65%|98.70|98.83|99.87|-0.45%|

Arithmetic means of3, tokens/s. Per-case rates, standard deviations and control drift are in `evidence-public/legacy-accept-aba-1/means.public.json`.

## Capacity and transfer ledger

The raw/cache candidate and the combined early/raw/cache configuration passed image recognition, forced tool JSON, prefix reuse and the two-slot full-context fixture. Both full-context requests consume261624 input tokens and512 output tokens,262136 total. Native slot logs observe2 running slots; frontend sampling peaks at1 in-flight, so this is capacity evidence rather than a concurrent-throughput result. Combined first-token logits match the original arithmetic contract at8K/32K/128K.

The combined8K ledger uses nominal layout8192 and actual8185 batched tokens. It records S9978, L1487, P8491, workspace4,111,025,664B and loan span4,111,638,272B. Physical expert stride is2,765,056B; the streamed expert payload is2,764,816B. The control layout's4,090,053,888B workspace needs1480 slots, so raw staging adds20MiB plus256B alignment and7 borrowed slots here. Do not infer a fixed8-slot increase from the nominal buffer size.

Raw rows:130960 occurrences,7800 fetched unique keys,7656 cache hits and144 mapped backing fetches /23040 logical bytes. Raw H2D remains20,953,600B and GPU expansion20,953,600 elements; FP32 output is83,814,400B. These warmed mapped backing fetches are not physical SSD IO measurements. Cache-owned67,108,128B is within64MiB, of which61,007,360B is payload. The transient dedup map records7800 entries and136607 buckets; its emitted minimum1,217,656B excludes node links/allocator overhead. Host RSS/HWM and one-second NVML sampling remain separate observations; they are not complete subsecond peak bounds.

## Decision and verification

Keep all three controls disabled by default and retain the original1b827daf production binary/config. Revisionv3's prefill gain is supported against production on this workload, around1.60–1.65%, with all nine matched changes positive. It is a useful optional schedule. Whole-service acceptance remains limited: paired decode changes range−6.23% to+5.98%, and control mean drift reaches4.30% in the production comparison. Raw transport has no consistent prefill gain; cache gains0.98% at8K,0.16% at32K and−0.09% at128K. These results do not justify forcing new defaults or claiming a decode improvement.

The differential native builds use Linux/CUDA13.2 SM89, reproduce the original executable when relinking retained inputs, and replace only the specified objects. Fused MoE libraries and expert arithmetic are retained. This is not a freshly rebuilt full upstream tree; the server root source was stale and was not substituted for the matching source packet. Host fence/cache checks, all-code GPU parity, CUDA canaries and three recovery failure-injection tests pass. Six existing repository tests pass with the bundled Python runtime; system Python3.9 initially lacked the already-required jinja2 dependency, and no production dependency was added. Real engine mid-request cancellation/fatal-device-error injection was not run; cancellation/failure evidence is the stated fixture scope plus successful request/loan reuse.

At2026-10-13 08:23:24 UTC the exact original production binary is loaded on30001, with protected configuration/profile/index/vocabulary/scheduler hashes unchanged. Chat Completions, Responses, image input, tool JSON and both30000 gateway routes passed. Native PID3470324, API3470308; GPU processes are the native model and its vision worker. Miner remains off. The current Humanize audit was cancelled by the user; no new audit was started. Read-only boundary inventories and miner guards found no competing GPU workload; they are not a continuous system-wide process trace.

The earlier raw campaign's outer-driver path error and the negative v2 scheduling diagnostic are retained as described above. The initial macOS recovery fixture needed its temporary-directory symlink resolved before its injections reached the lifecycle boundary; all three repaired fixture checks pass. No incomplete/failure row was discarded and no profiler timing substitutes for the unprofiled means.

Public JSON/JSONL, source/build hashes and row recomputation are checked by `verify.py`; native reports, SQLite, prompts, output text, weight rows and logit arrays stay on the server. `index.html` displays the original R256 alongside v1 and v3 diagnostics. GPU row caching, upload-only-unique rows and broader expert scheduling changes remain outside this implementation.
