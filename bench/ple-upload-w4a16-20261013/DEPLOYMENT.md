# PLE production defaults

## Plan

The user authorized deployment on October 13, 2026, followed by a pull request and merge into `yhfgyyf/strata-nvfp4:main` after verification.

1. Verify the current scheduler/configuration and the matching tested candidate; back up the scheduler, configuration, executable and build record before stopping the owned service.
2. Install the candidate under `engine/strata-nvfp4-w4a16-r256-ple`. Enable early PLE upload, raw transport and a 64 MiB CPU row cache in the scheduler/configuration. Remove the explicit `--vram-reserve-mib 1536` pair. Preserve all other arguments, model data, gateway and miner-off state.
3. Restart using `gpu_schedule.sh`. Verify the loaded executable, effective environment, W4A16/R256/chunk8192, context262144/two slots, Chat/Responses/image/tool/prefix behavior and long-context memory capacity. Recover the backed-up configuration and scheduler if verification fails.
4. Save public checks without prompts or generated text, update the deployment result, run relevant repository tests, then create and merge the PR.

Removing the explicit reserve uses the engine's automatic reserve policy; it does not set the reserve to zero. The C++ switches retain their opt-in defaults, while the model-specific installer enables them. Historical A/B/A results remain unchanged and do not establish the performance of this new combined configuration.

## Result

Verified on October 13, 2026 at 09:55:33 UTC. `gpu_schedule.sh` now starts `engine/strata-nvfp4-w4a16-r256-ple`, binary SHA256 `4ae23d2d995866b3ddf98525b0de700d3fb48e289135fea39f5643d2cd55af78`. Its actual native process is PID 3505071. The scheduler and configuration both enable early upload, raw transport and the 64 MiB row cache. Comparing the restored view of the new configuration with the backup confirmed that only the executable, these three controls and the explicit reserve pair changed.

The config SHA256 is `a93493f9f8983c31e75441be5cedc25aa88fd7b0dc212bad823d3696ea3c9b20`; scheduler SHA256 is `dad2b50badccd179a1e5b99c3d401f9252a0b8404d254259361372fb326f360f`. GPU expert slots rose from 9978 to 10295. The engine reports its 700 MiB automatic reserve and 338 MiB free at startup. The final NVML observation was 135 MiB free; these are observations, not a measured allocation-peak guarantee.

Validation:

- Chat, Responses, image recognition and tool-call checks passed; the existing port 30000 gateway passed Chat/Responses checks.
- One cold full-512 request at each of 8192, 32768 and 131072 input tokens passed. All three retained five of six synthetic quality checks, consistent with the controls' existing code-result error; periodic-tail and long-digit-loop screening passed.
- Two distinct full-context branches sharing a pinned prefix each completed 261624 input plus 512 output tokens. Native running slots peaked at two and prefix reuse was observed. This verifies shared-prefix capacity, not concurrent cold throughput.
- Installation: 6 tests passed. Serving configuration/VRAM/Responses: 55 tests, 1 skipped. Packing: 4 tests passed. The retained 393 artifact hashes and 135 historical samples also verify. Missing local regex/Torch dependencies were resolved by running the tests in existing server environments; no dependencies were installed.
- The new binary matches the retained Linux/CUDA 13.2 SM89 differential build and the current seven changed source/header hashes. This does not claim a fresh full-tree build or synchronization of the server root's historical source checkout.

| Cold deployment sample | Prefill tokens/s | Decode tokens/s | Output tokens |
|---:|---:|---:|---:|
|8192|3335.50|87.1|512|
|32768|3607.22|94.6|512|
|131072|3660.10|95.8|512|

These are individual smoke requests after a fresh load, without the historical campaign warmups. They must not be compared with three-trial averages as a speedup or regression estimate. The new combined settings and smaller reserve have not received a separate A/B/A campaign.

Public receipts are in [evidence-public/production-20261013](evidence-public/production-20261013). Run `python3 bench/ple-upload-w4a16-20261013/verify_deployment.py` to check them separately from the original campaign manifest. Weights, prompts, generated text, private controller logs and raw traces remain on the server. Miner remains off. Humanize remains stopped.

## Backup and rollback

The original scheduler, configuration, executable and candidate build record are backed up in:

```text
/home/tc/Strata-nvfp4/backups/ple-default-20261013T095028Z
```

The original executable also remains at `engine/strata-nvfp4-w4a16-r256`. To roll back on this host after waiting for active requests to finish:

```bash
bash /home/tc/wildrig/gpu_schedule.sh stop_strata
cp /home/tc/Strata-nvfp4/backups/ple-default-20261013T095028Z/strata-qwen38-flash-next-nvidia-30001.json /home/tc/Strata-nvfp4/strata-qwen38-flash-next-nvidia-30001.json
cp /home/tc/Strata-nvfp4/backups/ple-default-20261013T095028Z/gpu_schedule.sh /home/tc/wildrig/gpu_schedule.sh
bash /home/tc/wildrig/gpu_schedule.sh start_strata
```

These commands restore this deployment's exact backup. Compare current configuration/scheduler hashes first if an operator has subsequently changed them.
