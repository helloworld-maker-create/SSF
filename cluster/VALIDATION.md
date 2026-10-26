# Handoff validation

- Full original model/source snapshot:95files matched. Only.gitignore differs for packaging exclusions; no model,State,DMD,KV,scheduler math was changed.
- Archived runtime/contracts/vendor:51files pass SHA256 andsize verification.
- NativeB64 accumulation CPU regressions:5tests PASS, including arithmeticmean,AdamW mapping,single clip/step,rank averaging withoutdoubledivision,shared exit rule,and200/400/600 EMA counters. Command:PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' self-python -B -m unittest test_accumulation_cpu test_sync_candidate_cpu -v, inoriginal_run/scripts.
- Full rootpytest suite was NOT rerun:the intended training Python has no pytest. An attempted unittest discovery failed to importpytest-based tests;this is not reported as a modeltest PASS. No package was installed or upgraded.
- Standardlibrary staticpreflight PASS; it compilesall recorded.py files without importingmodels.
- Gitpayload credential/largefile/model/video guard PASS beforecommit.
- No GPU model load,training,generation,evaluationornewruntimequalification was executedfor thishandoff.
- Existingworld2/H20 G1 evidence remains historicalonly;8-cardclusterqualification has NOT occurred.

Archival scripts retainoriginalhostpaths andgrantchecks intentionally. Cluster resource/path/launcher adaptation remains required; do notrunoldlauncher unchangedor fabricateoldauthorizations.

- CPU-only complete DMD module import was BLOCKED by nativewan/modules/t5.py import-time torch.cuda.current_device(). NoGPU was exposed, no modelweights loaded. This is documented inIMPORT_AND_PORTING_NOTES.md; no modelsource workaround was introduced.
- Full localGitbundle was cloned offline intoan independentdirectory; all95source/51archive integrity checks passed inthatclone.

## Portable launcher verification (2026-10-26)

- Added `launch.sh`, a standard-library SSH controller, per-node probe, source deployment, native GPU qualification, and portable training/checkpoint/monitor modules. Historical `original_run/` files remain unchanged.
- `utils/wan_wrapper.py` now has an optional asset-root path adapter. Its exact hash is recorded separately under `portability_overrides`; the historical 95-file byte-identity statement above describes the original handoff, not this adapter.
- Launcher CPU suite: 15 tests passed (`python3 -B -m unittest discover -s cluster -p test_launch.py -v`), including qualification-before-training, failed qualification and cost gates, shared-output failure, SSH quoting, B64 topology restrictions, balanced monitor calls, source/receipt isolation, and owned-process cleanup.
- Original accumulation/EMA CPU regressions: 5 tests passed in the available fastvideo Python with `CUDA_VISIBLE_DEVICES=''`; no model import, CUDA allocation or native DMD execution was needed for these tests.
- Standard-library preflight: 95 recorded source files, 51 archived files and 12 portable Python entry files checked/compiled. Only the documented packaging and path overrides differ from the historical snapshot.
- Live `gpuall` found 32 online nodes. The new read-only node probe successfully selected 0014 (`10.191.1.38`) after asset-size, package metadata, GPU/RAM and persistent-output checks. Other nodes' missing assets, inaccessible Python, occupied devices and insufficient storage were reported individually.
- Source-only rsync deployment to 0014 succeeded; all 183 manifested files matched after deployment and the remote static preflight passed. No weights were copied or downloaded; no model was imported or GPU workload launched.
- Full asset SHA, multi-node shared-output handshake on a real shared filesystem, native DMD qualification, GPU checkpoint restore, video generation and 600-outer training have NOT been executed for this change. CPU/controller checks do not constitute a cluster runtime PASS.
- After the user's remote-update notice, fetched all remote heads again: main remains `6e8fd0db76fde73a64eeb548b3e31c070d705816`, the direct parent of this launcher work. Its offline evaluation and frozen 48/96/944 prompt data are preserved; all 18 evaluation CPU regressions also passed. Combined targeted CPU checks: 38 passed (15 launcher + 5 accumulation + 18 evaluation).

## Direct launch requested by user (2026-10-26)

- Default `launch.sh` now starts baseline after necessary resource/source preparation. The long native-DMD qualification and full weight SHA scan are opt-in via `--with-qualification` / `--qualify-only`; skipped qualification is recorded explicitly and is never labeled PASS.
- No further test suite was run for this simplification; only entry syntax/help was checked.
- Executed a real direct startup on 0014 (`10.191.1.38`), 4 H100 GPUs, run `20261026_112356_02ef0c41`. Weights loaded, then the first training forward failed at `wan/modules/attention.py:116`: `RuntimeError: unflatten: Provided sizes [1, 4680] don't multiply up to the size of dim 0 (12) in the input tensor`.
- Workers exited and GPU allocations were released. Per-rank logs remain in the remote run directory. The controller captured a local structured error report under ignored `cluster/run_receipts/`; no automatic retry or model-code workaround was applied.
