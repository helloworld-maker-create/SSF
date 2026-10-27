# Automatic result publisher validation

Validated with `PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''` and `-B` in the qualified local Python.

- 23 publisher tests pass, including a real, completely offline local bare-Git roundtrip: create a dedicated result branch, append a second snapshot, retain the first, and identify an already-published snapshot without a duplicate push.
- A second real local-Git case advances the remote ref concurrently; ordinary push is rejected, the other writer's head remains intact, the package remains local, and no force/retry occurs.
- Tests cover incomplete appended JSON lines, missing ranks, rank divergence, resumed-attempt counting, native accepted update order, nonfinite metrics, credential patterns, symlinks, modified package hashes, prohibited weight artifacts, command forwarding and upload failure without terminating the experiment.
- Existing 18 evaluation CPU regressions pass unchanged (41 targeted CPU/local-Git tests total).
- All 95 historical qualified source files and 51 archived runtime files retain their recorded hashes. Only new publisher files and a handoff-document link are added.
- The public payload scanner passes. No model/checkpoint/video/training-data/credential file is in this code commit.

No GPU training, model generation, evaluation, background watcher, SSH/controller job or scheduled service was started. No production source or running training configuration was changed. The collaborator's `wubin` branch is not merged or overwritten.

Not claimed: live server Git authentication or remote runtime qualification. The collaborator must use their own already-authorized server Git credentials once. The implementation uses existing Git/stdlib only and does not export this conversation's connector credentials.
