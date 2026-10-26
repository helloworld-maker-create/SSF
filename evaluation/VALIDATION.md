# Evaluation handoff validation

- 18 CPU-only unittest cases pass using the qualified Self-Forcing interpreter with `PYTHONNOUSERSITE=1`, `CUDA_VISIBLE_DEVICES=''`, and `-B`.
- Frozen 48/96/944 panel IDs, exact prompt text/order and source SHA256 verified. All 16 official dimension memberships, including auxiliary metadata, checked against the archived official metadata.
- Tested: Human Action hardlink/name contract; raw/EMA mismatch rejection; initial/all-renoise mismatch rejection; missing inputs; no missing-to-zero conversion; MUSIQ units; numerical tie tolerance; prompt-first seed averaging; partial result labeling; immutable evidence writes; video shape/RGB contract; external-GPU-process rejection; metric asset template coverage.
- `evaluation/run.py plan --config evaluation/config.example.json` reports the expected 192 existing videos and dimension-specific memberships without loading a model or touching GPUs.
- `cluster/preflight.py` verifies all 95 qualified source files and 51 archived runtime files remain unchanged, with the previously documented packaging-only `.gitignore` override.
- Code-only Git payload guard passes. Model weights, scoring weights, original videos, private local configs and training datasets are excluded.

Not performed: GPU scoring, new model video generation, training, dependency installation/upgrade, or a claim of new-cluster runtime qualification. The new worker calls the installed verified VBench `compute_<dimension>` functions; it does not substitute a mock metric. Missing local metric assets/dependencies or source-version mismatches stop execution with a traceback.

CPU tests exercise fixtures and metadata; they are not evidence that all third-party metric kernels/dependencies will execute on an as-yet-unqualified cluster environment.
