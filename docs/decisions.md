# Baseline decisions

Status: initial ML milestone, 26 October 2026. This implements the model/data portion of the v1.1 project plan; future product defaults remain subject to the documented gates.

| Decision | Baseline and reason |
| --- | --- |
| Audience | Experimental educational/portfolio demo. Exact document transcription is not guaranteed. |
| Dataset | Official EMNIST ByClass, preserving 62 distinct labels. Correct raw-image transpose once during preprocessing. |
| Data partitions | Deterministic stratified validation from official training data. Official test data remains untouched until model selection is frozen. |
| Model | Conditional convolutional VAE from the first product run. Condition both encoder and decoder on the requested label. Random initialization only. |
| Latent / loss | Start at 32 latent dimensions, pixel-summed BCE and dimension-summed KL; record scaling and warmup. Tune only on validation. |
| Local hardware | NVIDIA GTX 1650, 4 GiB VRAM, driver 566.36 detected. CUDA 12.6 wheel import failed with Windows error 1455 (paging memory). CPU wheels selected for the working baseline; Windows settings were not changed. No paid compute authorized or used. |
| Training ownership | The project owner runs all model training manually. Do not launch smoke training, full training or weight-update tests automatically. Default tests skip optimizer/learning checks unless the owner explicitly uses `--run-training`. |
| Run budget | Prepare commands for a bounded real-data smoke run followed by a manually launched baseline. Do not describe smoke output as release-quality training. Pilot timing informs the owner's full-run budget. |
| Checkpoints | Versioned contracts and weights-only loading; reject alphabet/preprocessing mismatches. No arbitrary pretrained checkpoint imports. |
| Punctuation | Proposed product exception: draw procedural punctuation rather than misrepresenting it as EMNIST-generated. Revisit before compositor acceptance. |
| Model gate | Inspect generated prior samples across all labels and a composed paragraph before UI polish. A training-loss decrease is insufficient. |

## Decisions still open

- Canonical passage, visual reference and frozen human-review rubric.
- Maximum full-run experiment count/training time after a measured pilot; paid-spend cap remains zero unless separately authorized.
- Final quality thresholds and whether the initial supported alphabet must be narrowed.
- Hosting instance, recurring budget and measured service limits.
- Software license selection and publication/redistribution review. No license is chosen on the owner's behalf.

The initial implementation does not include a font fallback, pretrained generator, external AI service, web deployment or database.
