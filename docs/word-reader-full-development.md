# Full development evaluation and next bounded stage

1 November 2026. Inference-only evaluation of the current checkpoint `artifacts/word-reader-spatial-continue/best.pt` (SHA-256 `6341013e7f14951d64898377af40f4005919feb25b63dcadedb7ebdad8b49c0c`). All model parameters and checkpoint bytes remained unchanged. No training was performed and reserved writer-validation/final-test images were not opened.

| Full development partition | Examples | Exact words | CER |
|---|---:|---:|---:|
| Known-writer selection | 3,612 | 1,900 (52.60%) | 21.59% |
| Unseen content | 4,250 | 1,215 (28.59%) | 28.53% |

The full results are close to the earlier 256-example probes (52.73%/28.52% exact words). The latest progress is not confined to the small fixed samples. These remain development measurements, not independent final-test scores. On unseen content, 4,141 substitutions, 1,322 deletions and 254 insertions account for 5,717 errors. Frequent substitutions include `b` -> `h` (200), `a` -> `o` (113), and `m` -> `n` (90). Exact accuracy for reference strings of seven or more characters is 127/1,028 (12.35%).

Full evaluation took 37.01 seconds for selection and 48.81 seconds for unseen content, excluding setup. Evidence and all predictions are saved in `artifacts/word-reader-full-development/`.

## Next owner-run stage

The previous run ended at its compute budget with its last checkpoint best, and the full-development scores support continuing the same model. Prepared:

```powershell
.\.venv\Scripts\python.exe -m ml.train_word_reader --config configs/word-reader-spatial-continue-2.json
```

This uses the latest trained spatial weights with a fresh optimizer; it is not exact optimizer/sampler-state resume. Architecture and learning rate (0.0005) stay unchanged. Seed 316 starts a new coverage-sampling order, which visits all training examples before repeating. The output `artifacts/word-reader-spatial-continue-2` must be new; all prior checkpoints remain preserved.

Maximum budget is 5,000 updates / 80,000 word draws. Every 500 updates, selection now uses **all 3,612 selection images**, not the earlier 256-image subset. The initial state is evaluated on this full set before fitting and remains eligible as the best checkpoint. Early stopping still uses three checks without a cumulative selection-CER gain of at least 0.5 percentage points. At completion, the best checkpoint is evaluated on **all 4,250 unseen-content images**. No content-probe labels are used for early stopping or checkpoint ranking.

Compare this stage against the full-set baseline above; do not treat changes in evaluation sample size as learning. Full validation adds overhead: approximately 45 minutes for the maximum budget at recently measured speeds, with variability and possible earlier stopping. Use the progress bar's ETA. Do not automatically extend the run.

The reader remains below its provisional targets and is not ready as a generator content-loss network. This continuation may improve it but cannot guarantee readiness. If the full-selection score stalls or unseen-content results fail to improve, review the errors before spending more compute. The word generator and writer-style network remain separate unfinished components; no generated-handwriting improvement is claimed here.

Preflight validated the configuration, latest checkpoint checksum, same-architecture loading, full development partition sizes/identities and fresh output path. No model/training code changed in this stage; no optimization was launched.
