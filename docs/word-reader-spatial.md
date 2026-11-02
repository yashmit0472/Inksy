# Short-output investigation and spatial reader candidate

1 November 2026. All diagnostics and preflight checks were performed without optimizer updates. The owner must launch any training below. No improvement in generated handwriting is claimed.

## Measured failure

Used the latest broad reader, SHA-256 `f139555630833362cc86e3de16e4fd4a5d9b07c2bc4a38a3694e7c06037480bb`, on the same fixed 256 examples in each development partition. Checkpoint bytes and in-memory weights were unchanged. Reserved test images were not opened.

On unseen content, the 816 character errors decompose into **443 deletions, 370 substitutions and three insertions** under a minimum-edit alignment (ties prefer substitution, deletion, insertion). No seven-or-more-character word was read exactly: 0/49. Exact accuracy for four-to-six-character words was 1/82. Predicted length averaged 2.72 characters versus 4.44 in the references.

| Unseen-content decoding diagnostic | CER | Exact words | Deletions |
|---|---:|---:|---:|
| Current greedy decoding | 71.83% | 7/256 | 443 |
| CTC prefix beam, width 8 | 70.86% | 7/256 | 431 |
| Greedy with blank log score reduced by 1 | 70.86% | 8/256 | 405 |

The beam uses every character, no lexicon, no language model, and no reference-length constraint; prefix count is pruned to eight. Tests compare the implementation with exhaustive CTC path enumeration on a small alphabet, including repeat/blank cases. The blank-score change is an ablation, not a new default. Neither method solves reading accuracy, so neither is presented as a quality fix.

The minimum remaining CTC alignment capacity is eight frames on unseen content and three on the selection subset, after allowing for repeated target letters. These words are not forced to lose letters by an insufficient time axis. Blank is the highest-scoring token on about 92% of frames, but frequent blanks are permitted by CTC; this alone is not a bug diagnosis. Both incorrect letter evidence and sequence scoring remain possible contributors.

Evidence: `artifacts/word-reader-short-output-analysis/errors.json`. Reproduce into a new directory:

```powershell
.\.venv\Scripts\python.exe -m ml.reader_errors --checkpoint artifacts/word-reader-broad-2500/best.pt --output artifacts/word-reader-short-output-repeat
```

## Code limitations addressed

**Vertical information.** The existing reader has three shallow convolutions and uniformly averages the remaining 16 vertical feature positions before its GRU. The mean cannot retain which vertical position contributed a given activation. This is a plausible representation bottleneck, not a proven sole cause of the errors.

The optional `spatial` architecture adds a learned full-height `(16, 1)` convolution alongside the mean, mixing channels and vertical positions. Its weights start at zero, so the complete reader initially reproduces the owner's latest trained model exactly. All existing convolution, GRU and classifier weights are copied. The adapter adds 147,456 parameters (541,258 -> 688,714 total). Its output is scaled by feature height. This is a project-specific, untrained candidate, not a reproduction of a published model or a demonstrated accuracy improvement.

Old checkpoints without an architecture field still load as `mean`. The only cross-architecture migration allowed is same-dimension `mean` -> `spatial`; incompatible dimensions and reverse migrations fail. Spatial checkpoints retain their architecture explicitly. No downloaded model weights are involved.

**Coverage across capped training blocks.** Previously, each 500-batch block began a new shuffled permutation and discarded the unused remainder. An index-only replay of the exact seed, loader, runtime and five-block budget visits **24,864 unique images**, leaving **8,484 of 33,348 unvisited** during that 40,000-draw run. This does not prove those images were never seen by earlier runs.

The new optional `coverage` sampler retains its position across capped blocks and reshuffles after completing the data permutation. The same 40,000-draw budget now visits all 33,348 training images once, then 6,652 images from the next pass. No validation examples enter training. Sampler counts are saved; optimizer/sampler state resume is still unsupported. Existing presets retain their original sampling behavior.

**Initial checkpoint selection.** The trainer now evaluates and saves the initial state before fitting, includes it in best-checkpoint selection, and writes `initial-selection.json` plus `selection-epoch-000.csv`. If subsequent blocks fail to improve selection CER (then CTC loss for ties), `best_epoch` remains zero. This protects the selected starting performance on that development subset, not on all possible words.

## Next owner-run command

```powershell
.\.venv\Scripts\python.exe -m ml.train_word_reader --config configs/word-reader-spatial-2500.json
```

This initializes from `artifacts/word-reader-broad-2500/best.pt` with a fresh optimizer and the zero-initialized adapter. It writes a new `artifacts/word-reader-spatial-2500` directory, preserving existing checkpoints. The manual preset is five capped blocks of 500 updates, batch size 16, learning rate 0.001, seed 314, CPU with four threads. It uses the same fixed 256 selection examples and 256 unseen-content probe as the baseline. Checkpoints and predictions are saved after each block; progress bars show elapsed time and ETA. The previous model's run took 17.59 minutes; do not assume the changed model has identical runtime.

No additional smoke or memorization run is requested before this bounded candidate run: real-data preflight already preserves the prior predictions and verifies a nonzero finite gradient to the new adapter. The manual optimization path itself still requires owner execution. The two changes are combined in this candidate; an improvement cannot be attributed separately to architecture versus coverage without a controlled comparison.

After completion, compare exact unseen-word accuracy, CER, deletions and results by word length against the saved baseline. Inspect longer words, not just common short words. A modest gain is not readiness: the provisional reader targets remain at least 80% exact words and at most 10% CER on the fixed unseen-content probe, followed by full-development review. These targets are not promises of this run's outcome. Review before extending training again.

## Verification and remaining product work

**160 default tests passed; four weight-update tests skipped; lint passed.** Tests cover legacy/spatial checkpoint loading, rejected migrations, unchanged predictions at migration, padding invariance, distinct gradients for vertical positions, finite adapter gradients without weight updates, coverage across truncated loaders, beam decoding, and edit accounting.

Real-data preflight compared **all 512 development images** under both readers: maximum log-probability difference **0.0**. A gradient-only check on the 16 verified training images produced finite nonzero adapter gradients while adapter weights remained zero and original weights/checkpoint bytes were unchanged. The full dataset index replay verified complete coverage. Evidence is in `spatial-preflight.json` and `sampling-replay.json` beside the diagnostic report.

This work improves the implementation and addresses measured reader failures. Generated paragraph handwriting remains the retained VAE output. A useful word generator, shared writer style and its visual comparison still need to be implemented and trained after auxiliary-reader competence is established; they are not delivered by this reader update.

## Research context

[Graves et al., Connectionist Temporal Classification (2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) defines CTC's path-to-label mapping and distinguishes best-path decoding from label-sequence search. It motivates testing decoding before attributing all errors to image features.

[Shi, Bai and Yao, CRNN (2015)](https://arxiv.org/abs/1507.05717) combines visual feature extraction, sequence modeling and transcription for variable-length recognition. It supports the broad reader design direction; it does not validate this custom residual adapter, this dataset preparation, or any Inksy quality claim. No external code or weights were copied.
