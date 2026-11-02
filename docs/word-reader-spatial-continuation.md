# Bounded continuation of the spatial reader

1 November 2026. Prepared for the owner to run manually; no training was launched.

The latest reader remains inadequate: 20/256 exact unseen-content words and 45.16% CER. However, its fixed selection CER improved at every block (62.52%, 56.14%, 46.41%, 41.71%, 36.28%), and the final block still improved by 5.42 percentage points. This supports testing continued optimization of the same architecture. The new spatial parameters have received 40,000 draws, about 1.20 training-partition passes; the inherited parameters also retain earlier training. This is evidence of ongoing learning, not proof that more training will reach the quality target.

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m ml.train_word_reader --config configs/word-reader-spatial-continue.json
```

- Start from `artifacts/word-reader-spatial-2500/best.pt`, preserving all learned spatial, convolution, GRU and classifier weights. This is a weight warm start with a fresh optimizer, not an exact optimizer-state resume.
- Keep the spatial architecture and coverage sampler. Learning rate is reduced from 0.001 to 0.0005 as a conservative continuation setting; that choice is not independently validated. Seed 315 begins a new recorded training order.
- Maximum ten capped blocks of 500 updates: **5,000 updates / 80,000 word draws**, about 2.40 additional data passes. The coverage sampler exhausts each permutation before repeating.
- Evaluate and checkpoint every 500 updates on the unchanged fixed 256-example selection subset. The initial model remains eligible for best-checkpoint selection.
- Stop after **three consecutive checks without a cumulative selection-CER improvement of at least 0.005 (0.5 percentage points)** from the last significant improvement. Small gains can accumulate to reset patience. Checkpoint ranking still uses every improvement in CER, then loss; it is separate from the patience threshold.
- On either plateau or budget completion, load the best checkpoint, evaluate the same 256 unseen-content words, and write `summary.json` including `stop_reason`. No content-probe or final-test score is used by early stopping. Readiness remains false pending review.
- Save to the new `artifacts/word-reader-spatial-continue` directory. Existing checkpoints are preserved. The directory must not already exist.

At the previous run's measured speed, the full budget extrapolates to about 43 minutes; actual time varies, and early stopping can shorten it. Use the progress bar's ETA. Do not automatically extend the run after it ends.

Compare the resulting unseen-content CER, exact words, substitutions and word-length buckets against 45.16%, 20/256 and 379 substitutions. If selection improves while unseen-content performance stagnates, investigate generalization rather than interpreting training loss as success. If both stop improving far below target, review the representation/data failure before another large run. The provisional reader targets are still 80% exact words and 10% CER followed by full-development review, not promised results from this stage. Reader progress does not itself improve the current paragraph generator.

Verification: **167 default tests passed; four weight-update tests skipped; lint passed.** Non-training tests cover plateau patience, cumulative improvements, ties, disabled stopping for existing presets, invalid settings and the bounded continuation configuration. Preflight verified the latest checkpoint hash, same-architecture compatibility, unchanged selection identity and new output path. The new training loop behavior is prepared but has not been executed by the agent.
