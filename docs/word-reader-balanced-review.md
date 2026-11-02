# Frequency-balanced run: substantial-gain target not met

Reviewed 1 November 2026. The run completed its final evaluation after early stopping at three blocks / 1,500 updates / 24,000 word draws. Recorded training and evaluation time was 1,084.38 seconds (18.07 minutes). Best checkpoint is block 3.

Verified checksum: `41b14f56eb951f02b08ad19d2ee8755c8ba41d0df1944a1a661ad762b30632c0`. Source hashes matched; the checkpoint loaded with finite parameters. Metrics were recomputed from every saved selection and content prediction and matched the summary. This review did not replay model inference and did not perform training.

| Full-set metric | Previous selected model | Balanced run |
|---|---:|---:|
| Unseen-content exact words | 1,434/4,250 (33.74%) | 1,443/4,250 (33.95%) |
| Unseen-content CER | 25.81% | 25.21% |
| Selection exact words | 2,035/3,612 (56.34%) | 2,044/3,612 (56.59%) |
| Selection CER | 18.73% | 18.24% |

The frozen substantial-gain criteria failed: unseen accuracy was below 43.74%, and unseen CER remained above 20%. Selection nonregression passed. The net unseen exact-word gain is only nine examples, or 0.21 percentage points. Do not describe this as a significant handwriting improvement or silently lower the targets.

The stopping message reflects the configured threshold, not a crash. Selection CER improved from 18.7319% to 18.2377%: **0.4942 percentage points**, just short of the required 0.50. Across 16,592 reference characters this reduced errors by 82; an 83-error reduction would have crossed the threshold. None of the three checks reset patience. The best checkpoint was still saved, and final evaluation completed successfully. This threshold-triggered stop is not proof of complete model convergence.

Only 24,000 of the 33,348 training images were visited in this run before stopping. Therefore this bounded result does not establish that frequency weighting could never help under another protocol. It does establish that the tested candidate failed the predeclared gain criteria. No unchanged repeat or threshold relaxation was prepared automatically. Review the representation/data strategy before another compute commitment.

The reader remains below readiness targets; generated handwriting is unchanged. Evidence: `artifacts/word-reader-balanced-review/review.json`; frozen criteria: `artifacts/reader-gap-audit/next-run-targets.json`.
