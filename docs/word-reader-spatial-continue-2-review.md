# Second spatial continuation results

Reviewed 1 November 2026. The owner completed nine 500-update blocks: 4,500 optimizer updates and 72,000 word draws. Training stopped normally with `selection_plateau`. Best checkpoint is block 8, selected by lowest selection CER and then CTC loss, not necessarily highest exact-word accuracy. Block 9 had higher exact accuracy but worse CER, so it was not selected.

Verified best-checkpoint SHA-256: `077536c9de0804fcc64fd54451907a421a08763ce1a5fa0300201bcf686dc829`. The review checks source hashes, data identity, all saved full-development predictions and unchanged weights using inference only. No new training was launched; official test images remain unopened.

| Same full development sets | Previous checkpoint | Selected block 8 |
|---|---:|---:|
| Unseen-content exact words | 1,215/4,250 (28.59%) | 1,434/4,250 (33.74%) |
| Unseen-content CER | 28.53% | 25.81% |
| Selection exact words | 1,900/3,612 (52.60%) | 2,035/3,612 (56.34%) |
| Selection CER | 21.59% | 18.73% |

Recorded training/evaluation time was approximately 40.2 minutes. Coverage records two complete training-data passes and 5,304 examples into the next pass.

Early stopping means the configured improvement threshold was not met; it does not prove full convergence. The last significant selection-CER improvement was block 6 (19.033%). Block 8 reached 18.732%, but that cumulative gain was less than the configured 0.5-percentage-point threshold. Blocks 7, 8 and 9 therefore exhausted patience. The best checkpoint was preserved even though patience was not reset.

The full-set gains are real but smaller than in the preceding stage. Unseen-word accuracy remains below the provisional readiness target, and most unseen words still contain errors. Repeating the identical training setup indefinitely is not justified by this review. The next investigation should examine remaining character confusions and the difference between training and held-out performance before proposing another compute budget.

The handwriting generator remains unchanged. These are auxiliary-reader development results, not generated-handwriting quality or final-test results. Evidence is saved in `artifacts/word-reader-spatial-continue-2-review/review.json`; raw predictions remain in `artifacts/word-reader-spatial-continue-2/content-predictions.csv`.
