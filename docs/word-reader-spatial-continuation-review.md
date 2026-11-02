# Spatial continuation results

Reviewed 1 November 2026. The owner completed all 5,000 updates and 80,000 word draws. The run ended with `budget_completed`, not plateau stopping. Best checkpoint is block 10. Coverage records two full data passes and 13,304 examples into the next pass. Recorded training and evaluation time was 2,264.36 seconds (37.74 minutes), including initial selection and the final content probe.

Verified checkpoint SHA-256: `6341013e7f14951d64898377af40f4005919feb25b63dcadedb7ebdad8b49c0c`. Source hashes and data identity passed. Inference-only replay reproduced all 512 saved development predictions and their losses/accuracy metrics. Checkpoint bytes and model weights remained unchanged. No additional training was launched, and official test images were not evaluated.

| Same fixed development samples | Previous spatial reader | Continued spatial reader |
|---|---:|---:|
| Unseen-content exact words | 20/256 (7.81%) | 73/256 (28.52%) |
| Unseen-content CER | 45.16% | 29.40% |
| Unseen-content substitutions | 379 | 245 |
| Unseen-content deletions | 96 | 73 |
| Unseen-content insertions | 38 | 16 |
| Known-writer selection exact words | 86/256 (33.59%) | 135/256 (52.73%) |
| Known-writer selection CER | 36.28% | 21.37% |

On unseen content, 57 examples became exact and four previously exact examples regressed, for a net gain of 53. New successes include examples of `Foreign`, `formal`, `Congo`, `but` and `no`. Errors remain: `stop` -> `stoo`, `may` -> `nay`, `discuss` -> `disos`, and `parties` -> `pashes`. Substitutions remain the largest error category (245 of 334 edits).

Unseen-content exact accuracy by reference length is 46/125 for one-to-three characters, 23/82 for four-to-six, and 4/49 for seven or more. Long-word exact accuracy rose from 1/49 to 4/49; long-word CER fell from 40.33% to 27.59%. This is a broader improvement than merely learning common short words, while long-word reading remains unreliable.

Selection CER briefly worsened at blocks 7 and 8, then improved at blocks 9 and 10. The last block was best, and the plateau counter ended at zero. The learning curve supports continued learning, but does not guarantee convergence to the provisional quality targets. These are partial development results, not official IAM benchmark or final-test results.

The reader remains below the provisional 80% exact-word / 10% CER readiness targets. Generated paragraph handwriting is unchanged: this run trains an auxiliary reader, not a word generator. Readiness remains false; no subsequent training is started automatically.

Evidence: `artifacts/word-reader-spatial-continue-review/review.json`; full predictions: `artifacts/word-reader-spatial-continue/content-predictions.csv`.
