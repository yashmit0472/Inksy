# Broad reader run review

Reviewed 1 November 2026. The owner completed all five capped epochs: 2,500 additional optimizer updates and 40,000 word draws, initialized from the earlier 200-update reader with a fresh optimizer. Recorded training, selection and final probe time totals 1,055.23 seconds (17.59 minutes). Best checkpoint is epoch 5.

Checkpoint SHA-256: `f139555630833362cc86e3de16e4fd4a5d9b07c2bc4a38a3694e7c06037480bb`. Source hashes and data identity passed. Inference-only replay reproduced all 256 selection predictions and all 256 unseen-content predictions, including the saved losses and accuracy metrics. Checkpoint bytes and in-memory weights remained unchanged. No training was performed during review and the official test set was not evaluated.

| Fixed development metric | Earlier 200-update reader | Latest broad reader |
|---|---:|---:|
| Known-writer selection exact words | 3/256 (1.17%) | 53/256 (20.70%) |
| Known-writer selection CER | 91.95% | 69.54% |
| Unseen-content exact words | 0/256 (0%) | 7/256 (2.73%) |
| Unseen-content CER | 92.61% | 71.83% |

The same fixed sample identities are used for each comparison. The memorization checkpoint is not the baseline for this run. Selection CER over the five capped epochs was 81.90%, 75.20%, 71.13%, 70.65%, 69.54%: improvement continued, but slowed. These epochs each process only 8,000 shuffled examples, not a complete pass through 33,348 training images.

Seven correct unseen-content examples cover only three distinct strings: `than`, `his`, and `had`. Errors include `stop` -> `shs`, `may` -> `wiy`, `but` -> `fat`, `discuss` -> `das`, and `parties` -> `pad`. Mean predicted length is 2.72 characters versus 4.44 in the unseen-content references. This documents short outputs and spelling failures; it does not establish whether decoding, representation or optimization is the main cause.

The reader is learning, but remains far below the provisional reader-readiness targets. It is not ready to guide generator training. Generated paragraph handwriting is unchanged: this run trained the auxiliary reader, not a new generator. Before another long run, inspect errors by word length and character, and investigate the short-output behavior. This review does not authorize or launch automatic follow-on training.

Evidence: `artifacts/word-reader-broad-2500-review/review.json`; full predictions: `artifacts/word-reader-broad-2500/content-predictions.csv` and `selection-epoch-005.csv`.
