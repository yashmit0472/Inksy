# Decoder comparison review — 5 November 2026

Both owner-run arms completed all 1,000 additional updates and their final exports. The position-aware candidate visibly reproduces several training words better than the plain control, especially `and`, `was` and `an`, where broad horizontal smears have become recognizable strokes. **This is improved training-set image fitting; the candidate's formal fit gate remains unconfirmed and unseen-word quality is unestablished.** No further training was launched during this review.

## Matching and verification

The configurations differ only in architecture and output directory. Both used the same parent checkpoint (`2b5e0b17e5ea7a79c2a6e8e19e225fd0e8943ad3785a413c3f3f12aa7852a558`), writer, 32 training IDs, ordering seed, batch size, fresh optimizer, objective, learning rate and update/time budgets. Both stopped at the update cap. The different wall-clock times are observations, not a controlled architecture speed benchmark.

| Measurement | Plain control | Position-aware candidate |
|---|---:|---:|
| Additional updates completed | 1,000 | 1,000 |
| Recorded training/evaluation/export time | 540.95 s (9m 01s) | 316.95 s (5m 17s) |
| Selected checkpoint's run-local update | 900 | 1,000 |
| Selected checkpoint training objective | 0.71982 | 0.70133 |
| Matched final-update training objective | 0.72285 | 0.70133 |
| Matched final-update edge error | 0.01872 | 0.01634 |

At the matched 1,000-update endpoint, the candidate's objective is 2.98% lower and its edge error is 12.70% lower. These are relative reconstruction-metric changes, not percentage-point gains in recognition accuracy. The selected visual grids use each arm's lowest-training-objective checkpoint under the same selection rule; the control's selected grid is from update 900. This distinction is retained in the exported comparison labels.

Current code hashes match both runs' source records. Dataset/scope identities and source-image hashes were verified. Both best and last checkpoints loaded with finite parameters, their initialization identities matched, and inference reproduced every saved selected/final metric within 1e-7. All 64 individually exported best-checkpoint word images were reproduced exactly. Checkpoint bytes and in-memory weights remained unchanged throughout review.

Selected checkpoint hashes:

- Control: `ed4f16fc41fbc8eb08dd787b919325abd03fa14e41cc8c29b0243e5a828cd962`.
- Candidate: `c9734bb7f464d3b95c394001c539cd12c544af6887d04b4e519fa9e0559ba2b3`.

## Visual and readability assessment

Inspected both complete 32-word blind grids, then the candidate's labeled grid, real targets and conditioning controls. The control retains obvious smears in multiple examples and still fails the visual component of the fit gate. The candidate improves target resemblance substantially in those cases. Some candidate words remain ambiguous or imperfect, including `taking` and `loyalty`; a readable reference and an accurate reproduction are also distinct from attractive handwriting.

Both owner transcription CSVs are entirely empty. No human exact-word accuracy or CER is assigned, and no 31/32 pass is claimed. The agent's inspection is not a blinded human study: these prompts have appeared earlier in the conversation. The candidate's all-images-visually-acceptable requirement also needs explicit reviewer acceptance. Retain it as the better-fitting candidate, without marking the fit gate complete.

The candidate's custom probes `letter`, `little` and `letters` remain visibly malformed. Those probes are outside this model's 32-word fit list; they are illustrative diagnostics rather than a formal held-out test. The model has not demonstrated reliable new-word generation. The prepared 69-word content holdout and reserved official test pixels were not evaluated here.

## Next action

A portable browser form is now prepared at `artifacts/word-generator-detail-coordinate/blind-review.html`; see [the human review guide](word-generator-human-review.md). It replaces manual CSV editing for the reviewer and exports answers for the same scoring gate. No real answers are filled in automatically.

Do not repeat either run automatically. Complete the candidate's human transcription/visual review before any broader pilot. Keep the original gate: at least 31/32 exact words using predicted widths and all 32 images visually acceptable. If it passes, the next experiment can address generalization on the wider training pool. If it fails, preserve the failure and reassess; lower loss alone is insufficient to expand training.

The existing review materials are in `artifacts/word-generator-detail-coordinate/best-review/`. Prefer a reader who has not seen the prompts. Fill `transcriptions.csv` after viewing `blind-predicted-width.png`, then run the existing inference-free scoring command:

```powershell
.\.venv\Scripts\python.exe -m ml.review_word_generator --review artifacts/word-generator-detail-coordinate/best-review
```

Review evidence: `artifacts/word-generator-detail-review-20261105/review.json`. Selected target/control/candidate examples: `artifacts/word-generator-detail-review-20261105/selected-comparison.png`. The comparison includes improvements and remaining difficult examples; it is not an accuracy estimate. No application code changed and no training tests were run for this inference/documentation review.
