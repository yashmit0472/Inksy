# Word-reader 200-update review - 31 October 2026

The owner's latest run completed all 200 updates and 3,200 training-word draws from scratch. Recorded training plus selection/content evaluation time was 186.22 seconds (3.10 minutes), excluding setup and saving. This was much faster than the smoke-based estimate; that earlier eight-batch timing was not a reliable steady-state forecast.

The best-checkpoint hash matches the summary (`dd1a2833a0ea53e5900dc833478d7c03ee4b7c4194495e1e15c9e0dff07526f8`), and all recorded source hashes match. An inference-only replay reproduced every one of the 256 content-probe prediction rows and matched loss/CER/accuracy. Evidence is in `artifacts/word-reader-check-200-review/`. No training was launched during review.

| Diagnostic | Latest content probe |
| --- | ---: |
| Words evaluated | 256 |
| CTC loss | 3.4452 |
| Character error rate | 92.61% |
| Exact-word accuracy | 0% |
| Empty predictions | 8/256 (3.125%) |

The other predictions are single `t` (151) or `a` (97). For example, `stop` is predicted as `a` and `may` as `t`. Nonempty output does not establish reliable reading or meaningful sensitivity to the image. The known-writer selection probe scored 91.95% CER and 1.17% exact words (3/256).

The run has executed correctly, but the reader remains unsuitable as a generator content-loss teacher. These are partial development probes, not independent generator evaluation or official test results. No word generator has been trained. The smoke and 200-update probes use different sample counts/IDs, so their aggregate scores are not a matched improvement measurement.
