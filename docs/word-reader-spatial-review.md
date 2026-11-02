# Spatial reader run results

Reviewed 1 November 2026. The owner completed five capped blocks, 2,500 optimizer updates and 40,000 word draws. Coverage metadata records one complete 33,348-image pass plus 6,652 examples into the next pass. Recorded training and evaluation time, including initial selection, was 1,299.76 seconds (21.66 minutes). Best checkpoint is block 5.

Checkpoint SHA-256: `79eac860a1ae414ecb1aa138864108c27a69e7f3c13c6240202881fc65046db5`.

Source hashes and data identities passed. Fixed-weight replay reproduced all 256 selection and all 256 content-probe predictions, and the saved loss/accuracy metrics. Original checkpoint bytes and in-memory weights remained unchanged. No training was launched during review; reserved test images remain unopened.

| Same fixed development samples | Previous broad reader | Spatial reader |
|---|---:|---:|
| Unseen-content exact words | 7/256 (2.73%) | 20/256 (7.81%) |
| Unseen-content CER | 71.83% | 45.16% |
| Unseen-content deletions | 443 | 96 |
| Unseen-content substitutions | 370 | 379 |
| Unseen-content insertions | 3 | 38 |
| Known-writer selection exact words | 53/256 (20.70%) | 86/256 (33.59%) |
| Known-writer selection CER | 69.54% | 36.28% |

Mean unseen-content output length rose from 2.72 to 4.21 characters against 4.44 reference characters. The short-output failure is substantially reduced. Substitutions now dominate the remaining errors: 379 of 513. This is not a fully corrected reader.

On unseen-content examples, `countries` improved from `coe` to `countries`, `thread` from `thd` to `thread`, and `ribald` from `wad` to `ribald`. Remaining errors include `stop` -> `stoe`, `may` -> `way`, and `parties` -> `paslies`. Eighteen examples became exact and five previously exact examples regressed, yielding a net gain of thirteen. For reference strings of at least seven characters, exact accuracy rose from 0/49 to 1/49 and CER fell from 81.13% to 40.33%.

Selection CER continued decreasing at each block: 62.52%, 56.14%, 46.41%, 41.71%, 36.28%. Architecture, sampling coverage and additional optimization all changed relative to the prior checkpoint, so this run cannot isolate their individual contributions.

The reader is still below the provisional readiness targets. The paragraph handwriting generator has not changed, and this does not constitute a generated-handwriting quality evaluation. Evidence: `artifacts/word-reader-spatial-2500-review/review.json`; full saved predictions: `artifacts/word-reader-spatial-2500/content-predictions.csv`.
