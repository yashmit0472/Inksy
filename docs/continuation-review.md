# Refined continuation review - 27 October 2026

The owner completed another five epochs in `artifacts/refined-continue-5`, initialized from `artifacts/refined-5/best.pt`. All refined parameters were restored; the optimizer started fresh. Recorded training plus validation time was 99.57 minutes. Source and output checkpoint hashes matched their recorded provenance. No model training was launched during this review.

## Paired results

| Diagnostic | Previous refined checkpoint | After five more epochs |
| --- | ---: | ---: |
| Common beta-1 BCE + KL validation loss | 130.4517 | 130.0756 |
| Validation edge error | 28.3568 | 28.1742 |
| Exact-label recognizer agreement | 59.48% | 60.92% |
| Agreement ignoring case | 77.26% | 78.73% |
| Mean soft-ink fraction | 75.98% | 75.45% |
| Mean edge energy | 0.02183 | 0.02223 |

The comparison uses all 69,790 validation examples with fixed noise, 6,200 paired generated examples for recognizer scoring, and 1,240 paired generated examples for descriptive softness metrics. The previous checkpoint's cached reports were reused after verifying model hashes, recognizer identity, common objective, seeds and sample counts. Full results are in `artifacts/comparison-refined-continue/comparison.json`; the cached baseline report location is explicitly recorded there. The official final test set remains untouched.

## Interpretation

Additional training did help the character-agreement proxy again: exact agreement rose by 1.44 percentage points and ignoring-case agreement by 1.47 points. Gains for the fixed generated sample include `8` (72% to 80%) and `H` (68% to 76%). Some rates fell, including `T` (72% to 70%) and `w` (75% to 73%). Per-class rates use 100 samples each.

The comparable validation loss improved by approximately 0.29%, and the visual grids still look close to the previous refined run. Soft-ink fraction decreased by only 0.52 percentage points; lower gradient error and higher edge energy likewise show small changes. These metrics are affected by stroke geometry and do not constitute a perceptual-quality score.

The best saved checkpoint under the weighted objective is epoch 5. Within this continuation, common BCE + KL validation loss was slightly lower at epoch 4 (130.0554) than epoch 5 (130.0756); weighted loss still improved because its edge term decreased. Thus not every metric improved monotonically.

The frozen recognizer itself achieves 79.77% exact-label accuracy on real audited handwriting. Generated agreement is therefore an imperfect proxy, not ground-truth human accuracy. Case/shape ambiguity, diversity and page-level readability remain unvalidated, and the release quality gate has not been met.

Assessment: learning is still continuing slowly, with modest character-agreement gains and little visible clarity change. This does not prove that more training cannot help, and it does not establish that more epochs alone will solve softness. No further training was queued or started during review.
