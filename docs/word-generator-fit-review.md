# First word-generator fit review — 2 November 2026

> Follow-up: a fixed-weight detail audit and matched plain/position-aware decoder comparison are now prepared in [word-generator-detail-comparison.md](word-generator-detail-comparison.md). Both owner-run arms are pending. This failed first-run assessment is unchanged.

The owner completed 1,000 updates / 8,000 training-word draws in 222.70 seconds (3 minutes 43 seconds). The update cap ended the run normally. Update 1,000 was selected for lowest full 32-word training objective. **The visual component of the fit gate failed.** Do not start the broader 1,930-image pilot based on this result.

## Verified execution

Checkpoint SHA-256: `2b5e0b17e5ea7a79c2a6e8e19e225fd0e8943ad3785a413c3f3f12aa7852a558`.

Source hashes, dataset/scope identities, the 32 source-image hashes and checkpoint metadata matched. The checkpoint loaded with finite weights. Inference-only replay reproduced all 32 exported word images exactly and reproduced every saved selected-checkpoint loss within 1e-7. Weights and checkpoint bytes remained unchanged during review. No optimization occurred.

| Diagnostic | Initial | Selected checkpoint |
|---|---:|---:|
| Combined training objective | 1.35190 | 0.77807 |
| Foreground-weighted absolute error | 0.42042 | 0.10952 |
| Background-weighted absolute error | 0.13353 | 0.04642 |
| Edge error | 0.02512 | 0.02364 |

Losses demonstrate improved image fitting, not a spelling/readability score. The loss was still decreasing when the fixed cap was reached; this is not evidence of convergence or proof that more updates cannot help.

Continuous width prediction has mean absolute error 3.00 pixels. After the model's multiple-of-eight rounding, 27/32 widths are exact, mean absolute error is 1.50 pixels and maximum error is 16 pixels. The target-width and predicted-width sheets both contain substantial image defects, so width prediction alone cannot explain or repair this result.

## Visual result and limits

Inspected the blind predicted-width grid first, then the real targets, target-width diagnostic and conditioning controls. Word-level shapes have emerged, but multiple training words contain smeared, merged or missing letter strokes. For example, `and`, `was`, `taking` and `conference` have obvious detail loss compared with their real targets. Some short examples such as `to` and `of` are more recognizable. This is not reliable reproduction even of the tiny training set.

The owner transcription CSV remains blank. **No human exact-word accuracy or CER is assigned.** The all-images-visually-acceptable requirement already fails on inspection, independently of the pending 31/32 transcription criterion. The agent's visual judgment is not a completed blinded human study; previous exposure to training prompts also limits blinding. No unseen-word quality or improvement over the retained glyph generator is established.

The changed-text controls produce different images, but `letter` and `little` do not become reliably readable. Pixel sensitivity to the prompt does not establish correct compositional spelling. The prepared 69-word content holdout and reserved test images were not evaluated in this review.

## Next decision

Keep the checkpoint and failed diagnostic intact. Diagnose stroke/detail reproduction in the image decoder and objective before committing to broader data or an unchanged continuation. The coarse spatial representation, optimization budget and reconstruction objective are candidate explanations, not isolated causes. This review does not prescribe another training run or claim that a particular architecture change will fix the problem.

Evidence: `artifacts/word-generator-fit-32-review/review.json`. Selected real/generated examples: `artifacts/word-generator-fit-32-review/real-versus-generated.png`; the full 32 generated outputs remain in `artifacts/word-generator-fit-32/best-review/`. The comparison is labeled as selected examples, not an accuracy estimate. No application code was changed and the test suite was not rerun for this inference/documentation review.
