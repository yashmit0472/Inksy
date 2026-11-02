# Reader diagnosis and the next bounded experiment

Reviewed 1 November 2026. No training was performed during this review.

## Why more training time has not meant better handwriting

The retained VAE, failed class-conditional GAN pilot, and current CTC word reader are different models. Their training time does not accumulate into one improving generator. The latest reader processed 3,200 word draws against a training partition of 33,348 images (less than one complete pass). Its failure after 200 updates does not prove that the architecture cannot learn, but neither does reduced CTC loss establish reading competence.

The VAE remains the paragraph preview's generator. The word generator and style network have not been implemented. Reader training alone cannot improve that preview.

## Fixed-weight diagnostic

Checkpoint: `artifacts/word-reader-check-200/best.pt`, SHA-256 `dd1a2833a0ea53e5900dc833478d7c03ee4b7c4194495e1e15c9e0dff07526f8`.

Evaluated the same 256 content-validation words under three image conditions, preserving each sample's width and CTC sequence length. No final-test images were opened. Checkpoint bytes and in-memory parameters remained unchanged.

| Input | Exact words | Character error rate | CTC loss | Predicted strings |
|---|---:|---:|---:|---|
| Real handwriting | 0/256 | 92.61% | 3.4452 | 151 `t`, 97 `a`, 8 empty |
| Blank image, same width | 0/256 | 93.84% | 3.7670 | 256 `t` |
| Horizontally reversed real image | 0/256 | 93.31% | 3.4908 | 241 `t`, 14 `a`, 1 `o` |

Blanking changes 41.02% of decoded outputs; reversing changes 39.45%. Mean per-frame probability total variation from real input is 0.0888 for blank and 0.0235 for reversed input. The reader responds to pixels; it is not a completely disconnected image pipeline. This does not establish letter recognition, a specific implementation bug, or an architecture diagnosis. These ablations alone cannot distinguish insufficient optimization from an inadequate representation.

The real IAM reference grid was visually checked: upright handwriting, varied writers, matching example transcriptions. This is a sample inspection, not a complete label audit.

Evidence: `artifacts/word-reader-image-diagnostics/diagnostics.json`. Reproduce into a new directory:

```powershell
.\.venv\Scripts\python.exe -m ml.diagnose_word_reader --checkpoint artifacts/word-reader-check-200/best.pt --output artifacts/word-reader-image-diagnostics-repeat
```

## Next owner-run check: memorize 16 verified training words

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m ml.train_word_reader_sanity --config configs/word-reader-memorize-16.json
```

This is an optimization run for the owner to launch manually. It loads the owner's existing reader weights with a fresh optimizer; it does not restart the reader from random weights. The original checkpoint is preserved. The new output directory must not already exist.

- Exactly the 16 pinned, distinct, single-writer training examples from `artifacts/iam-words/sanity-set.json`, with source image and dataset hashes checked. Cached in memory to avoid repeated disk reads.
- A maximum of **600 updates**, 16 images per update: at most 9,600 repeated word draws. No automatic extension or larger follow-on job.
- Progress bar with completed updates, CTC loss and ETA. Every 25 updates, save exact-word predictions, CER and blank/reversed-image controls. Save best and last checkpoints.
- Early pass requires **16/16 exact words**, blank-image exact accuracy no greater than 25%, and lower CTC loss on real than blank images at **three consecutive checks**. The blank control limits obvious width-only shortcuts; it does not prove generalization.
- This warm-start reader currently gets **0/16**, with 90.625% CER on these specific examples. Baseline is in `artifacts/word-reader-image-diagnostics/sanity-before.json`.
- At completion inspect `artifacts/word-reader-memorize-16/summary.json`, `history.json` and `best-predictions.csv`. `passed` is a memorization result only; `ready_for_generator_training` stays false.

Do not interpret a perfect score on these 16 training images as a useful OCR model. If the check fails, inspect its learning curve and predictions before changing the representation/optimization. If it passes, return to representative full training data and a fixed development protocol. Do not use the memorization checkpoint as an independent quality evaluator.

## Route to a visible improvement

1. **Prove learning on the tiny fixed task.** The bounded run above is the immediate owner action. If it cannot fit these images, investigate and fix that failure before another large run.
2. **Establish reading on unseen examples.** Review real transcriptions, CER, exact-word accuracy, and blank controls on fixed known-writer development subsets. Use meaningful coverage of supported characters. A provisional engineering target is at least 80% exact words and at most 10% CER on the fixed unseen-content probe; these are proposed gates, not achieved results or guarantees. Check the full development partition before declaring readiness. Retain the official test split for later independent evaluation.
3. **Implement the word generator with a shared writer style and explicit content loss.** First fit a small single-writer task with paired target images; inspect visible generated word samples at fixed intervals. Expand to held-out text and more writers only after the small task succeeds. The architecture is currently a design, not completed code.
4. **Compare actual generated handwriting.** Use the same requested words and paragraph across checkpoints, examine spelling and consistent style, and conduct visual review plus an independent evaluation. Reader loss and the training-loss reader's score cannot certify generator quality. Keep the retained VAE default until a candidate demonstrates an improvement.

This sequence tests prerequisites and limits wasted compute; it does not promise that the next reader run will produce better handwriting. No packages, GPU configuration or Windows settings were changed.

Code verification: 145 default tests passed; four weight-update tests remained skipped. Tests cover padding-preserving ablations, unchanged reader parameters, sanity-set integrity and training-partition restrictions, and rejection of width-only or incomplete memorization. Optimization behavior of the new manual runner is still pending owner execution.
