# Completed memorization diagnostic

Reviewed 1 November 2026. The owner suspected an accidental interruption. The run completed normally: `summary.json` reports success, and the saved history shows three consecutive passes at updates 200, 225 and 250. The configured early exit stopped it at 250 of the maximum 600 updates. Recorded runtime was 276.26 seconds (4.60 minutes).

Inference-only replay verified the source hashes, checkpoint SHA-256 (`be717c38c0c70b243079d4978dd4808c3ff89534cf4df301ec507647ed75d81c`), data identity and unchanged model weights. All 16 training words were read exactly, with 0% CER and CTC loss 0.019665. Blank images at identical widths produced 0/16 exact words; reversed handwriting produced 1/16. The tiny task is learnable and gross width-only memorization does not explain this pass. No additional training was launched.

The same fixed 256 unseen-content examples still produced 0/256 exact words, with 111.00% CER and CTC loss 9.4245. CER can exceed 100% because inserted characters count as errors. This checkpoint is specialized to 16 examples, and is not a generally useful reader or an improved handwriting generator. Its unseen-content results are worse than the original 200-update reader (92.61% CER, also 0/256 exact). Do not rerun or extend the 16-word diagnostic.

Evidence: `artifacts/word-reader-memorize-16-review/review.json` and `predictions.json`. Checkpoint and source verification were sufficient to establish completed execution. Windows process-command-line inspection was unavailable; no claim about other currently running Python processes is made.

## Next owner action

Prepared `configs/word-reader-broad-2500.json`. It initializes from the original broad-data 200-update checkpoint, preserving the memorization model as a diagnostic artifact. This avoids treating the tiny overfit model as a demonstrated general improvement. The optimizer is fresh; this is a weight warm start, not an exact optimizer-state resume.

```powershell
.\.venv\Scripts\python.exe -m ml.train_word_reader --config configs/word-reader-broad-2500.json
```

Budget: five capped epochs of 500 updates, 16 examples per batch, for 2,500 additional updates and 40,000 word draws from the 33,348-image training partition. Repeated partial shuffles do not guarantee that every training image is visited. Evaluate the same 256 known-writer selection examples and save best/last checkpoints after each 500-update block. At completion, probe the same 256 unseen-content words used for the baseline. The output directory is new and must not already exist.

Use progress-bar ETA instead of the earlier unreliable smoke-based runtime estimate. Review the saved metrics and predictions after this bounded run; passing a handwriting-quality gate is not guaranteed. Reader training still does not alter the paragraph generator. Do not use the memorization or content-loss reader as an independent generator evaluator.
