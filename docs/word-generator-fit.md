# Word generator: first bounded owner-run fit

> **Completed-run update, 2 November:** The owner ran this experiment. All 1,000 updates completed, but the visual fit gate failed. See [the result review](word-generator-fit-review.md). Do not rerun the original command against its existing output or start broader training based on its lower loss. Instructions below record the original run and remain reference material.

Implemented 2 November 2026. **Training is pending.** This is the first word-image generator in Inksy, separate from both the old isolated-glyph VAE and the IAM word reader. It is a 216,474-parameter, randomly initialized, single-writer model. It has no pretrained components, OCR loss, image lookup, external generation API or handwriting font fallback.

## Run this next

From `C:\Users\yashm\Desktop\CODE\Inksy` in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m ml.train_word_generator --config configs/word-generator-fit-32.json
```

This command **does train model weights**; the project owner runs it manually. No installation or data download is needed in this workspace. It uses the working CPU environment.

The first run fits the same 32 training words frozen during reassessment. It is capped at **1,000 updates or 1,200 seconds**, whichever occurs first, with eight examples per update. Each shuffled pass visits all 32 words. A tqdm bar shows completed updates, elapsed time, estimated remaining update time, current loss, gradient norm and remaining time budget. Dataset loading and periodic checks count toward the time budget. The limit is checked between updates; final evaluation/checkpoint export adds overhead. It is not a hard process-kill deadline.

Do not launch another reader continuation. Do not run this command again against the same output directory: it refuses to overwrite a run. Ctrl+C during the training loop retains the last completely saved check and exports the best saved model for review; up to one check interval of updates may be lost. Automatic optimizer/sampler resume is intentionally unsupported for this bounded diagnostic. A second interruption during export can leave review artifacts incomplete; checkpoints remain separate.

## What to inspect

Output directory: `artifacts/word-generator-fit-32/`.

| File | Meaning |
|---|---|
| `predicted-width-0000.png`, `predicted-width-0100.png`, etc. | All 32 training prompts at initialization and every 100 updates; generated with predicted widths. |
| `best.pt` | Lowest full training-set objective, including width loss. This is not automatically the most readable checkpoint. |
| `last.pt` | Most recently completed saved check. Separate format from the reader and glyph checkpoints. |
| `summary.json`, `history.json` | Stop reason, saved updates, losses, duration and checkpoint digest. |
| `scope.json`, `config.json`, `source-sha256.json`, `environment.json` | Exact training IDs, partition/dataset identity, settings, seed and code/runtime provenance. |
| `best-review/blind-predicted-width.png` | **Open this first** for a transcription review with requested words hidden. |
| `best-review/transcriptions.csv` | Enter each sample's transcription and `yes`/`no` under `no_visual_defects`. Use `<unreadable>` for an unreadable word. |
| `best-review/answer-key.csv` | Requested words and source identities; reveal only after transcription. |
| `best-review/real-targets.png` | Actual IAM training references, clearly labeled as real. |
| `best-review/generated-target-width.png` | Generated images supplied the real target widths; useful diagnostic, not end-to-end inference. |
| `best-review/generated-predicted-width.png` | Same trained model using text alone and its own width prediction. |
| `best-review/conditioning-controls.png` | Original/reversed/sorted text and repeated-letter prompts. These are controls, not a held-out accuracy score. |

For a less biased check, ask a reader who has not seen the training prompts or labeled progress sheets to transcribe the blind grid. The tool cannot enforce human blinding. All 32 samples must be assessed; do not choose only attractive outputs.

After filling the CSV:

```powershell
.\.venv\Scripts\python.exe -m ml.review_word_generator --review artifacts/word-generator-fit-32/best-review
```

This is inference-free scoring of human answers, not OCR or training. It verifies answer-key and image digests, rejects missing/duplicate answers, computes exact words and character error rate, and writes `human-review.json`. The fit gate is **at least 31/32 exact** and **all 32 visually acceptable**: no missing strokes, clipping or other visible defects. A low reconstruction loss cannot pass this gate. Even a pass establishes memorization only.

## Generate your own diagnostic words

Once the checkpoint exists:

```powershell
.\.venv\Scripts\python.exe -m ml.word_generator_preview --checkpoint artifacts/word-generator-fit-32/best.pt --text hello little letters --output artifacts/word-generator-custom-preview
```

This command is inference only. It produces individual PNGs, a labeled grid and provenance. The model receives **only the requested text**: no target image, known width or dataset lookup. Output is deterministic; no diversity is claimed. The supported contract is one learned writer and lowercase ASCII words of 2–10 letters. Unsupported case, spaces inside a word, punctuation and longer words raise an error. These custom words are not automatically an unbiased unseen-word evaluation.

## What the implementation learns

Character embeddings feed a bidirectional GRU. Contextual character features are interpolated across the word's horizontal grid and combined with global text features and position coordinates. A projection creates an eight-pixel-high spatial feature map, followed by three nearest-resize/convolution stages to produce a 64-pixel-high grayscale word image. Padding is masked after every convolution, and there is no batch-dependent normalization. An independent head predicts width from text. The model has no per-word identity table or target-image encoder.

For paired training, the image decoder uses target widths so the generated and real images align. Width supervision separately trains the inference width head. The objective is foreground-weighted binary cross entropy, soft Dice overlap, edge error and log-width regression. Per-word normalization prevents wider images from dominating only through pixel count. Foreground and background errors are reported separately. These losses describe image fit, not spelling accuracy, realism or style generalization. Regressing variable handwriting can produce blur; this experiment tests that risk rather than assuming it solved.

The prepared data scope remains 1,930 lowercase training images / 679 distinct words for writer `000`. This command uses **only its pinned 32-example fit subset**, selected by seed-42 SHA-256 ordering and first occurrence per word. It does not fit all 1,930 images. The subset's IDs exactly match `artifacts/generation-reassessment/scope-audit.json`. Original validation partitions and reserved test pixels are untouched.

## Verification completed without training

- Full default suite: **200 passed, 4 weight-update tests skipped**. Lint passed. Tests cover text rejection, padding invariance, character-order sensitivity, masked losses, blank-output penalties, finite gradients, data tampering/scope selection, checkpoint compatibility and blind-review arithmetic/integrity.
- Real-data preflight: all 32 source images verified; all 29 parameter tensors received finite nonzero gradients; every state-dict tensor remained identical. No optimizer was constructed, no checkpoint was trained, and no reserved images were opened.
- Four forward/backward timing probes on one eight-word batch gave a warm median of **0.141 seconds**. Multiplying by 1,000 gives approximately 141 seconds of that batch's compute only; optimizer work, varying widths, evaluation, exports and system load are excluded. This is not an end-to-end runtime promise.
- Untrained preview layout was visually inspected. Its gray output is random initialization, not a trained result or handwriting-quality evidence.

Evidence: `artifacts/word-generator-preflight-20261102/preflight.json`. A fresh read-only preflight can be run with `--preflight --preflight-output artifacts/word-generator-preflight-new`; use a new output directory. The optimizer loop itself has not been executed by the agent. Code tests and gradients do not establish successful training or readable words.

## What follows this run

Review the best predicted-width outputs before any broader training. If the 32-word gate fails, diagnose that failure instead of automatically adding updates. If it passes, prepare the separately bounded 1,930-image pilot with seen-validation checkpoint selection and the frozen 69-distinct-unseen-word comparison from [the reassessment](generation-reassessment.md). That broader trainer/evaluation workflow is the next stage; this fit command cannot establish unseen-word quality. The provisional generalization/release targets remain unchanged, and word-to-page integration remains deferred.
