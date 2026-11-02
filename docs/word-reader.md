# Word recognizer: manual execution guide

Implemented 29 October 2026; owner smoke run reviewed 31 October 2026. The agent has not trained the reader. It reads an image into text and will later supply the word generator's content loss. It is not a handwriting generator or its independent final judge. The completed smoke run currently decodes only blanks; see [the review and smaller next check](word-reader-smoke-review.md) before launching the full pilot below. The historical implementation-verification notes at the end predate that owner-run smoke test.

## Model and input

The reader is initialized randomly. Three convolutional stages extract features from real 64-pixel-high IAM word images. Width is downsampled by four, height is averaged, and a two-layer bidirectional GRU predicts a character distribution at each column position. A CTC loss aligns these distributions with the word transcription without needing individual character bounding boxes.

CTC uses token zero as blank. Repeated character predictions collapse unless separated by a blank, so `book` can retain its two `o` letters. The decoding time axis stops at each word's true width. Convolution masks and packed recurrent sequences prevent right-hand batch padding from changing the prediction for a shorter image.

The model uses the IAM token and preprocessing contracts. No pretrained weights, font substitution, external OCR service or generated training samples are used. CTC is computed on CPU even if model execution is later placed on CUDA, preserving the project's strict determinism setting and gradient flow. GPU execution is not verified in the current CPU-only environment.

## First command: execution smoke test

Run from `C:\Users\yashm\Desktop\CODE\Inksy`:

```powershell
.\.venv\Scripts\python.exe -m ml.train_word_reader --config configs/word-reader-pilot.json --output artifacts/word-reader-smoke --epochs 1 --max-train-batches 8 --max-validation-batches 2
```

This command performs training and must be launched by the owner. It processes 128 training-word draws, then up to 32 selection words and 32 unseen-word probe samples. It checks real-data forward/backward execution, optimizer steps, finite losses, progress reporting, checkpoint save/reload and prediction export. It is not enough training to establish useful reading quality. Blank or incorrect predictions can be expected this early.

Bars show batch progress, elapsed time/ETA, average CTC loss, character error rate (CER) and word count. At the end each epoch also prints exact-word accuracy and recorded training/selection time. Output directories must be new. `--no-progress` hides the bars for redirected logs.

Inspect execution and timing before committing to the pilot. The current environment is explicitly CPU-based. No GPU setup or long-running training is triggered automatically.

## Bounded pilot, after reviewing the smoke test

```powershell
.\.venv\Scripts\python.exe -m ml.train_word_reader --config configs/word-reader-pilot.json
```

The preset caps training at five epochs of 500 batches with batch size 16: **2,500 optimizer updates and 40,000 training-word draws**. These are capped epochs, not complete traversals of all 33,348 training words. A new shuffle is used for each epoch; draws may repeat across epochs. The pilot starts fresh unless the owner explicitly passes `--init-checkpoint` with a compatible checkpoint from this project.

Selection uses 512 evenly spaced samples from `seen_validation`, evaluated after each epoch. The checkpoint with the lowest CER is retained as `best.pt`, breaking ties with CTC loss. An unseen-word probe uses 512 evenly spaced `content_validation` samples once, after selection. Those counts are partial development diagnostics, not a full audit, and the selected IDs are hashed in the run metadata. Official unknown-writer validation and final-test pixels remain unopened.

Further runs can use `--init-checkpoint artifacts/word-reader-pilot/best.pt` with a new output directory. Initialization checks the word contract, architecture dimensions and prepared-record digest. It copies the project's learned weights but starts a fresh optimizer; interrupted-run resume is not supported. Longer runs and wider validation should be justified by the pilot evidence rather than repeated blindly.

## Output files

- `history.json`: training/selection loss, CER, exact-word accuracy, blank-prediction fraction, actual update counts, word draws and elapsed time per epoch.
- `selection-epoch-001.csv` etc.: source word IDs, exact reference text and decoded predictions for the fixed selection samples.
- `content-predictions.csv`: reference/prediction pairs for the post-selection unseen-word probe.
- `best.pt`, `last.pt`: versioned word-reader checkpoints. They cannot be loaded as character generators.
- `summary.json`: selected epoch, probe scores, checkpoint hash, partial-coverage metadata and an explicit `ready_for_generator_training=false` pending review.
- `config.json`, `environment.json`, `data-manifest.json`, `evaluation-partitions.json`, `source-sha256.json`: configuration, data/source provenance and fixed sample identities.

The smoke output is marked `pipeline-smoke-only`; a longer run remains `unreviewed-content-reader-pilot`. The code does not claim that a loss decrease or successful save makes the reader competent.

## Metrics and interpretation

**CER** is the total character edit distance divided by the total number of reference characters. Lower is better. Insertions can make CER exceed 100%; it is not clipped. Case changes count as errors. **Exact-word accuracy** is the fraction of words reproduced exactly; a separately reported case-insensitive score does not replace it.

CTC loss may fall while outputs remain mostly blank. Look at CER, exact words, empty-prediction fraction and the exported actual strings together. For a future training-loss reader, reading mistakes are especially consequential: a weak reader could reward incorrect generated words. Its real-data competence must be assessed before it is used in generator optimization.

The IAM data has rare uppercase/digit examples and no examples for six punctuation tokens under the initial filter. Review those limitations before claiming general mixed-case, numeric or punctuation support. This network's role is the training content reader; an independent word-level evaluator and human review are still required later.

## Inference-only evaluation

After an owner-run training:

```powershell
.\.venv\Scripts\python.exe -m ml.word_reader --checkpoint artifacts/word-reader-pilot/best.pt --partition content_validation --output artifacts/word-reader-content-review
```

With no `--max-batches`, this evaluates the complete selected development partition. It writes `evaluation.json` and `predictions.csv`, verifies prepared-record compatibility and does not perform optimizer updates. The command only permits the two development validation partitions; official final-test evaluation is deliberately unavailable here.

## Verification boundaries

Non-training tests verify CTC repeated-letter decoding, edit-distance arithmetic, impossible-alignment rejection, source-contract compatibility, frozen-reader gradients to image pixels, invariance to added batch padding and deterministic evaluation without parameter updates. A synthetic optimizer test is explicitly marked for the owner's `--run-training` opt-in and is not run by the agent.

An untrained real-data preflight verifies finite CTC loss and gradients from the reader back to eight actual IAM word images, while all reader parameters remain frozen and unchanged. This verifies the future content-loss path, not learned reading quality. Actual optimization execution and reading ability remain pending the owner's manual run.

Completed checks: **136 passed, four weight-update tests skipped; lint passed**. The default reader has 541,258 parameters. Real-data preflight used an 8x1x64x240 image batch and returned a 60x8x74 CTC tensor. Its report is `.cache/word-reader-preflight.json`. No actual training command was launched by the agent.
