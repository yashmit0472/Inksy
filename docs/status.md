# Implementation status - 2 November 2026

## Position-aware decoder comparison prepared; owner execution pending

Fixed-weight diagnostics found reduced horizontal image detail (61.00% of target edge magnitude versus 92.71% vertically); this is a diagnostic association, not a causal proof. Edge-loss gradients were present, so the loss is unchanged. Implemented optional per-word position inputs at all three decoder enlargement stages, with a zero-weight migration that preserves the existing model's starting predictions.

Prepared a plain-decoder control and coordinate-decoder candidate, both initialized from the same pinned owner checkpoint with the same 32 IDs, ordering seed, fresh optimizer, objective and 1,000-update/20-minute budget per arm. No training was launched. All 32 starting predictions matched exactly in target/predicted-width modes, new input gradients were finite/nonzero, and weights remained unchanged. Verification: 207 tests passed, four weight-update tests skipped, lint passed. Quality remains unproven; the 31/32 plus all-images-visually-acceptable fit gate is unchanged. See [the controlled comparison](word-generator-detail-comparison.md) for sequential manual commands and stopping/decision rules.

## First word-generator fit completed; visual gate failed

The owner completed 1,000 updates in 222.70 seconds; update 1,000 had the lowest training objective (1.35190 initially, 0.77807 finally). Source/data/checkpoint hashes passed, all selected metrics were reproduced, and inference-only replay matched all 32 saved word images exactly. No weights were updated during review.

Generated words now have handwriting-like shapes but still contain smeared, merged and missing strokes, including when supplied the correct target widths. Predicted rendered widths match 27/32 targets, so width prediction alone does not explain the image defects. The visual component of the fit gate failed; the owner transcription CSV is unfilled, and no exact-word accuracy is claimed. The broader pilot and unseen-word evaluation remain pending. Diagnose image-detail fitting before prescribing another run. See [the fit review](word-generator-fit-review.md).

## Word generator implemented; first manual fit is pending

Implemented a 216,474-parameter, randomly initialized single-writer text-to-word-image generator, a bounded 32-word manual trainer, predicted-width and target-width previews, checkpoint/data provenance, and blinded human-transcription scoring. The model uses paired image/edge/width supervision; it does not depend on the weak reader. The owner command is in [word-generator-fit.md](word-generator-fit.md). This is a training-fit experiment, not the broader 1,930-image pilot or a demonstrated quality improvement.

Verification: 200 default tests passed, four weight-update tests skipped; lint passed. Real-data preflight verified all 32 pinned examples, finite nonzero gradients for all 29 parameter tensors, and unchanged weights. No optimizer was constructed during preflight and no training was launched. Warm forward/backward timing was 0.141 seconds for one eight-word batch, excluding optimizer/evaluation/export overhead. Training execution, readable generated output and generalization remain unestablished. The existing page renderer/checkpoints are unchanged.

## Reassessment: prioritize actual word generation

Routine reader continuation is paused. The latest reader's 33.95% unseen exact accuracy does not measure generated handwriting, and the new word generator remains unimplemented. Requiring reader readiness before any generator prototype delayed the visible goal. The next implementation is a bounded, from-scratch, directly supervised single-writer word-image baseline; reader readiness still applies before using the reader as a content-loss network, but does not block this separate experiment.

Prepared and inspected a data-only scope audit: writer `000`, 1,930 lowercase training images covering 679 words, 147 seen-validation images, and 215 content-holdout images covering 69 globally unseen words. Verified 32 deterministic training references and preserved all partitions. No training occurred. The baseline is not implemented or quality-proven; no new training command is ready. See [the reassessment](generation-reassessment.md) for evidence, research alternatives, explicit limitations and proposed fit/generalization stopping gates. The balanced reader's failed targets and existing release gates remain unchanged.

## Balanced reader run completed; substantial-gain criteria failed

The owner ran 1,500 updates before configured early stopping; final evaluation completed. Recorded time was 18.07 minutes. Source/checkpoint hashes and finite checkpoint loading passed; metrics recomputed from all saved predictions matched the summary (inference was not replayed during this review). Unseen exact words rose only from 1,434/4,250 (33.74%) to 1,443/4,250 (33.95%); CER decreased from 25.81% to 25.21%. Both frozen substantial-gain targets were missed.

Selection CER improved by 0.4942 percentage points, just below the 0.50-point patience threshold, explaining the stop despite a small improvement. Best block 3 was saved. No training or automatic repeat was launched. Generated handwriting remains unchanged. See `docs/word-reader-balanced-review.md` for the stopping-rule details, limits and failed target assessment.

## Error/gap audit completed; frequency-balanced reader candidate prepared

Audited training vocabulary, held-out word frequency/length, writer representation, case, character counts and real error crops. The ten most common training words account for 27.59% of images. Within the 4–6-letter selection group, common-word exact accuracy is 63.82% versus 29.97% for rare words. Tiny vertical shifts/contrast changes do not show a severe fragility in the tested subset. These are measured associations, not a single proven cause.

Implemented optional training-only inverse-square-root word-frequency weights, clipped to a 10:1 per-example ratio. All training images remain in the coverage sampler; validation remains unweighted. Prepared `configs/word-reader-balanced.json` using the latest trained spatial checkpoint, fixed full-development evaluation, a 5,000-update cap and plateau stopping. Predeclared substantial-gain criteria are at least 43.74% unseen-word exact accuracy, at most 20% unseen CER and no worse selection CER. These are targets, not promised results.

Verification: 176 default tests passed, four weight-update tests skipped, lint passed. Real-data gradient-only preflight preserved weights and all 256 checked predictions. No training was launched. See `docs/reader-gap-audit.md` and `artifacts/reader-gap-audit/` for findings, provenance, limits and the manual command. Generated handwriting remains unchanged.

## Second spatial continuation stopped on the configured plateau threshold

The owner completed 4,500 updates and 72,000 draws in approximately 40.2 minutes. Early stopping triggered after block 9; block 8 was selected for lowest selection CER. On the same full development sets, unseen-content exact words rose from 1,215/4,250 (28.59%) to 1,434/4,250 (33.74%) and CER fell from 28.53% to 25.81%. Selection exact words rose to 2,035/3,612 (56.34%), with CER 18.73%.

Progress has slowed enough to trigger the configured stopping rule; this does not prove complete convergence. The reader remains below readiness targets, and generated handwriting is unchanged. No further training was launched. See `docs/word-reader-spatial-continue-2-review.md` for the stop-rule explanation and full-set comparison.

## Full development evaluation confirms progress; next stage prepared

Evaluated the current reader without training on all 7,862 development images: selection exact words 1,900/3,612 (52.60%), CER 21.59%; unseen-content exact words 1,215/4,250 (28.59%), CER 28.53%. These agree closely with the small probes, but remain below readiness targets. Checkpoint bytes and parameters were unchanged; reserved evaluation data was not opened.

Prepared `configs/word-reader-spatial-continue-2.json`, using the latest spatial weights, unchanged learning rate, seed 316 and a maximum 5,000 updates. Checkpoint selection and plateau stopping now use the full 3,612-image selection set; final evaluation uses all 4,250 unseen-content words. No training was launched. See `docs/word-reader-full-development.md` and `artifacts/word-reader-full-development/review.json`.

## Spatial continuation completed: unseen-word exact accuracy 28.52%

The owner completed all 5,000 updates and 80,000 draws in 37.74 minutes of recorded training/evaluation time. Stop reason was budget completion; best checkpoint was block 10. Source/checkpoint hashes passed, and fixed-weight replay reproduced all 512 saved development predictions. On the same unseen-content probe, exact words rose from 20/256 to 73/256 (28.52%) and CER fell from 45.16% to 29.40%. Selection exact accuracy rose from 86/256 to 135/256 (52.73%). Unseen-content substitutions, deletions and insertions all decreased.

The reader remains below readiness targets; generated handwriting is unchanged. No new training was launched during review. See `docs/word-reader-spatial-continuation-review.md` and `artifacts/word-reader-spatial-continue-review/review.json`.

## Spatial continuation prepared with plateau stopping

The latest reader is still weak, but selection CER improved at all five checkpoints and fell by another 5.42 percentage points in the last block. Prepared `configs/word-reader-spatial-continue.json` to reuse the latest spatial weights, use learning rate 0.0005 and seed 315, and cap further fitting at 5,000 updates (80,000 draws). Added optional selection-only plateau stopping: three checks without cumulative CER improvement of 0.5 percentage points. Existing presets leave early stopping disabled. Initial/best checkpoint preservation and fixed development identities remain in place.

Verification: 167 default tests passed, four weight-update tests skipped, lint passed; checkpoint/config/data preflight passed. No training was launched and no new accuracy gain is claimed. See `docs/word-reader-spatial-continuation.md` for the manual command, budget, stopping rule and review criteria.

## Spatial reader training completed; fewer missing letters, accuracy still below target

Owner completed all 2,500 updates and 40,000 word draws in 21.66 minutes of recorded training/evaluation time. Coverage includes all 33,348 training images before repetition. Checkpoint/source hashes passed and inference-only replay reproduced all 512 saved development predictions. Unseen-content CER fell from 71.83% to 45.16%; exact words increased from 7/256 to 20/256 (7.81%). Deletions fell from 443 to 96, while 379 substitutions remain. Selection exact accuracy rose to 86/256 (33.59%).

The reader now retains substantially more letters, but is not ready to guide generator training. Generated handwriting remains unchanged. No additional training was launched during review. See `docs/word-reader-spatial-review.md` and `artifacts/word-reader-spatial-2500-review/review.json`.

## Short-output diagnosis complete; spatial reader and data-coverage changes prepared

Fixed-weight analysis found 443 deletions among 816 unseen-content errors, with 0/49 words of seven or more characters read correctly. Prefix beam decoding left exact accuracy at 7/256, so decoder replacement is insufficient. The time axis has enough capacity for all analyzed transcriptions.

Implemented an optional zero-initialized full-height residual adapter that preserves all existing reader weights and initial predictions, plus a sampler that continues through the training permutation across capped blocks. Replaying the prior 40,000-draw schedule showed 24,864 unique images; the new schedule covers all 33,348 before repetition. Initial selection is now included when choosing the best checkpoint. Prepared the manual `configs/word-reader-spatial-2500.json` using the latest broad-reader weights; no training was launched, and no accuracy gain from these changes is established.

Verification: 160 tests passed, four weight-update tests skipped, lint passed. Real-data preflight reproduced identical log probabilities on all 512 development images and verified nonzero adapter gradients without updating weights. See `docs/word-reader-spatial.md` for the command, measured evidence, limitations and remaining generator work.

## Broad reader run completed: measurable progress, still below target

The owner completed 2,500 additional updates and 40,000 word draws in 17.59 minutes of recorded training/evaluation time. Best checkpoint is epoch 5. Source/checkpoint hashes passed, and inference-only replay reproduced all 512 saved development predictions. On the same unseen-content probe, exact words increased from 0/256 to 7/256 (2.73%) and CER fell from 92.61% to 71.83%. Known-writer selection accuracy rose from 3/256 to 53/256 (20.70%).

Outputs remain frequently wrong and short (2.72 predicted characters versus 4.44 reference characters on average in the unseen-content probe). The reader is not ready to guide generator training, and generated handwriting has not changed. No training was launched during this review. See `docs/word-reader-broad-review.md` and `artifacts/word-reader-broad-2500-review/review.json`.

## Memorization diagnostic completed normally; broader reader run prepared

Owner's 16-word diagnostic passed at updates 200, 225 and 250, then exited early as designed after 4.60 minutes. Checkpoint/source hashes passed and fixed-weight replay reproduced 16/16 exact training words, 0/16 blank controls and 1/16 reversed controls. No restart is needed. On the same 256 unseen words it still reads 0 exactly (111.00% CER, including insertion errors), so this is a successful learnability check, not general reading competence or generated-handwriting improvement.

Prepared the manual `configs/word-reader-broad-2500.json`: warm-start the original 200-update broad-data reader with a fresh optimizer, then 2,500 capped updates across broader IAM training examples, keeping fixed 256-example development probes. This preserves the overfit 16-word checkpoint for diagnostics. No training was launched. See `docs/word-reader-memorization-review.md`; older pending statements below describe earlier milestones.

## Fixed-weight reader diagnosis and bounded memorization check prepared

Compared the latest reader on the same 256 real, blank and horizontally reversed images, preserving widths. Real inputs: 92.61% CER and 0/256 exact; blank: 93.84% CER and 0/256 exact, all `t`; reversed: 93.31% CER and 0/256 exact. Predictions respond to pixels, but reading competence remains absent. Checkpoint bytes and in-memory weights were unchanged. This does not identify a single causal bug.

Prepared `configs/word-reader-memorize-16.json` and the manual-only `ml.train_word_reader_sanity` command: reuse existing reader weights, fit 16 verified training words, check exact transcriptions plus image ablations every 25 updates, stop after three consecutive passes or at 600 updates. The original checkpoint is preserved; memorization never sets generator-readiness true. Input-only preflight confirmed the real 16-image batch and its starting 0/16 exact score. **Owner execution is pending; no new training or generated-handwriting improvement is claimed.** See `docs/word-reader-recovery.md` for evidence, command, gates and next steps. Older sections below record prior milestones.

## Latest word-reader check completed; reader still below target

Current implementation checks: 145 default tests passed, four weight-update tests skipped. No optimizer was run for the diagnosis or manual-run preparation.

The owner completed the 200-update check on 3,200 word draws. Recorded training plus evaluation time was 3.10 minutes, substantially faster than the earlier smoke-based extrapolation. Checkpoint/source hashes passed. Inference-only replay reproduced all 256 content-probe predictions: 92.61% CER, 0% exact-word accuracy, 151 single `t` outputs, 97 single `a` outputs and eight blanks. The reader is not ready to guide a generator; no new training was started. See `docs/word-reader-200-review.md` and `artifacts/word-reader-check-200-review/`.

## Owner completed word-reader smoke; reading competence not established

The owner ran eight updates on 128 real words. Checkpoint/source hashes passed, and inference-only replay reproduced the 32-word unseen-content probe: CTC loss 4.7266, 100% CER, 0% exact-word accuracy and 32/32 empty predictions. Execution works; the reader is not ready to guide generator training. No further training was launched during review.

Recorded training time was 67.15 seconds. The original 2,500-update pilot has a rough smoke-based extrapolation of 5.8 hours, with substantial startup/shape/system-load uncertainty. Prepared a smaller manual check, `configs/word-reader-check-200.json`: 200 updates, 3,200 training-word draws and 256 samples per development probe. See `docs/word-reader-smoke-review.md` for results, the command and budget caveats.

## Content word reader implemented; owner-run execution pending

Implemented the randomly initialized 541,258-parameter convolutional/bidirectional-GRU reader, CTC alignment/greedy decoding, CER and exact-word metrics, separate checkpoint contract and bounded manual training/evaluation commands. Width masks and packed recurrent sequences preserve shorter words when batch padding changes. This network is the future generator's content-loss reader; it is not an independent generator evaluator.

The next owner action is the eight-batch smoke run in `docs/word-reader.md`. The optional pilot is capped at 2,500 updates and 40,000 training-word draws; its fixed selection/probe samples are explicitly partial diagnostics. No word-reader optimization was launched by the agent, and no learned reading accuracy or word-generation improvement is claimed. The style network and word generator remain to be implemented.

Verification: **136 tests passed, 4 weight-update tests skipped; lint passed**. An untrained real-data preflight checked finite CTC loss and image gradients on eight IAM words while freezing and preserving reader parameters. Non-training tests cover repeated-character decoding, padding invariance, checkpoint-role compatibility, error-rate arithmetic and deterministic evaluation. See `.cache/word-reader-preflight.json`.

## IAM word-data foundation completed; next model specified

Downloaded IAM 3.0 from its author-authorized public Zenodo migration, verified the published archive MD5 checksums and recorded SHA-256 hashes. Imported all 115,320 word images with the ASCII metadata and official line-task split definitions. No IAM account was needed for the migrated public files. This remains non-commercial research data.

The corrected preparation is `artifacts/iam-words/`: **33,348 training words from 283 writers**, 4,250 globally held-out case-folded word-content samples, and 3,612 additional known-writer image holdouts. Official unknown-writer validation and final-test pixels remain unopened. All 115,320 metadata rows are accounted for in accepted/reserved records or explicitly logged exclusions. A rare-writer line-retention adjustment is recorded and does not move unseen-word content into training. The initial diagnostic preparation is marked superseded.

Implemented variable-width, aspect-preserving 64-pixel word preprocessing, per-image integrity checks, a CTC token/batch contract, a real training-reference grid and a fixed 16-word single-writer training sanity set. Six punctuation characters have no examples under the initial word filter, and several capitals/digits are rare; the preparation documents that coverage limitation rather than claiming full support.

Verification: **120 tests passed, 3 weight-update tests skipped; lint passed**. Loaded and checked 32 real images from each of the three development partitions, including width/CTC alignment, finite pixel batches, nonnegative style IDs, disjoint held-out vocabulary and unchanged existing generator hashes. The word generator and auxiliary networks are specified but not implemented or trained. See `docs/word-generation-design.md`, `artifacts/iam-words/manifest.json` and `.cache/iam-verification.json`.

Read-only compute inspection still shows PyTorch 2.13.0+cpu, no CUDA runtime, and a detected GTX 1650 with 4 GiB VRAM. GPU training is not ready. No packages or Windows settings were changed, and no training was launched.

## Conditional GAN pilot completed by owner; candidate below baseline

The owner completed the smoke run and all 1,500 pilot batches (192,000 real-example draws), with 20.49 minutes of recorded training-loop compute. Inference-only comparison found 8.18% exact-label recognizer agreement for epoch 3 versus 60.92% for the previous refined VAE; ignoring case, 11.31% versus 78.73%. Strokes are darker but characters are often wrong, repeated shapes show limited variation, and the rendered paragraph is substantially less readable. All three epoch grids and the final page were inspected. Source/checkpoint hashes and finite parameters passed integrity checks. The existing VAE remains the preview default; no additional training was launched. See `docs/gan-pilot-review.md` and `artifacts/comparison-cgan-pilot/`.

## Conditional GAN experiment implementation

Implemented a small from-scratch, class-conditioned generator and spectrally normalized projection discriminator on the existing 28x28 EMNIST split. The preset is bounded to three capped epochs of 500 batches (128 examples per batch): at most 1,500 updates per network and 192,000 real-example draws with replacement. Progress bars show both losses and draw counts. Each completed epoch is saved; adversarial loss is not used to declare a best checkpoint.

The shared checkpoint loader, glyph sampler and paragraph renderer support both the existing VAE and new GAN generator. VAE-only reconstruction evaluation explicitly rejects GAN checkpoints. `ml.compare_gan` prepares fixed-seed grids/pages, independent recognizer agreement, real/generated descriptive diagnostics, variation checks and blank human-review annotations. It does not compare incompatible reconstruction/adversarial losses or certify quality.

No GAN training was launched by the agent. The owner's completed pilot did not improve quality, and the preview still defaults to the existing refined checkpoint. See `docs/gan-pilot.md` for commands, research sources, provenance and evaluation, and `docs/gan-pilot-review.md` for measured results.

Implementation verification: **99 default tests passed; 3 weight-update tests skipped; lint passed**. Tests include gradient boundaries without optimizer updates, cGAN checkpoint/rendering compatibility and comparison report wiring with synthetic fixtures. Real-data preflight verified source/split hashes, all 62 training classes and unchanged hashes/finite inference for the three existing trained generator checkpoints. The owner subsequently completed the real smoke/pilot runs; their measured quality results are reported separately above.

## Local paragraph preview

Implemented the inference-only local compositor and report using the latest `refined-continue-5` checkpoint. The renderer handles proportional glyph widths, cap/x-height and descender heuristics, spaces, explicit paragraph breaks, word wrapping, long tokens, margins, bounded multi-page output and plain/ruled/grid backgrounds. Procedural punctuation is explicitly disclosed; model characters never use a font fallback.

The generated sample is in `artifacts/paragraph-preview/`: `preview.html`, PNG page(s), `pages.zip`, `input.txt` and a provenance/placement manifest. The report shows original typed text beside generated pages, with display zoom and download links. New text/layout settings are supplied through `python -m ml.preview`; this is a local artifact viewer, not the live web application.

Verification: 77 default tests passed; two weight-update tests skipped; lint passed. The real-checkpoint example has 260 model glyphs and 13 procedural punctuation marks. Identical-seed rerendering produced identical PNG bytes, ZIP entries match PNGs, and a 1,000-character case wrapped over two pages with all visible characters accounted for. The checkpoint was unchanged and no training was performed. Browser interaction inspection was blocked for local `file://` URLs; PNGs and report/export structure were checked without bypassing that restriction.

See `docs/paragraph-preview.md` for commands and limits. Model identity/readability/style errors remain visible; the compositor does not establish that the generator has passed its quality gate.

## Completed and verified

- Created the repository on `codex/ml-foundation`, Python 3.12 environment, pinned direct dependencies, exact CPU environment lockfile and CI checks.
- Implemented the 62-class conditional convolutional VAE, versioned label/preprocessing contracts, training CLI, checkpoint loading, deterministic glyph sampling and evaluation/review exports.
- Downloaded the official EMNIST archive and verified source integrity through torchvision; recorded SHA-256 hashes for the four ByClass source files.
- Prepared 628,142 training and 69,790 validation examples from 697,932 official training examples. Saved and reloaded the exact split with checksum checks.
- Visually inspected the upright image/label sanity grid. Some handwriting is naturally ambiguous across case or similar characters; the data is not a guarantee of distinguishable glyphs.
- Current default test suite: **136 passed, 4 skipped**, including word-reader CTC/gradient/padding checks, IAM split/leakage checks, aspect-preserving loading, archive safety and source integrity, plus the existing model/rendering checks. Skipped tests update weights and require the owner's explicit `--run-training` opt-in. Lint passed; dependency checks were completed at the earlier foundation milestone, and these additions require no new dependencies.
- Training and validation now show live per-epoch progress, completed/total batches, ETA, throughput, current/running loss and processed examples. Counters advance only after a successful batch; capped pilot runs use capped totals. Use `--no-progress` for plain epoch-summary logs. No training was started to verify this change.
- The owner manually completed the bounded smoke run: 2,048 training and 512 validation examples, one partial epoch. Train loss 463.726; validation loss 462.599 at different KL weights (0.2 / 1.0), so these two numbers are not directly comparable.
- Reloaded the owner's `artifacts/smoke/best.pt` without any training: checkpoint SHA-256 matches, all sampled outputs are finite, and identical seeds reproduce tensors on CPU. The checkpoint has 1,044,961 parameters.
- The smoke-run prior and reconstruction grids remained gray/noisy after eight optimizer steps. Official test examples have not been loaded for evaluation.
- The owner completed a second, independent 200-batch pilot: 51,200 sampled training examples and 5,120 validation examples. Training took 44.91 seconds and validation took 3.26 seconds. Losses were finite; train loss was 213.766 (beta 0.2), and validation loss was 192.398 (beta 1.0).
- Visually reviewed the pilot grids: reconstructions now show recognizable character shapes, while prior samples remain blurry and frequently inconsistent with the requested label. This is progress in reconstruction, not an established class-correctness result. The smoke and pilot used different validation sample counts, so their aggregate validation losses are not an identical evaluation protocol.
- Reloaded the pilot checkpoint and verified its SHA-256, finite inference outputs and same-seed reproducibility. No training was run during review. Character correctness, readability and diversity have not passed release evaluation.
- The owner completed `artifacts/baseline-10`: ten full epochs with 628,142 sampled training examples and 69,790 validation examples per epoch. All recorded losses are finite. The best checkpoint is epoch 10, with final train loss 134.208 and validation loss 132.493.
- Validation loss decreased from 160.014 at epoch 1 to 132.493 at epoch 10 (17.2%) under the same validation partition, fixed noise and beta. This is a loss reduction, not a measured character-accuracy score. Training beta changes during warmup, so early training losses are not directly comparable across epochs.
- Recorded training plus validation compute time for the baseline was 124.27 minutes, excluding setup/checkpoint/sample-saving overhead. Reloaded its best checkpoint and verified its hash, finite outputs and same-seed reproducibility without training.
- Epoch-10 prior samples now show many recognizable requested letters/digits, with remaining blurry, ambiguous or mismatched cases. Reconstructions visually preserve substantially more character shape than the initial smoke run. Formal class correctness, diversity and passage-level readability remain unmeasured; the official test set remains untouched.
- Full baseline validation was reproduced over all 69,790 examples, and 1,240 generated samples were exported for review. Separate balanced real/generated diagnostics found more intermediate-gray ink and lower edge energy in generated glyphs. These are descriptive softness indicators, not correctness scores. See `docs/quality-review.md` and `artifacts/quality-baseline-10/diagnostics.json`.
- Implemented an independent, from-scratch recognizer with disjoint checkpoint-selection and real-audit subsets, plus an inference-only per-class generation scorer. The owner completed three recognizer epochs; the agent did not launch training.
- Recognizer best epoch: 3. Exact-label real-audit accuracy is 79.77% on 34,897 real examples (macro 77.30%; ignoring case 88.36%). Verified checkpoint integrity and completed inference-only scoring of 6,200 generated glyphs: 57.94% exact-label agreement, 76.15% ignoring case. These are limited classifier proxies, not human accuracy or sharpness scores. The evaluator has below-85% real-audit accuracy in 32 classes; detailed limitations are recorded in `docs/quality-review.md`.
- Implemented a refined generator with label modulation at two decoder stages, a learned residual stroke refiner and a gradient-preservation loss. The owner completed all five additional epochs from the project's own ten-epoch baseline. The evaluator was not in the training loss.
- Verified initialization using the actual trained baseline: source hash unchanged, baseline/refined generated pixels exactly equal before learning, maximum pixel difference 0.0. Parameters increase from 1,044,961 to 1,052,994. No training was performed for these checks.
- Completed the paired inference-only comparison: common validation loss 132.4933 to 130.4517, exact-label agreement 57.94% to 59.48%, case-insensitive agreement 76.15% to 77.26%, soft-ink fraction 80.30% to 75.98%. The refined checkpoint is epoch 5 and recorded compute time was 129.01 minutes. Hash checks passed for both the refined checkpoint and retained source baseline.
- Gains are modest and some classes regress. The frozen recognizer is imperfect, and the matching continued-baseline control has not been run, so the changes cannot yet be isolated from the benefit of extra training. The release quality gate remains unmet. See `artifacts/comparison-refined/comparison.json` and `docs/refinement-experiment.md`. No training was launched during comparison.

## Local runtime

Working environment: PyTorch 2.13.0+cpu and torchvision 0.28.0+cpu. The attempted CUDA build failed during DLL import with Windows error 1455 (paging-file memory). Windows configuration was not changed. GPU training is not ready until the owner resolves that runtime issue; CPU training is available.

## Owner's completed manual smoke test

The owner ran the bounded real-data training smoke test from the repository root:

```powershell
.\.venv\Scripts\python.exe -m ml.train --config configs/baseline.json --output artifacts/smoke --device cpu --epochs 1 --max-train-batches 8 --max-validation-batches 2
```

This intentionally updates model weights and was launched by the owner. It created:

- `artifacts/smoke/history.json`, configuration, environment and summary;
- `artifacts/smoke/best.pt` and `last.pt`;
- prior and reconstruction grids in `artifacts/smoke/samples/`.

Eight training batches verify execution; they are not enough to claim usable handwriting. Finite losses, saved sample grids and checkpoint reloading were verified. Sampling can also be inspected manually:

```powershell
.\.venv\Scripts\python.exe -m ml.generate --checkpoint artifacts/smoke/best.pt --characters AaBb09 --variations 4 --output artifacts/smoke-check/glyphs.png
```

## Owner's completed 200-batch pilot

The owner ran a longer bounded pilot to inspect learning progress and measure CPU speed:

```powershell
.\.venv\Scripts\python.exe -m ml.train --output artifacts/pilot-200 --device cpu --epochs 1 --max-train-batches 200 --max-validation-batches 20
```

This trained on 51,200 sampled examples (balanced sampling uses replacement) and validated on 5,120 examples. It started from random weights and remains labelled pipeline-smoke-only because the epoch was bounded. It did not resume the eight-batch smoke checkpoint. Observed training/validation compute time was 48.17 seconds in total.

## Owner's completed ten-epoch research baseline

The owner completed the following ten-epoch research baseline, covering full training/validation partitions and the five-epoch KL warmup. It started from scratch in a new directory and did not resume the pilot. Measured training/validation time was approximately two hours four minutes. The best checkpoint is epoch 10.

```powershell
.\.venv\Scripts\python.exe -m ml.train --config configs/baseline.json --output artifacts/baseline-10 --device cpu --epochs 10
```

Run outputs must be new or empty. If rerunning a smoke test, choose a new output directory instead of deleting or overwriting prior experiments. Interrupted optimizer-state resume is not implemented in this milestone.

## Still to implement or validate

The owner completed the evaluator training:

```powershell
.\.venv\Scripts\python.exe -m ml.train_recognizer --output artifacts/recognizer --device cpu --epochs 3
```

The owner completed the generator-refinement experiment with:

```powershell
.\.venv\Scripts\python.exe -m ml.train --config configs/refined.json
```

The paired comparison completed and is saved in `artifacts/comparison-refined/`. It reports a modest diagnostic improvement, not a major visual-quality jump. Human review, diversity evaluation, better handling of case/shape ambiguity and a frozen visual reference/canonical passage remain necessary. The continued-baseline control remains available to distinguish architectural/loss changes from extra training. Glyph metrics and page composition now have a local assessment prototype; further calibration, FastAPI service, Next.js interface, service benchmarking, release QA and deployment remain to implement. No additional training was started during this review.

## Latest continuation results

The owner completed the additional five epochs in `artifacts/refined-continue-5`. Recorded training/validation compute time was 99.57 minutes. The inference-only paired comparison increased exact-label agreement from 59.48% to 60.92%, ignoring-case agreement from 77.26% to 78.73%, and reduced common validation loss from 130.4517 to 130.0756. Soft-ink fraction changed from 75.98% to 75.45%, with little visible difference in sample grids.

Learning continues with modest character-control gains; clarity improvement is small. These are imperfect-recognizer diagnostics, not human accuracy or a passed release gate. The evaluation reused verified cached prior-checkpoint reports and did not open the official test set or initiate training. See `docs/continuation-review.md` and `artifacts/comparison-refined-continue/comparison.json` for the protocol and limits. No additional training is queued.
