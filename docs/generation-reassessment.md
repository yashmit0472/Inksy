# Generation reassessment — 1 November 2026

> **Implementation update, 2 November:** The proposed word-generator fit stage is now implemented and verified without optimization. Its first owner-run training remains pending. See [word-generator-fit.md](word-generator-fit.md) for the exact command, evidence and scope. The reassessment below remains the record of the original decision; statements that implementation was pending refer to 1 November.

## Decision

Pause routine word-reader continuations. Implement a bounded, directly supervised text-to-word-image baseline next, using one known writer and lowercase words. Its first output must be generated word images, followed by a separate unseen-word evaluation. This is an engineering experiment, not a claim that this architecture will reach product quality.

The reader work was relevant to the proposed GAN's content loss, but making reader readiness block *all* generator implementation prolonged work on an auxiliary task. The current word generator and writer-style network still do not exist. Recent reader training therefore could not improve the handwriting displayed by the old glyph renderer. This sequencing was a project-level mistake.

The balanced reader run still failed its frozen substantial-gain criteria. This reassessment does not reinterpret that result or relax the release criteria in `acceptance.md`.

## Evidence and uncertainty

| Finding | Implication |
|---|---|
| Latest reader: 1,443/4,250 unseen words exact (33.95%); CER 25.21%. Only nine more exact words than its parent. | Do not repeat this reader configuration expecting a demonstrated large gain. |
| Full-set selection CER improved 0.4942 percentage points against a 0.50-point stopping threshold. Only 24,000/33,348 images were visited in that run. | Stopping was valid under the configured rule, but does not prove full convergence or that weighting can never help. |
| Prior checks found correct split integrity, feasible CTC lengths, successful tiny-set reader fitting, and no severe sensitivity to the limited shift/contrast probes. | No demonstrated single pipeline bug explains the remaining quality gap. |
| The reader has three convolutional layers and a linear height-sensitive residual adapter. | A richer visual encoder is a plausible future reader experiment, not an established cure. |
| Paragraph output still uses the old isolated-glyph VAE; the word generator is unimplemented. | Reader accuracy and generated-handwriting quality must remain separate in reporting. |
| Current working environment is CPU-only; earlier CUDA import failed. | No research-scale GPU runtime or 4 GiB training fit has been established. |

Sources for local findings: `word-reader-balanced-review.md`, `reader-gap-audit.md`, `word-generation-design.md`, and their linked artifacts. This reassessment adds data inspection, not new trained-model inference or optimization.

## Why not simply replace the model with diffusion?

[WordStylist](https://arxiv.org/html/2303.16576v2) demonstrates word-image generation without recognition or adversarial training objectives. However, it uses a pretrained Stable Diffusion VAE and reports 1,000 epochs with batch size 224 on an A100. It supports removing a universal OCR prerequisite; it does not establish that an all-from-scratch CPU version will be cheap or successful here.

[Luhman and Luhman](https://arxiv.org/pdf/2011.06704) generate online pen trajectories using IAM Online data, reporting 60,000 updates on a V100. Our prepared offline word crops lack those pen trajectories. That method is not a drop-in replacement for this dataset.

The earlier GAN design remains a possible later route, but its auxiliary networks and adversarial training add dependencies. Neither a new model name nor a longer run is evidence of better output. No external weights, datasets or research implementations were downloaded for this reassessment.

## Prepared scope

`artifacts/generation-reassessment/scope-audit.json` records source digests, exact IDs, character counts and deterministic diagnostic selection. Writer `000` was selected by the largest training-image count, with writer ID as the tie-breaker; validation performance and visual attractiveness were not selection criteria. It has 2,571 total training images.

Restricting original labels to ASCII lowercase words of 2–10 characters gives:

| Existing partition | Images | Distinct exact words |
|---|---:|---:|
| Training | 1,930 | 679 |
| Seen validation | 147 | 94 |
| Content validation | 215 | 69 |

No labels were lowercased and no partition assignments changed. The 69 content-validation words are absent from the entire original training vocabulary, case-insensitively. All 26 lowercase letters occur in this training subset, but `x` has only seven occurrences and `z` only three. Alphabet presence does not imply coverage quality. The content holdout also lacks several rare letters, so it cannot certify the full alphabet.

A fixed seed-42 SHA-256 ordering selected 32 distinct training words and one image per word for a future fit diagnostic. All 32 source-image hashes were checked, loaded through the existing preprocessing, and visually inspected in `real-reference-grid.png`. These are **real IAM references, not generated results**. Some handwriting is ambiguous and varies between samples; this writer is a data-rich experimental scope, not an assurance of attractive handwriting. Reserved writer-validation and test pixels remain unopened.

## Next implementation contract

1. Build a randomly initialized character-sequence encoder and spatial image decoder. Use ordered character features and spatial position to condition the decoder; do not use a word-ID lookup or retrieved training crop at inference. Predict word width from text. The output remains 64 pixels tall with aspect-preserving training targets and masked padding.
2. Start with deterministic, paired image supervision and a width loss for one writer. Measure foreground and edge errors separately so a mostly blank canvas cannot masquerade as success. Keep the weak reader out of the optimization objective. Direct image regression can blur varying examples; this is a known limitation to test, not a solved problem. One fixed target per diagnostic word reduces target ambiguity only for the tiny fit check.
3. Use a separate checkpoint role and provenance contract. Neither glyph checkpoints nor reader checkpoints initialize this generator. Unsupported case, length and punctuation must fail explicitly rather than silently substitute glyphs or fonts.
4. Provide non-training forward, padding, checkpoint and input-gradient checks; model weights must remain unchanged in default tests. Benchmark forward/backward without optimizer updates before publishing a manual training budget. Include a progress bar, elapsed time, ETA, finite-loss checks and fixed sample exports in the owner-run trainer.
5. Keep generated-word inference independent of ground-truth images and widths. A diagnostic using target width must be labeled as such and cannot satisfy the end-to-end gate. Include text-conditioning controls: changed, shuffled and repeated-letter prompts must visibly affect output.

No generator code or runnable training command has been implemented in this reassessment. The next implementation should supply both; the owner retains all optimizer execution.

## Bounded evaluation before another continuation

These are proposed experiment gates to freeze with the implementation before training, not achieved results or release criteria.

- **Fit diagnostic:** the fixed 32 training words only, maximum 1,000 updates and a 20-minute wall-clock cap, stopping at whichever occurs first. Export the complete grid at fixed intervals. Require at least 31/32 words correctly transcribed by a human without seeing the requested text, with no clipping or missing strokes on visual review. Passing demonstrates training-set fitting only. If the cap is impractical in the measured environment, revise the experiment openly before launch, not by running an unbounded job.
- **Generalization pilot:** only after the fit diagnostic passes, use the scoped training pool. Freeze a compute cap from measured runtime before owner execution. Select checkpoints using the seen-validation partition, not the content holdout. Do not repeatedly extend a run because its training loss falls.
- **Decision evaluation:** at the selected checkpoint, generate every one of the 69 distinct held-out words, using predicted widths. Render the same prompts with the retained glyph baseline. Shuffle and anonymize both sets; collect transcription before revealing prompts and record per-word spelling and visual defects. Report distinct-word exact accuracy and CER, and keep length and rare-letter failures visible. These are development results for one writer, not an official IAM benchmark or final test.
- **Continue only with an actual gain:** proposed gate is at least 80% exact on these distinct words (at least 56/69) and at least a 20-percentage-point paired gain over the old glyph baseline (at least 14 more correct words), plus visual review of all outputs for clipping, broken connections and excessive blur. These numerical gates are engineering choices, not paper-derived guarantees. If the baseline is too strong for the paired-gain criterion to be attainable, resolve that before training and preserve the original criterion in the record.
- **If it fails:** publish all failures, stop the automatic continuation chain, and decide whether the evidence points to image-regression blur, failed text conditioning, data/style ambiguity, or insufficient compute. A conditional diffusion or adversarial generator may then be justified, but requires a separate design and measured budget. Do not present another tiny-set success as unseen-text quality.

Only after this evidence should the project expand writer count, restore mixed case/punctuation, and connect word images to the page compositor. Personal-handwriting imitation from a new reference image remains a separate unimplemented capability.

## Verification performed now

Data-only audit completed successfully: manifest/record digest matched, zero training/content vocabulary overlap, 32 inspected source images passed their hashes, and existing partitions were preserved. The contact sheet was visually reviewed. No training, weight updates, checkpoint changes or new model-quality claims were made. No application code changed; the previously recorded default test result is 176 passed / 4 training tests skipped, not a suite rerun for this documentation update.
