# Bounded conditional GAN experiment

Prepared 29 October 2026. The owner has since completed the smoke run and bounded pilot; see [the pilot review](gan-pilot-review.md). The GAN candidate scored 8.18% exact-label recognizer agreement versus 60.92% for the previous VAE and rendered substantially less readable text. The preferred checkpoint remains `artifacts/refined-continue-5/best.pt`; its quality gate is still unmet. No GAN training was launched by the agent.

## Why this experiment

The latest VAE continuation improved recognizer agreement from 59.48% to 60.92%, but the paragraph still contains wrong characters, soft strokes and inconsistent styles. Another identical continuation has a weak justification given the small visible gains.

This pilot changes the generation objective and architecture while retaining the existing EMNIST ByClass data, preprocessing and prepared train/validation split. It tests whether adversarial training can produce more convincing character shapes. It does not promise that a GAN will outperform the VAE.

The generator and discriminator both start with random weights. No existing VAE, recognizer, external model, font or AI generation service supplies learned generator weights. The frozen recognizer is used only after training for an imperfect comparison proxy; it is not part of either training loss.

## Model and objective

- Generator: 32-dimensional Gaussian noise plus a learned character embedding, 7x7 features, nearest-neighbor enlargement with convolutions to 14x14 and 28x28, GroupNorm and label modulation. Output is a grayscale foreground mask through sigmoid.
- Discriminator: small convolutional network with spectral normalization and a projection term that scores compatibility between the image and the requested label. It receives the same normalized intensity range for real and generated images.
- Discriminator hinge loss: `mean(max(0, 1 - D(real, label))) + mean(max(0, 1 + D(fake, label)))`.
- Generator loss: `-mean(D(fake, label))`. There is no reconstruction loss or KL term, so GAN loss cannot be compared numerically with VAE loss.
- Each batch performs one discriminator update using detached generated images, then one generator update with discriminator parameters and spectral-normalization buffers frozen. Separate Adam optimizers use learning rate 0.0002 and betas (0, 0.9); gradients are checked for finiteness and clipped at norm 10.

The design is informed by [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784), [cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637), and [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957). This small project implementation is not a reproduction of those papers' reported experiments.

## Data and budget

Uses the already downloaded official [NIST EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) ByClass data. Native samples are 28x28. No new download or data preparation is needed if `artifacts/data/manifest.json` and `split.npz` remain intact. Their source/split hashes are checked before training.

Only the existing 628,142-example training partition supplies training images. Sampling is balanced across all 62 labels, with replacement. Validation examples are not optimized; the official test partition remains reserved. The held-out data and frozen scorer are used later by the comparison tools.

`configs/cgan-pilot.json` caps the run at **3 epochs x 500 batches**, batch size **128**: at most **1,500 generator updates, 1,500 discriminator updates and 192,000 real-example draws**. Draws can repeat images; they are not 192,000 distinct examples. Each displayed epoch is a capped pass, not a full traversal of the training partition.

CPU is selected explicitly because the installed project PyTorch is the CPU build. GPU training has not been repaired or enabled. No reliable runtime estimate is available for this new model; the smoke-run timing and subsequent progress-bar ETA provide an initial indication. Saving sample grids and checkpoints adds overhead.

## Run manually

Run from `C:\Users\yashm\Desktop\CODE\Inksy` using the project environment.

First run eight batches to check execution:

```powershell
.\.venv\Scripts\python.exe -m ml.train_gan --config configs/cgan-pilot.json --output artifacts/cgan-smoke --epochs 1 --max-train-batches 8
```

This is training: it updates both networks. It should display percentage, batch counts, elapsed time, ETA, current/average discriminator and generator loss, and real-example draws. The output is marked `pipeline-smoke-only`. Eight batches do not establish handwriting quality; poor images at this point are expected.

If that completes with finite losses, run the bounded pilot:

```powershell
.\.venv\Scripts\python.exe -m ml.train_gan --config configs/cgan-pilot.json
```

The pilot starts fresh rather than continuing from the smoke run. Output directories must not exist; choose another `--output` for a rerun. `--no-progress` hides bars when redirecting logs. Ctrl+C stops the run; already saved epoch checkpoints remain available, but partial-epoch work is not saved.

## Saved evidence

`artifacts/cgan-pilot/` contains:

- `epoch-001.pt`, `epoch-002.pt`, `epoch-003.pt` and `last.pt`: generator-only checkpoints for inference. Every completed epoch is retained. **There is no `best.pt`**, because GAN loss does not rank perceptual quality.
- `samples/prior-epoch-001.png` etc.: all 62 requested characters, four samples each, fixed seed 42 and no filtering or sharpening.
- `history.json`: losses, real/fake scores, gradient norms before clipping, counts of actual updates, real-example draws and epoch compute time.
- `config.json`, `environment.json`, `data-manifest.json`, `source-sha256.json` and `summary.json`: configuration, initialization, provenance, compatibility, source hashes and budget accounting.

These are inference checkpoints, not full training-state backups. Discriminator and optimizer state are not saved; exact interrupted-run resume is unsupported. The existing VAE checkpoints and preview default are retained.

## Inspect and compare after training

Render the same default passage using the new candidate:

```powershell
.\.venv\Scripts\python.exe -m ml.preview --checkpoint artifacts/cgan-pilot/epoch-003.pt --output artifacts/cgan-paragraph
```

For the complete inference-only comparison:

```powershell
.\.venv\Scripts\python.exe -m ml.compare_gan --checkpoint artifacts/cgan-pilot/epoch-003.pt --output artifacts/comparison-cgan-pilot
```

This loads the current refined VAE baseline and our existing audited recognizer. It exports the same passage with the same renderer/layout/seed for each model, fixed all-class grids, balanced real/generated validation references, 6,200 generated samples per model for recognizer agreement, and within-class variation diagnostics using 20 samples per class. The recognizer's real-data audit accuracy is only 79.77%, so agreement is not human character accuracy.

The comparison also writes `human-review.csv`: observed character, readability and notes are intentionally blank. Review grid cells in row-major order using their one-based indices. It does not automatically label outputs correct or declare a quality pass. To inspect another epoch, use its checkpoint and a new output directory; preserve all reports rather than selecting only favorable examples.

Compare:

1. Correct letters and case, including known confusing pairs such as `0/O/o`, `I/l/1` and `P/p`.
2. Stroke clarity in actual words, alongside the independent character grids. High edge energy alone can reward noisy or broken strokes.
3. Repeated-letter variation. Identical or near-identical outputs for many noise inputs can indicate collapse; pixel differences can also reflect noise rather than useful variety.
4. Per-class regressions and blank/malformed glyphs. An aggregate proxy increase must not hide unusable classes.

Use the pilot to decide whether the new approach shows a promising trend at an acceptable cost. Only plan a longer run if the saved samples and diagnostics justify it. A weak early result is inconclusive about a fully trained GAN, but it is not a reason to spend more hours without diagnosing it. GAN losses may oscillate; a falling generator loss alone is not success.

The comparison is not an equal-compute ablation: the VAE has substantially more training history than this small fresh GAN. Matching inputs and diagnostics makes the practical comparison reproducible, not causally conclusive.

## Limits that remain

Native resolution stays 28x28. Higher-resolution output would need suitable higher-resolution training data rather than simply enlarging EMNIST. Neither this pilot nor independent latent sampling implements a shared writer identity, word context, connected handwriting or cursive joins. Those require a separate data/model change.

The preview uses the existing proportional layout heuristics and procedural punctuation. It does not repair wrong characters. No frontend/API work or automatic replacement of the current model is included.

## Verification status

The default non-training checks cover generator/discriminator gradient boundaries, finite output shapes, deterministic sampling, checkpoint compatibility, prevention of VAE-only reconstruction evaluation on GANs, the existing paragraph renderer and comparison safeguards. They do not update weights. The synthetic GAN training-loop test is explicitly opt-in and has not been run by the agent.

Verification completed: **99 passed, 3 skipped**, with lint passing. Pytest's default sandbox temporary directory was unwritable, so the successful run used a dedicated workspace temporary directory: `--basetemp .cache/pytest-cgan-20261029-final`. A separate read-only preflight checked source/split integrity, all 62 training classes and finite inference with unchanged hashes for the baseline, refined and continued-refined checkpoints. Its report is `.cache/cgan-preflight.json`. Comparison report wiring is tested with synthetic fixtures; a real trained-GAN comparison remains pending.

The owner subsequently completed all 1,500 batches. Inference-only review verified the saved checkpoints and found poorer character correctness and paragraph readability than the retained VAE. See [the measured results and limitations](gan-pilot-review.md); this pilot did not establish a GAN quality improvement.
