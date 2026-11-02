# Generator refinement experiment

Status as of 27 October 2026: the owner completed all five refinement epochs. The agent completed the paired inference-only comparison. Results show a modest improvement; the model has not passed the handwriting-quality gate.

## Observed results

The best refined checkpoint is epoch 5. Recorded training plus validation time was 129.01 minutes, excluding setup/checkpoint/sample-saving overhead. Its source baseline hash matches the retained original, and the refined checkpoint hash matches its saved summary.

| Paired diagnostic | Baseline | Refined |
| --- | ---: | ---: |
| Common beta-1 BCE + KL validation loss | 132.4933 | 130.4517 |
| Validation edge error | 31.4537 | 28.3568 |
| Exact-label recognizer agreement | 57.94% | 59.48% |
| Agreement ignoring case | 76.15% | 77.26% |
| Mean soft-ink fraction | 80.30% | 75.98% |
| Mean edge energy | 0.02004 | 0.02183 |

Both models were evaluated on the same 69,790 validation examples with fixed noise, 6,200 paired generated samples for recognizer scoring, and 1,240 paired generated samples for descriptive softness diagnostics. Full outputs are in `artifacts/comparison-refined/`.

Of the 6,200 paired predictions, 250 changed from recognizer disagreement to agreement and 154 changed in the reverse direction, a net gain of 96. This is not a ground-truth count of corrected handwriting. The scorer's real-audit exact-label accuracy remains 79.77%, with material case/shape errors.

Example generated-agreement gains include `A` (69% to 77%), `G` (54% to 63%) and `Q` (41% to 48%). Some classes regressed, including `B` (88% to 85%) and `i` (74% to 69%). These per-class rates use 100 generated samples each. Visual grids remain close to the baseline, with softer/ambiguous cases still present.

The matching continued-baseline control has not been run, so observed gains cannot yet be attributed specifically to the new layers or edge loss rather than extra training. These are preliminary development diagnostics, not a release certification or human readability score. No further training was started during the review.

Verification: 54 non-training tests passed; two weight-update tests stayed skipped. Against the actual baseline checkpoint, warm-start output tensors matched exactly (maximum pixel difference 0.0), and the source checkpoint's hash was unchanged. The refined model has 1,052,994 parameters versus 1,044,961 in the baseline. Backward-only tests confirmed gradients reach both conditioning modules and the residual output without any optimizer updates.

## Problem and hypothesis

The original generator produces soft glyphs and has preliminary exact-label agreement of 57.94% (76.15% ignoring case) on 6,200 prior samples. The independent recognizer itself is imperfect: real-audit accuracy is 79.77%, macro 77.30%. Agreement is therefore a development proxy, not ground-truth accuracy or readability.

The refined model targets two observed weaknesses:

- **Label control:** add feature-wise label modulation at both the 7x7 and 14x14 decoder stages. The character label can affect spatial features directly rather than entering only through the decoder input vector.
- **Stroke clarity:** add a small learned convolutional residual refiner, conditioned through those features, and an auxiliary loss that matches horizontal/vertical ink gradients during reconstruction.

Feature modulation follows the FiLM idea: affine feature changes conditioned on label information. Reference: Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, https://arxiv.org/abs/1709.07871. The paper does not establish that this particular handwriting experiment will improve.

The extra edge term is a hypothesis about preserving stroke transitions, not proof of sharpness. It can trade off reconstruction, label control and variation. More edge energy can also indicate noise. Real and generated samples must still be visually reviewed.

## Initialization and compatibility

`configs/refined.json` initializes from our own `artifacts/baseline-10/best.pt`. All compatible learned encoder/decoder weights are copied. Label-modulation layers start as identity transforms; the residual output starts at zero. A successfully transferred model therefore reproduces baseline pixels before learning. This continues the project's own learned model; it does not use external pretrained weights or an AI API.

The new run gets a fresh optimizer and a lower learning rate of 0.0002. It keeps beta at 1.0 with no new KL warmup, because the source checkpoint has already completed warmup. Training noise is reseeded after initialization so the refined run and continuation control can use the same seeded noise sequence.

Version-1 baseline checkpoints still load. New checkpoints use format version 2 and record the architecture plus source checkpoint hash/epoch and which parameters are newly initialized. Unsupported dimensions, mismatched dataset splits and silent removal of learned refinement layers are rejected. This is weight initialization, not optimizer-state resume.

## Manual run

From the repository root, the owner can run the five-epoch candidate:

```powershell
.\.venv\Scripts\python.exe -m ml.train --config configs/refined.json
```

This starts from the trained baseline and learns the added layers along with existing generator weights. Its output directory is `artifacts/refined-5/`. Progress bars remain enabled. Extra convolutional work can affect CPU time; no runtime or quality promise has been measured for this training path.

An optional owner-run runtime pilot uses a separate directory:

```powershell
.\.venv\Scripts\python.exe -m ml.train --config configs/refined.json --output artifacts/refined-pilot --epochs 1 --max-train-batches 200 --max-validation-batches 20
```

A capped run remains smoke-only; it does not certify model quality. Each command initializes from the original baseline unless another compatible checkpoint is explicitly selected. Never overwrite an existing experiment directory.

## Compare the candidate after the manual run

```powershell
.\.venv\Scripts\python.exe -m ml.compare_generators --candidate artifacts/refined-5/best.pt --output artifacts/comparison-refined
```

This performs inference only. Baseline and candidate use the same validation partition/noise, 100 generated examples per class with paired seeds, the same frozen recognizer, and 20 generated/real references per class for diagnostics. It exports common validation metrics, per-class predictions/confusions, reference/sample grids, descriptive softness statistics and a comparison summary. The official test set remains untouched.

The candidate's training objective includes an edge term, so its weighted training/validation objective cannot be compared directly with the original loss. The comparison command explicitly evaluates both models under a common BCE-plus-KL objective with beta 1 and zero edge weight, and reports edge error separately.

The recognizer is not part of the generator training loss. No sample rejection, thresholding, contrast enhancement or font fallback is introduced. The evaluation must reflect the model's actual grayscale output. Review stroke continuity, wrong labels, case ambiguity and variation before considering the candidate better; there is no automatic release approval.

## Continued-training control

To distinguish the proposed changes from the effect of extra training, an optional matching control is prepared:

```powershell
.\.venv\Scripts\python.exe -m ml.train --config configs/continue-baseline.json
```

It initializes from the same source checkpoint and uses the same seed, optimizer settings, five-epoch budget, beta, sampling and data split. Only the architecture, edge weight and output directory differ. The candidate bundles architecture and edge-loss changes; a later ablation would be needed to attribute gains to either component individually.

The model's learned output resolution remains 28x28. These changes do not add high-resolution source information or guarantee printable handwriting. Higher-detail output may need a separate data/model experiment.
