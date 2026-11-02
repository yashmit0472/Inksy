# GAN pilot review - 29 October 2026

The owner completed the eight-batch smoke run and the three-epoch, 1,500-batch GAN pilot. The review performed inference, integrity checks, diagnostics and rendering only. No additional training was launched.

## Execution and integrity

All three capped epochs completed: 500 generator updates and 500 discriminator updates per epoch, 192,000 real-example draws in total. Recorded training-loop time was 1,229.21 seconds (20.49 minutes), excluding checkpoint/sample-saving overhead. Recorded losses and gradient norms were finite, and all saved generator parameters were finite. Negative generator loss in epoch 3 is valid for the hinge objective and does not indicate quality by itself.

Source fingerprints match the current implementation. Source/split compatibility passed. `last.pt` matches its recorded SHA-256 and has exactly the same generator state as `epoch-003.pt`. The epoch-3 file hash is `46554b45991e671b313b9499d9f87f86ef888d3c16ed7b735b0fec809439ef63`; the differently serialized `last.pt` file hash is `63467ec5dbd7d7effde7807f403720e5eeab93b49ac0d42d0a8c19e44700ff79`. Integrity evidence is saved in `.cache/cgan-training-integrity.json`.

## Fixed-input comparison

Compared `artifacts/cgan-pilot/epoch-003.pt` with the retained `artifacts/refined-continue-5/best.pt` using `ml.compare_gan`. The frozen recognizer, renderer, source text, sample counts and seeds were held fixed.

| Diagnostic | Previous refined VAE | New GAN pilot |
| --- | ---: | ---: |
| Exact-label recognizer agreement, 6,200 samples | 60.92% | 8.18% |
| Agreement ignoring case | 78.73% | 11.31% |
| Soft-ink fraction, 1,240 samples | 75.45% | 65.22% |
| Mean edge energy | 0.02223 | 0.03186 |
| Near-blank fraction | 0% | 0% |
| Mean within-class pairwise pixel L1, 20 samples/class | 0.16038 | 0.06500 |

The recognizer has 79.77% exact accuracy on its real-data audit, so generated agreement is an imperfect proxy, not human character accuracy. Pixel softness, edge energy and pairwise differences are descriptive metrics affected by shape, coverage and noise; they do not establish perceptual quality.

The GAN produces darker, more abrupt strokes but often the wrong character. Examples from the scorer include `s` predicted as `y` in 100/100 samples, `M` as `W` in 96/100, `i` as `l` in 93/100, and `4` as `n` in 91/100. These are scorer observations, not manually assigned labels.

All three epoch sample grids and the final rendered paragraph were visually inspected. Many shapes are malformed or mismatched to their requested label; repeated samples within a class often look very similar. The final paragraph is substantially less readable than the previous model's paragraph. Low variation suggests limited use of latent noise, but complete mode collapse is not established: all 20 sampled 8-bit images per class are distinct. Lower pairwise distance alone would not prove collapse.

## Decision and limits

This pilot does not justify replacing the VAE. The preview default remains the previous refined checkpoint. The GAN's correctness and useful variation need diagnosis before another long run; repeating the same configuration is not automatically the next step.

This is a short, fresh GAN pilot against a VAE with much more training history, not an equal-compute comparison or proof that conditional GANs cannot work. The useful finding is that this configuration has not produced a better candidate at the tested budget. No release gate passed, and no exhaustive human character-accuracy measurement was made. Human-review CSV fields remain blank. The official test set remains untouched.

## Outputs

- `artifacts/comparison-cgan-pilot/comparison.json`: full diagnostic summary, hashes and seeds.
- `artifacts/comparison-cgan-pilot/candidate/paragraph/page-001.png`: actual GAN page.
- `artifacts/comparison-cgan-pilot/candidate/paragraph/preview.html`: typed source beside the GAN page.
- `artifacts/comparison-cgan-pilot/baseline/paragraph/page-001.png`: previous model on the same text.
- `artifacts/comparison-cgan-pilot/candidate/glyph-grid.png`: four examples per requested character.
- `artifacts/comparison-cgan-pilot/human-review.csv`: unfilled review sheet.

PNG output was inspected directly. No browser interaction check was attempted; the existing restriction on local `file://` navigation was not bypassed.
