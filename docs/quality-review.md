# Baseline quality review - 26 October 2026

## Finding

The ten-epoch model is a working generative baseline, but its glyphs are still soft and some requested labels produce ambiguous or incorrect shapes. It has not passed the project's handwriting-quality gate. Further training duration alone is not yet justified as the remedy.

## Work completed without training

- Re-ran full validation on 69,790 examples, reproducing the checkpoint's loss of approximately 132.4933 at beta 1.0.
- Exported 20 prior-generated examples for each of 62 classes: 1,240 images in `artifacts/evaluation-baseline-10/review/`, with a blank human-review CSV.
- Compared a separate fixed-seed set of 1,240 generated images against 1,240 real validation references, balanced at 20/class. Full metrics are in `artifacts/quality-baseline-10/diagnostics.json`.
- Visually inspected representative real/generated comparisons. Lowercase `d` and `h` include ambiguous/mismatched outputs. Real `M` references themselves include lowercase-like forms, so a strict visual case distinction is not always available from the dataset.

| Diagnostic | Real references | Generated |
| --- | ---: | ---: |
| Mean edge energy | 0.03199 | 0.02004 |
| Mean fraction of soft ink pixels | 62.9% | 80.3% |
| Near-blank images | 0% | 0% |

Soft ink is defined as intensities between 0.1 and 0.9 among pixels above 0.1. Edge energy is the mean squared adjacent-pixel difference. These support the observation that the generated strokes are softer, but they also depend on stroke shapes/coverage and are not a perceptual-quality or correctness score. Comparisons show independently generated images and real references from the same class; they are not reconstruction pairs.

## Recognizer audit and generated-character agreement

The owner manually trained the independent CNN recognizer from scratch on real handwriting for three epochs:

```powershell
.\.venv\Scripts\python.exe -m ml.train_recognizer --output artifacts/recognizer --device cpu --epochs 3
```

The evaluator has disjoint selection and final real-audit subsets carved from the existing validation partition. Its report records exact-label, case-insensitive and per-class real-audit scores. The official final test data is not opened. The scoring command refuses a smoke/unaudited recognizer and checks compatibility with the generator's data split.

The inference-only scorer evaluated 6,200 generated images (100/class), exporting per-sample predictions and a confusion matrix in `artifacts/agreement-baseline-10/`. Recognizer checkpoint integrity was verified against its saved hash; its best epoch is 3.

| Measurement | Result |
| --- | ---: |
| Recognizer real-audit examples | 34,897 |
| Recognizer real-audit exact-label accuracy | 79.77% |
| Recognizer real-audit macro accuracy | 77.30% |
| Recognizer real-audit accuracy ignoring case | 88.36% |
| Generated exact-label agreement (6,200 images) | 57.94% |
| Generated agreement ignoring case | 76.15% |
| Classes with real-audit accuracy below 85% | 32 of 62 |

The real-audit aggregate uses the natural real class distribution, while generated sampling gives all classes equal weight. Real macro accuracy is the corresponding equal-class summary. These are different image domains, so agreement cannot be converted into human accuracy by dividing by, subtracting or otherwise correcting with the real-audit score.

Frequent predicted confusions include `P -> p` (90/100), `V -> v` (85/100), `Y -> y` (79/100) and `0 -> o` (53/100). The evaluator itself has substantial errors on similar shapes and cases, so these are review flags, not proof that every such output is incorrect. Its real accuracies for `O` and `I` are only 25.5% and 29.6% respectively.

There are also gaps in classes that the evaluator reads well in real examples: generated `Q` agreement is 41%, versus 96.2% real-audit accuracy; `t` is 52%, versus 92.8%; and `G` is 54%, versus 92.9%. These warrant visual review for generator shape/clarity failures. Examples with stronger generated agreement include `r` (93%) and `D` (91%). Each generated per-class rate is based on 100 samples, not a full population.

A recognizer agreement score must never be presented as ground-truth human accuracy, clarity or a release-quality certification. High softmax confidence is not proof of a correct glyph. No recognizer training or additional generator training was launched by the agent during this review. The owner's recognizer run changed the measuring tool, not the handwriting generator. The official final test set remains untouched.

## Improvement decisions after the evaluation

1. Establish the weak classes and failure types from recognizer agreement plus human review. Distinguish identity errors, visually ambiguous labels, soft edges and dataset style/case overlap.
2. Compare decoder/loss variants against the saved baseline on the same frozen sample seeds and evaluation protocol. The current pixel reconstruction objective does not explicitly optimize edge clarity or requested-class recognition; changes need a measured experiment, not a claim that more epochs will solve them.
3. Evaluate any sampling or contrast adjustment as a separate, disclosed variant. Do not sharpen samples for the report while using unmodified output in the product, or filter mistakes without reporting the rejection rate.
4. Treat 28x28 as the model's native learned resolution. Upscaling alone cannot restore missing stroke detail. A genuinely higher-detail model may require higher-resolution source data and a separate experiment.

The measuring tool is now trained and its limits are quantified. The next model decision should address generator shape, clarity and label control, supported by visual review and comparable measurements. Additional training budget and any generator changes should follow that evidence; no further training was started.
