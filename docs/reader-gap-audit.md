# Reader gap audit and frequency-balanced candidate

1 November 2026. No training was performed. The owner runs every optimizer-based experiment manually.

## What the evidence says

Baseline: `artifacts/word-reader-spatial-continue-2/best.pt`, SHA-256 `077536c9de0804fcc64fd54451907a421a08763ce1a5fa0300201bcf686dc829`. Existing full-development predictions were joined to verified training metadata. A fixed-weight 1,024-image training probe, a 256-image perturbation probe and a real-error contact sheet supplied additional diagnostics. Official writer-validation/test pixels remain unopened.

| Gap | Evidence | Interpretation and limits |
|---|---|---|
| Frequent words dominate training | Ten case-folded word types comprise 27.59% of 33,348 training images; `the` alone occurs 2,534 times. There are 5,408 distinct case-folded training words. | Per-image training spends substantial weight on repeated vocabulary. This is a candidate cause, not proof of all generalization failures. |
| Rare-word recognition is weak even within a length band | Among held-out selection words of length 4–6, exact accuracy is 63.82% for words with 51+ training occurrences (304 examples), versus 29.97% for words with 1–5 occurrences (297 examples). | The association is not explained solely by comparing short common words with long rare ones. These are observational groups, not a controlled causal experiment. |
| There is a generalization gap | Fixed 1,024-image training probe: 66.80% exact, 13.48% CER. Full selection: 56.34%, 18.73%. Full unseen content: 33.74%, 25.81%. | The training probe is partial and populations differ. Both imperfect fitting and generalization remain; this is not pure memorization or proof that capacity is sufficient. |
| Less-represented writers are harder | Unseen-content exact accuracy: 24.60% for writers with fewer than 50 training images, versus 38.28% for writers with 200+. | Text/style mix also differs across groups. Reweighting vocabulary alone is not guaranteed to solve writer variation. |
| Capitals and rare characters remain weak | Unseen-content words with uppercase letters: 18.18% exact versus 37.01% without. Training has 2,021 lowercase `b` versus 8,155 `h`; several capitals/digits are rare. Six punctuation tokens (`! ? : ; ( )`) have no training examples in the prepared data. | Do not claim full alphabet/punctuation support. Frequency weighting cannot create missing examples. |
| Tiny image shifts do not explain most errors | On a fixed 256-image selection probe: original CER 18.50%; two-pixel upward shift 17.70%; downward shift 18.42%; 80% contrast 18.82%. | These limited perturbations do not show severe translation/contrast fragility. They do not test every handwriting-style variation. No augmentation change is introduced in this candidate. |

Visually inspected 20 deterministically spaced real error crops. They are upright and include readable failures (`nuclear`, `believe`, `distinguished`) as well as ambiguous cursive examples. No systematic orientation error was apparent in this small inspection. This is not a complete annotation audit. `artifacts/reader-gap-audit/error-crops.png` contains real IAM references with the reader's text predictions, not generated handwriting.

The prior full-content edit analysis still shows substitutions as the main problem: 3,826 of 5,172 character edits, with recurring confusions such as `b/h`, `r/n`, and `m/n`. Continued uniform training recently hit the configured plateau threshold. The next candidate therefore changes the training emphasis rather than merely extending the same objective.

## Implemented change

The optional `balance_word_frequency` setting computes weights **only from training records**:

`weight(word) = max(0.1, training_count(casefold(word)) ** -0.5)`

Each example remains in the coverage sampler. Frequent words retain a nonzero contribution; the maximum rare/common per-example weight ratio is ten. Case is preserved in labels and predictions; case-folding only groups counts consistently with the existing vocabulary partition contract.

Training uses unreduced CTC loss, divides each example's loss by its target length as before, then takes the weighted average normalized by the batch's weight sum. With unit weights this matches the existing loss. The standard CTC reduction convention is documented in [PyTorch's CTC API](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.ctc_loss.html); the proposed frequency weighting is a project-specific experiment, not a framework-prescribed solution.

Across the complete training records, the top ten words' share of raw weighting mass falls from 27.59% to 9.26%. That is not an exact gradient-share estimate: batch normalization and actual example losses affect gradients. The ratio is bounded to avoid assigning arbitrarily large influence to rare or potentially noisy examples.

The architecture, image preprocessing and learned weights are unchanged. No dictionary, spelling correction, pretrained weights, generated labels or validation examples are used for fitting. Validation remains **unweighted** and reads original images. Historical training-loss values are not comparable with the new weighted training loss; unweighted development metrics remain comparable.

## Next manual run

```powershell
.\.venv\Scripts\python.exe -m ml.train_word_reader --config configs/word-reader-balanced.json
```

- Start from the latest selected spatial checkpoint, preserving learned weights; fresh optimizer, not exact optimizer-state resume.
- Retain architecture, learning rate 0.0005 and complete-coverage sampling. Seed 317 records a new order. Vocabulary weighting is the substantive training change.
- Maximum 5,000 updates / 80,000 word draws; selection checks every 500 updates on all 3,612 selection images.
- Stop after three checks without a cumulative selection-CER improvement of 0.5 percentage points. Preserve the starting model if no candidate beats it under the existing selection criterion.
- At completion, score the best checkpoint on all 4,250 unseen-content words. Save to the new `artifacts/word-reader-balanced` directory; existing runs remain untouched.
- Budget roughly 45 minutes at recent speeds, subject to machine load and early stopping. No automatic follow-on run.

The source frequency hash, formula and weighting summary are recorded in `training-weighting.json` and checkpoints. The trainer also records the balance module's source hash. Old presets retain uniform-example weighting by default. Combined sampling and existing CTC progress bars continue to show executed updates and loss; only held-out scores establish improvement.

## Predeclared review criteria

We will call the next result a **substantial gain** only if all these full-set conditions hold:

1. Unseen-content exact accuracy at least **43.74%** (at least 1,859/4,250), a gain of ten percentage points over 33.74%.
2. Unseen-content CER at most **20%**, versus 25.81% now.
3. Selection CER no worse than the current **18.73%** baseline (allow only numerical tolerance when comparing stored values).

These are engineering acceptance criteria, not a forecast, a statistical significance test or a promise. They are frozen in `artifacts/reader-gap-audit/next-run-targets.json` before training. A smaller improvement must be reported as such. A matched uniform-training control would be needed to isolate how much of any gain came from weighting rather than additional training and ordering.

Also inspect rare words, capitals, writer coverage and long words, including regressions. If the candidate fails, review the visual representation and data coverage with these measurements before choosing another experiment; do not silently lower the target or repeat the same configuration indefinitely. The separate provisional reader-readiness targets (80% unseen exact words and 10% CER plus review) still apply.

## Verification and project gaps

**176 default tests passed; four weight-update tests skipped; lint passed.** Tests check training-only frequency derivation, case-fold counts, bounded weights, unchanged input images/labels, explicit weighted CTC arithmetic, unit-weight equivalence, finite gradients without weight updates, invalid weights, and unweighted evaluation even if a batch accidentally contains training weights.

Real-data preflight checked 32 training images with both loss calculations and finite nonzero gradients, verified 256 unchanged baseline predictions, and confirmed unchanged model weights and checkpoint bytes. It did not construct an optimizer or perform training. Evidence is in `artifacts/reader-gap-audit/`: `data-gaps.json`, `generalization.json`, `audit-provenance.json`, `balanced-preflight.json`, `next-run-targets.json`, and the error contact sheet. Gradient correctness is not evidence of a future accuracy gain.

Good generated handwriting is still unestablished. The current paragraph generator is the retained glyph VAE. The word generator, writer-style network and independent word-level generator evaluation remain unfinished. This reader candidate addresses one prerequisite; it cannot by itself improve the generated page or fix missing training-character coverage.
