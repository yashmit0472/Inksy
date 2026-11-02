# Controlled decoder-detail comparison — 2 November 2026

## Decision and evidence

Prepare one matched comparison of the existing decoder and a position-aware decoder, both warm-started from the owner's first from-scratch word-generator checkpoint. **Neither arm has been trained by the agent.** Keep the 32-word fit scope and its original readability gate. Do not expand the dataset until the fit gate passes.

The prior run failed visually despite decreasing training loss and mostly correct widths. Fixed-weight diagnostics measured generated horizontal absolute image differences at 61.00% of the targets' magnitude, versus 92.71% for vertical differences. This is consistent with the inspected horizontal smearing, but edge magnitude also contains paper texture and is not a readability score or causal explanation.

The image-space edge-loss gradient was finite and roughly half the BCE gradient norm in the four tested batches. It was not absent or orders of magnitude smaller. Therefore this experiment leaves the objective and its weights unchanged instead of simultaneously changing architecture and loss.

Code inspection found that the original generator projects smooth, interpolated text features onto an 8-pixel-high, width/8 grid, then enlarges it three times. Fine-resolution layers had no explicit pixel coordinates. This is a plausible representation constraint, not a proven bug: learned multichannel convolution features can represent more detail than a downsampled grayscale image. The prior run was also still learning when its cap ended. A matched plain-decoder continuation is necessary to distinguish the proposed change from additional optimization alone.

Evidence: `artifacts/word-generator-detail-audit/diagnosis.json`. No model weights were updated for that diagnosis.

## The single experimental change

Optional `architecture="coordinate"` adds normalized per-word horizontal/vertical positions and sine/cosine position signals at frequencies 1, 2, 4 and 8 to the first convolution of each enlargement stage. Coordinates use the individual word's width, not the padded batch canvas. They are masked outside each word. Text remains the only semantic input; there is no image lookup, pretrained component or OCR loss.

The checkpoint migration copies every original parameter and initializes only the new input-channel weights to zero. This preserves the trained function at initialization. Original architecture: 216,474 parameters. Candidate: 225,546 parameters, an increase of 9,072. The old checkpoint format/config continues to load as `plain`; new checkpoints explicitly record the architecture. Reverse migration is refused because it would discard learned position inputs.

Both arms use the exact same parent checkpoint, 32 IDs, batch ordering seed, fresh optimizer, learning rate, objective and budget. Optimizer state is not restored in either arm. Warm-start validation pins the SHA-256 and checks writer, dataset digest and exact training IDs. Run-local update counts exclude the parent's 1,000 updates. The parent and original run outputs remain untouched.

## Owner-run commands

Run these **sequentially**, not concurrently, from the Inksy repository root:

```powershell
.\.venv\Scripts\python.exe -m ml.train_word_generator --config configs/word-generator-detail-control.json
```

Then:

```powershell
.\.venv\Scripts\python.exe -m ml.train_word_generator --config configs/word-generator-detail-coordinate.json
```

The first is the unchanged-decoder control. The second tests the added position inputs. Each has a 1,000-update / 20-minute training budget; final evaluation/export adds overhead. Both include the existing progress bar, gradient checks, previews and checkpoint saving. Existing output directories are never overwritten. These commands update weights and are reserved for the owner.

The latest single-batch, no-optimizer timing probes measured warm forward/backward medians of 0.580 seconds for the control and 0.803 seconds for the candidate. These timings exclude optimizer work, evaluation, exports and variable widths; they are not an end-to-end ETA. System load differs from the earlier 0.141-second preflight. The time cap remains the budget guarantee between updates, not a promised completion time.

New outputs are `artifacts/word-generator-detail-control/` and `artifacts/word-generator-detail-coordinate/`. Each has `best-review/blind-predicted-width.png`, `transcriptions.csv`, labeled diagnostic grids, and its own summary/provenance. Do not use the supplied-target-width grid as end-to-end quality evidence.

## Decision rule frozen before training

1. Verify both runs' initialization digest, IDs, sample ordering settings, objective and update counts. Both should finish the same 1,000 extra updates. If the wall-clock cap makes counts unequal, do not attribute a final-score difference to architecture alone. Record the comparison as incomplete rather than silently increasing the budget.
2. Review every generated training word using predicted widths. Human transcription should precede revealing requested text. The existing gate stays **at least 31/32 exact words and all 32 visually acceptable**, with no missing strokes, clipping or other visible defects. The weak OCR reader is not the scoring authority. The tools cannot enforce reviewer blinding.
3. If the control passes, additional training was enough for this tiny fit task; do not credit the architecture change. Prefer the simpler model unless the candidate has a separately demonstrated benefit. If only the candidate passes, that supports proceeding with it to a bounded broader-data experiment, not a claim of unseen-word quality or proof across multiple seeds.
4. If both fail, preserve both results and stop this comparison. A lower reconstruction loss, sharper but wrong letters, or success on selected words is insufficient. No automatic continuation or broader pilot follows.

The original 69-distinct-unseen-word comparison remains deferred and unchanged. Training-set fitting is not compositional generalization or product readiness.

## Verification without optimization

- **207 default tests passed; four weight-update tests skipped. Lint passed.** Added tests cover migration, legacy/new checkpoint loading, padding invariance, gradient flow through the new inputs, and digest/training-identity validation.
- All 32 real training examples were checked in both target-width and predicted-width modes. Maximum starting logit difference and width-prediction difference were **exactly zero** on this CPU runtime.
- New coordinate-channel gradient norms were 0.13944, 0.11966 and 0.04781 in the checked batch. Parent checkpoint bytes and migrated model weights remained unchanged. This verifies learnable inputs, not better handwriting.
- Both manual configurations completed no-optimizer preflight. Their random/function-preserving starting previews are not new trained outputs.

Evidence: `artifacts/word-generator-detail-audit/migration.json`, `artifacts/word-generator-detail-control-preflight/preflight.json`, and `artifacts/word-generator-detail-coordinate-preflight/preflight.json`. Training and any quality improvement are still pending owner execution and review.
