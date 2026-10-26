# Implementation status - 26 October 2026

## Completed and verified

- Created the repository on `codex/ml-foundation`, Python 3.12 environment, pinned direct dependencies, exact CPU environment lockfile and CI checks.
- Implemented the 62-class conditional convolutional VAE, versioned label/preprocessing contracts, training CLI, checkpoint loading, deterministic glyph sampling and evaluation/review exports.
- Downloaded the official EMNIST archive and verified source integrity through torchvision; recorded SHA-256 hashes for the four ByClass source files.
- Prepared 628,142 training and 69,790 validation examples from 697,932 official training examples. Saved and reloaded the exact split with checksum checks.
- Visually inspected the upright image/label sanity grid. Some handwriting is naturally ambiguous across case or similar characters; the data is not a guarantee of distinguishable glyphs.
- Default test suite: **29 passed, 2 skipped**, including progress-bar checks. Skipped tests update weights and require the owner's explicit `--run-training` opt-in. Lint and dependency checks passed.
- Training and validation now show live per-epoch progress, completed/total batches, ETA, throughput, current/running loss and processed examples. Counters advance only after a successful batch; capped pilot runs use capped totals. Use `--no-progress` for plain epoch-summary logs. No training was started to verify this change.
- The owner manually completed the bounded smoke run: 2,048 training and 512 validation examples, one partial epoch. Train loss 463.726; validation loss 462.599 at different KL weights (0.2 / 1.0), so these two numbers are not directly comparable.
- Reloaded the owner's `artifacts/smoke/best.pt` without any training: checkpoint SHA-256 matches, all sampled outputs are finite, and identical seeds reproduce tensors on CPU. The checkpoint has 1,044,961 parameters.
- Prior and reconstruction grids remain gray/noisy after eight optimizer steps, as expected for this short run. Character correctness, readability and diversity have not passed evaluation. Official test examples have not been loaded for evaluation.

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

## Next manual training run

Use a longer bounded pilot to inspect learning progress and estimate steady-state CPU speed before choosing the full training budget:

```powershell
.\.venv\Scripts\python.exe -m ml.train --output artifacts/pilot-200 --device cpu --epochs 1 --max-train-batches 200 --max-validation-batches 20
```

This trains on 51,200 sampled examples (balanced sampling uses replacement) and validates on 5,120 examples. It starts from random weights and remains labelled pipeline-smoke-only because the epoch is bounded. It does not resume the eight-batch smoke checkpoint. The first smoke run spent 11.6 seconds in training and 0.28 seconds in validation; a rough linear estimate is about five minutes for 200 training batches or one hour for a full epoch, but startup, caching and sustained CPU behavior can change these estimates.

For a later full research baseline, review the pilot time and experiment budget first, then manually run:

```powershell
.\.venv\Scripts\python.exe -m ml.train --config configs/baseline.json --output artifacts/baseline --device cpu
```

Run outputs must be new or empty. If rerunning a smoke test, choose a new output directory instead of deleting or overwriting prior experiments. Interrupted optimizer-state resume is not implemented in this milestone.

## Still to implement or validate

Longer owner-run training; genuine class correctness/diversity evaluation; frozen visual reference and canonical passage; glyph metrics and page compositor; FastAPI service; Next.js interface; service benchmarking, release QA and deployment. The model feasibility gate comes before UI polish.
