# Inksy

A handwriting-generation project built around a **conditional convolutional VAE trained from scratch** in PyTorch. The model generates isolated character images; a later page compositor and web application will turn them into downloadable pages. No pretrained model or external AI generation API is used.

## Current milestone

The offline ML foundation is implemented: EMNIST ByClass preparation, deterministic train/validation splits, image/label sanity grids, conditional VAE, training/checkpoints, seeded glyph sampling, validation diagnostics and tests. No usable-handwriting claim is made until the generated samples pass the documented review. The compositor, API and frontend follow the model feasibility gate.

The revised blueprint is in `output/pdf/Inksy_End_to_End_Project_Plan_v1.1.pdf`.

**Training is manual.** The project owner will launch all model-training runs. The commands below are instructions for the owner; setup, data preparation and the default test suite do not train a model. Tests that update weights are skipped unless the owner explicitly supplies `--run-training`.

See `docs/status.md` for verified local results and the next manual training command. The exact tested CPU dependency set is in `requirements-lock.txt`; install it with `python -m pip install -r requirements-lock.txt`, then install this project with `python -m pip install -e . --no-deps`.

## Setup (Windows PowerShell)

The existing `.venv` uses Python 3.12. For a fresh machine, install Python 3.11-3.13 first; the globally installed Python 3.8 on this machine is too old. With Python 3.12 registered with the Windows launcher, create an isolated environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The CPU build is the working baseline for this workspace. The CUDA build was attempted, but Windows reported error 1455 (insufficient paging-file memory) while importing its DLLs, before any training. Windows settings were not changed. Run all commands from the repository root. Dependencies stay in the project environment; they do not modify global Python.

The local GTX 1650 has 4 GiB VRAM. If the owner resolves the Windows host-memory/paging requirement and wants to train on it, manually replace the CPU packages with the official CUDA 12.6 builds:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Verify this import succeeds before launching GPU training. Use the CPU index URL with the same replacement command to return to the working CPU environment.

## 1. Prepare and inspect data

```powershell
.\.venv\Scripts\python.exe -m ml.dataset --download
```

This downloads the official EMNIST archive (approximately 536 MiB compressed), including other splits supplied in the archive. Only **ByClass training data** is loaded for preparation. Allow several GiB of free disk space for the archive, extracted data and CUDA environment.

Preparation creates `artifacts/data/manifest.json`, checksums, `split.npz`, and `orientation-and-labels.png`. Inspect the grid before training: characters must be upright, the class labels must match, ink must be foreground, and uppercase/lowercase must remain separate. Prepared output is immutable by default; use a new `--output` path for a different split.

The default split uses 90% of official training examples for training and 10% for validation, stratified by class with seed 42. Official test examples are reserved for final evaluation. No dataset or generated user content belongs in Git.

## 2. Run a bounded real-data smoke test

```powershell
.\.venv\Scripts\python.exe -m ml.train --config configs/baseline.json --output artifacts/smoke --epochs 1 --max-train-batches 8 --max-validation-batches 2
```

This verifies the training pipeline on real data. It is labelled **pipeline-smoke-only** and is not evidence of readable handwriting. Unit tests use synthetic fixtures solely to verify implementation correctness.

## 3. Train a research baseline

```powershell
.\.venv\Scripts\python.exe -m ml.train --config configs/baseline.json
```

The configuration proposes 20 epochs with early stopping, a 32-dimensional latent vector, balanced sampling and KL warmup. Device `auto` uses CUDA when available. Pilot-run timing should determine the final epoch/experiment budget before a long run. Use `--device cpu` for the portable deterministic baseline. Run output must be new or empty to avoid destroying experiments.

Each run records configuration, environment, data identity, per-epoch train/validation losses, fixed-seed prior grids, reconstruction pairs and versioned `best.pt` / `last.pt` checkpoints. Loss uses reconstruction BCE summed over pixels plus weighted KL summed over latent dimensions, both averaged over examples. Validation always uses the final KL weight and fixed sampling noise for comparable checkpoint selection. Checkpoints restore inference weights; interrupted optimizer-state resume is not implemented yet.

### Live training progress

Training and validation display separate progress bars automatically. Each shows the epoch number, completed/total batches, percentage, elapsed time, estimated time remaining, speed, current batch loss (`loss`), running average loss (`avg`), and processed examples. Bounded runs use their actual batch limit (for example, `200/200`), not the size of the full dataset. The training bar advances after each successful optimizer step; the validation bar advances after each inference batch. ETA appears after enough timing data is available and may change during a run.

Data loading and checkpoint/sample saving also announce their stages. The bars measure work completed, not handwriting quality. Add `--no-progress` when redirecting output to a log file; epoch summaries remain available. New bars apply to the next manually launched run; an already-running Python process keeps its previously loaded code.

## 4. Sample and evaluate

```powershell
.\.venv\Scripts\python.exe -m ml.generate --checkpoint artifacts/baseline/best.pt --characters AaBb09 --variations 8 --output artifacts/generated/glyphs.png
.\.venv\Scripts\python.exe -m ml.evaluate --checkpoint artifacts/baseline/best.pt --output artifacts/evaluation
```

Sampling uses only the trained decoder, requested labels and a local seeded generator. Same seed/model/runtime/settings reproduce pixels; CPU and CUDA results need not be identical. Checkpoint loading checks the alphabet, preprocessing and schema, and uses PyTorch's weights-only loader. Load only trusted project artifacts.

Evaluation defaults to validation data. It reports reconstruction diagnostics by class and exports sample sheets plus a human-review CSV. **Class accuracy remains unreported until genuine human or independently validated classifier evaluation exists.** Opening the official final test data requires `--partition test --final-test` after model selection is frozen.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp .cache/pytest
.\.venv\Scripts\python.exe -m ruff check .
```

The default tests cover orientation, character identity, normalization, split integrity, tensor shapes, loss scaling, gradient connectivity, checkpoint compatibility and RNG isolation. They do not update model weights. Tests of the optimizer/training loop and fixed-batch learning require a separate, explicit manual invocation:

```powershell
.\.venv\Scripts\python.exe -m pytest --run-training --basetemp .cache/pytest-training
```

A passing test suite does not establish model quality.

## Product boundaries

- Model alphabet: `0-9`, `A-Z`, `a-z` (62 classes).
- Punctuation in the planned MVP is procedural and explicitly disclosed. Spaces/newlines are layout; unsupported symbols are rejected, never silently dropped.
- Smart punctuation normalization is explicit and must be shown before generation.
- The first release assembles isolated glyphs; it does not promise connected cursive, writer imitation or print-quality fidelity.
- Accounts, saved history and persistent user content are outside the first release.

See `docs/decisions.md`, `docs/model-card.md` and `docs/acceptance.md` for the baseline and remaining gates.

## Data and references

EMNIST derives from NIST Special Database 19. Record the original terms and attribution requirements before redistribution or publication; a local data download is not a commercial-use clearance. The source manifest records source-file checksums. Original dataset: https://www.nist.gov/itl/products-and-services/emnist-dataset

Cohen, G., Afshar, S., Tapson, J., and van Schaik, A. (2017), *EMNIST: an extension of MNIST to handwritten letters*: https://arxiv.org/abs/1702.05373

PyTorch installation versions: https://pytorch.org/get-started/previous-versions/
#   I n k s y  
 