# Inksy

Inksy turns typed text into handwritten pages you can preview and download as PNGs. The standard style supports lowercase and uppercase English letters, digits, and common punctuation.

**[Try Inksy on Vercel](https://inksy-handwriting.vercel.app/)**

## What you can do

- Paste text, preview the generated pages, and download each page as a PNG.
- In the local studio, train a personal style from a photo of a printable worksheet or by drawing in the browser. The worksheet collects two samples for each of 62 letters and digits. Review the samples, watch training progress, and save the style for reuse.

## How it works

- **PyTorch:** A convolutional decoder trained from scratch learns glyph shapes from reviewed IAM and EMNIST samples. The standard checkpoint contains 144 learned glyph variants across 62 alphanumeric characters.
- **Python and Pillow:** The compositor chooses variants, applies small baseline shifts, wraps text, and renders pages. It draws punctuation separately from the learned letters and digits.
- **Browser and API:** The studio uses HTML, CSS, and JavaScript. The local Python server handles photo alignment, crop review, training, and rendering. The public Vercel demo uses a Python serverless function with precomputed glyphs from the trained model, so it can generate PNGs without loading PyTorch on every request.

A blind AI transcription review of the standard style matched **240/240 isolated prompts** (200 held-out lowercase words, 20 mixed-case strings, and 20 digit strings), with zero character edits across 1,622 characters. This was a development check by one AI reviewer. Independent human validation is pending. See [model and evaluation notes](docs/print-handwriting.md).

## Run locally

Install Python 3.12, then run these commands from the repo root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m rendering.studio --checkpoint artifacts/print-handwriting-v3-balanced/model.pt
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The studio needs the standard checkpoint to start. Datasets, checkpoints, and personal styles are kept outside Git. If you only want to try rendering, use the live demo.

Run the default tests with `.\.venv\Scripts\python.exe -m pytest`. Tests that update model weights require an explicit `--run-training` opt-in.

For more detail, see the [project status and training experiments](docs/status.md) or the [deployment guide](docs/deployment.md).
