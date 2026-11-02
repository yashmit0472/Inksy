# Local word-and-paragraph preview

The local compositor renders typed text with the owner's latest trained generator. This stage performs inference only; it does not train a model, repair wrong character identities, or add connected cursive.

## Open the generated example

The example is in `artifacts/paragraph-preview/`:

- `preview.html`: typed text beside the generated page, display zoom and download links.
- `page-001.png`: the actual rendered page at 1240 x 1754 pixels.
- `pages.zip`: the same PNG bytes packaged for download.
- `input.txt`: the source passage.
- `manifest.json`: seed, model hash, layout settings, page hashes, coverage, per-character placements and limitations.

Open `preview.html` in your browser. Keep its folder together so the relative PNG and ZIP links continue to work. The report is local and uses no external scripts or fonts. The source text is read-only; changing text or layout requires a new CLI render. Zoom changes the display only.

## Render your own text

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m ml.preview --text "Hello, this is Inksy." --output artifacts/hello-preview
```

For a UTF-8 text file:

```powershell
.\.venv\Scripts\python.exe -m ml.preview --text-file my-passage.txt --normalize --output artifacts/my-preview
```

`--normalize` explicitly converts smart quotes/dashes, tabs and Windows line endings according to the character contract. The report shows both original and normalized text when they differ. Other unsupported symbols are rejected with their positions; they are not silently dropped. Without this flag, text must already contain only supported characters and LF line breaks.

The default checkpoint is `artifacts/refined-continue-5/best.pt`. Select a different trusted project checkpoint with `--checkpoint`. Every letter/digit occurrence is sampled separately using a request-local seed; repeated letters can vary. Rendering uses bounded inference batches rather than decoding an entire long passage at once.

## Layout controls

```powershell
.\.venv\Scripts\python.exe -m ml.preview --text "Hello, this is Inksy." --glyph-size 28 --character-spacing 3 --word-spacing 2 --line-spacing 14 --margin 70 --paper plain --seed 42 --output artifacts/hello-layout
```

| Option | Default | Meaning |
| --- | ---: | --- |
| `--width`, `--height` | 1240, 1754 | Page dimensions in pixels; each side is limited to 320-2048. |
| `--margin` | 80 | Writing-area margin on every side. |
| `--glyph-size` | 32 | Nominal capital height; supported range 12-64 pixels. |
| `--character-spacing` | 2 | Additional gap after visible characters. |
| `--word-spacing` | 0 | Additional width per space. |
| `--line-spacing` | 12 | Additional gap beyond cap-height/descender allowance. |
| `--paper` | ruled | `plain`, `ruled` or `grid`. |
| `--seed` | 42 | Reproducible sampling within the same pinned model/runtime/settings. |
| `--no-progress` | off | Hide inference progress bars. |

Choose a new output directory for each experiment. Existing directories are not overwritten. The input is limited to 3,000 rendered characters; output is limited to three pages and seven million total pixels. Unbreakable words are split across lines only when they cannot fit. Explicit spaces, paragraph breaks and reading order are preserved; trailing whitespace alone does not add empty pages.

## How the page is composed

The generator supplies 28x28 grayscale foreground masks. The compositor finds padded ink bounds, preserves aspect ratios, uses proportional advances and applies heuristic cap/x-height/ascender/descender metrics. It aligns glyphs to baselines, wraps words, paginates and alpha-composites ink over paper. Cropping uses a low-intensity threshold only to find bounds; retained mask pixels keep their grayscale values. Enlargement does not create additional learned detail.

Punctuation is drawn with procedural paths and is disclosed separately from generated alphanumerics. The original typed text uses normal browser text only in the reference panel. No font fallback supplies model characters on the generated page.

This preview exposes actual model errors: incorrect letters, unusual case shapes, inconsistent writing style and softness can remain visible. Layout metrics are heuristics and should be calibrated using representative paragraphs before a product release.

## Verification

The non-training test suite passed **77 tests**, with two weight-update tests skipped. It covers proportional widths, descenders, wrapping, source-index coverage, whitespace, pagination, resource limits, missing glyphs, seeded sampling, escaped report text, Windows line-ending preservation and PNG/ZIP byte fidelity.

The actual latest-checkpoint example contains 260 generated alphanumeric glyphs and 13 procedural punctuation marks on one page. A repeated render with the same seed produced identical PNG bytes; ZIP entries matched the exported PNGs. A separate 1,000-character real-checkpoint render wrapped onto two pages, with every visible input character accounted for. The checkpoint hash remained unchanged, and no training was performed.

The generated example and second page of the long passage were visually inspected as PNGs. Automated browser interaction verification was unavailable because the browser tool blocks local `file://` navigation; no alternate route was used to bypass that restriction.
