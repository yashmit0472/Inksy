# Readable print handwriting

## Train your own print style

Start the local studio from the Inksy folder:

```powershell
.\.venv\Scripts\python.exe -m rendering.studio --port 8766
```

Open `http://127.0.0.1:8766/train`. Name your style and add two samples for each lowercase letter, capital, and digit. Draw in the browser or open the printable worksheet and write inside its labeled boxes with a dark pen. Upload a JPEG or PNG of the whole sheet and select the centers of the four black squares in the order shown. Import the samples, review the labeled crops, and draw any missing samples. The photo is processed locally and the crop draft stays in the current browser.

Click **Train my style** to start the CPU run and see its progress. Then click **Try this style** to render new text. Each new style trains from scratch on 124 samples and saves its own checkpoint and training record under `artifacts/personal-styles/`. The renderer supplies punctuation. Styles trained with the earlier lowercase worksheet, including Reviewed IAM, still render using their original alphabet.

## Mixed-case model

The mixed print decoder learned 144 forms across lowercase and uppercase English letters and the ten digits. Its training targets combine 72 reviewed IAM lowercase crops and 72 reviewed EMNIST crops from the pinned project training split. A second 2,100-update run adjusted the normalization for the heaviest isolated strokes. The checkpoint is `artifacts/print-handwriting-v3-balanced/model.pt`.

The renderer draws 11 punctuation marks with the shared compositor and aligns them to the learned glyphs. It also normalizes smart quotes and dashes using the shared text rules. The sample page is `artifacts/mixed-print-balanced-review/mixed-page.png`. It exercises all ten digits and 11 punctuation marks.

The final agent visual check read 26 of 26 capitals, 10 of 10 digits and 11 of 11 marks from category-specific unlabeled grids. This check used one form per character. The full sample page was inspected and reproduced from its seed. The v2 checkpoint also replayed its earlier lowercase sample pixel for pixel. Independent human review remains pending. Source indices, split hashes and training settings are stored in the mixed run's `run.json`.

The current studio checkpoint scored **65/65 exact word matches** and **0 character edits across 435 reference characters** in a separate blind visual transcription by one AI reviewer. The seeded sample came from distinct 3–9 letter lowercase IAM development prompts absent from the training vocabulary. The studio rendered each word in isolation. The sample covered 23 of 26 lowercase letters. These are development review results for the standard style. The frozen key, transcriptions, checkpoint hash, and per-word scores are in `artifacts/studio-readability-review/`.

The expanded blind review scored **200/200 held-out lowercase words**, **20/20 synthetic mixed-case strings**, and **20/20 synthetic digit strings**. The single AI reviewer made zero character edits across 1,622 characters. The lowercase prompts came from the IAM development partition and excluded both the training vocabulary and the earlier 65 prompts. The benchmark covers all 62 alphanumeric characters. The results measure isolated prompts rendered by the standard studio style. Human transcription remains pending. The blinded grids, blank transcription form, and per-prompt report are in `artifacts/studio-readability-240/`.

To score a completed human transcription CSV, use the frozen benchmark directory and pass the completed file. Open only the blinded grids while transcribing.

```powershell
.\.venv\Scripts\python.exe -m ml.review_studio artifacts/studio-readability-240 --answers path/to/completed-transcriptions.csv --reviewer human
```

## Lowercase variant model

The lowercase print decoder trained from scratch for 1,200 CPU updates on 72 reviewed IAM training crops. Most letters have three learned forms. The renderer varies the choice on repeated letters, offsets their baselines slightly and tightens letter spacing. Ink targets reinforce faint stroke edges. The output is reproducible from the text and seed. Its input scope is lowercase English and whitespace.

In a first pass through the unlabeled grid I read 24 of 24 fresh IAM development words correctly. These words were excluded from the original IAM training vocabulary and the previous 65-word review. The single agent reading is recorded in `artifacts/print-v2-review/agent-review.json`. The paired paragraph shows fuller strokes and more varied repeated letters. Independent human acceptance remains pending.

The v2 checkpoint is at `artifacts/print-handwriting-v2/model.pt`. The comparison is `artifacts/print-v2-review/before-after.png`. Run provenance and the reviewed crop list are recorded in the run's `run.json`.

## Original model

The print renderer uses a convolutional decoder trained from scratch on 26 visually reviewed lowercase letter crops from IAM training scans. Writer 150 supplies 21 of the crops. The remaining five letters use clear crops from writers 333, 160, 109, 064 and 158. Source word IDs, crop bounds, writer IDs, dataset hash and training settings are saved with the run.

The model learns one canonical image per letter. The renderer assembles those learned images in text order and wraps words onto lines. Repeated letters share a shape. The current input contract is lowercase English letters and whitespace, with a 600-character limit.

## Render text

From the Inksy folder, use the trained local checkpoint.

```powershell
.\.venv\Scripts\python.exe -m ml.print_handwriting render --checkpoint artifacts/print-handwriting-v2/model.pt --text "the quick brown fox jumps over the lazy dog" --output artifacts/my-handwriting.png
```

Use a fresh output filename for each render. The included example `artifacts/print-handwriting-v1/handwriting-preview.png` covers all 26 letters.

## Readability evidence

The decoder completed 650 CPU updates. I inspected its full alphabet and read the 65 distinct content-held-out words from unlabeled image grids before checking the key. The first pass matched 64 of 65 prompts. I read sample 039 as `s` in the grid. Its individual PNG visibly reads `his`. The first-pass score is preserved. This is a single agent visual review. Human acceptance is pending.

These words were absent from the original IAM training partition. The evaluation covers lowercase print composition on a development set. Official reserved test images remain unopened.

Local results are in `artifacts/print-handwriting-v1-review/`. The `agent-review.json` file contains the per-word readings and score. The print checkpoint and training provenance are under `artifacts/print-handwriting-v1/`.

## Reproduce the fit

```powershell
.\.venv\Scripts\python.exe -m ml.print_handwriting train --selection artifacts/print-glyph-audit/selected.json --output artifacts/print-handwriting-reproduction --updates 650
```

The selection file contains the reviewed crop references. It requires the prepared IAM training data. The six-writer whole-word experiment uses its own trainer and checkpoint format.
