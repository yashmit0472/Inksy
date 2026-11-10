# Readability review — 5 November 2026

The next action is to review the position-aware candidate's 32 training words before any broader training. A portable offline form is now available at `artifacts/word-generator-detail-coordinate/blind-review.html`.

## Complete the review

1. Open the HTML file in a browser. It includes all 32 images, so no server or network access is required. Ideally use a reader who has not seen the labeled examples. Do not inspect the answer key during review.
2. Type the word visible in each image, or check **I cannot read this word**. Separately mark whether its strokes are visually acceptable. Missing, smeared or clipped strokes should be flagged. There are no preselected answers.
3. When all 32 samples are complete, click **Download completed review**. The expected filename for this candidate is `inksy-word-review-c9734bb7f464.json`. Keep the downloaded file and tell the agent it is ready to score.

The page stores a draft in the browser when local storage is available. If storage is unavailable, it displays a message to keep the tab open until downloading. The downloaded JSON is the portable submitted record. Opening the page does not submit answers or update project files. The answer key and requested text are not included in the page, its script data, alt text or download. The original source images are embedded without pixel modifications.

## Score the saved answers

Scoring performs no training or OCR. The agent can do it after the owner reports the file saved. For manual scoring, adjust the downloaded file path if needed:

```powershell
.\.venv\Scripts\python.exe -m ml.review_word_generator --review artifacts/word-generator-detail-coordinate/best-review --answers "C:\Users\yashm\Downloads\inksy-word-review-c9734bb7f464.json"
```

The scorer checks the submitted checkpoint identity, all 32 unique sample IDs, image digests, answer completeness and existing answer-key/source-image integrity. It records the submitted file's path and hash in `best-review/human-review.json`. The original CSV-based workflow remains supported when `--answers` is omitted.

The gate remains **at least 31/32 exact words and every image visually acceptable**. Passing supports preparing the wider-data generalization experiment; it does not establish new-word generation quality. If the review fails, inspect the mistakes against the real IAM targets before choosing further work. Do not automatically extend training.

## Verification and limitations

The page builder verified all 32 saved image hashes. The default suite passed **211 tests**, with four weight-update tests skipped; lint passed. New tests cover hidden-answer exclusion, source-integrity checks, checkpoint/image-bound JSON scoring, incomplete-answer blocking, draft restoration and export/scoring round-trip using synthetic unreadable responses. The form logic was exercised with a Node DOM harness; this was not a live browser visual/render test. No real reviewer responses were generated or scored by these tests.

No model training was launched, no model weights changed, and no readiness gate was marked passed. Human blinding depends on the reviewer's behavior; the software cannot certify that they have never seen the prompts.
