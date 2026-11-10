# Clearer targets and unseen-word experiment — 5 November 2026

**8 November outcome.** The writer-conditioned model and bounded trainer are implemented. The run reached 2,178 updates and produced blurred unseen words despite a lower validation objective. The separate [print model](print-handwriting.md) delivers the readable lowercase output. The original design below records the experiment plan.

## Completed inspection

Reviewed eight deterministic, distinct training words from each of the 12 writers with the most eligible lowercase training images (96 references), then expanded writers 150 and 153 to 32 references each. Images were loaded through the existing hash-verifying IAM loader. These are real IAM images, not generated improvements. No validation/test images were used to select styles, and no model was trained.

Writer **150** is the preferred target for legible print-like handwriting. Writer **153** is the cursive alternative. Inspection of the larger sheets supports this preference, while showing imperfect examples too; this is a qualitative shortlist, not a measured writer-legibility ranking or a claim that these are the best writers in all IAM. The initial search covered only 12 well-represented writers.

The old writer 000 was selected for volume. That choice was poorly aligned with the owner's desired appearance. Changing targets can improve the intended style, but does not establish that the model can learn it or spell unseen words correctly.

## Prepared data scope

Prepared an index-only proposal for writers 150, 153, 154, 152, 670 and 338, selected using training counts and inspected training references. Labels remain exact lowercase ASCII words of length 2–10. Existing assignments are unchanged.

| Partition | Images | Distinct words |
|---|---:|---:|
| Train | 2,337 | 562 |
| Seen validation | 246 | 135 |
| Content validation | 265 | 65 |

Writer 150 alone has 389 eligible training images and 216 distinct words; writer 153 has 423 images and 230 words. A single clearer writer supplies substantially less data than writer 000. Combining writers requires explicit style conditioning: feeding all styles to the current deterministic single-writer regression model would leave different target appearances under the same text condition.

`artifacts/writer-target-audit/proposed-pool.json` records exact IDs, writer choices, source-record digest and counts. All 65 content-validation word strings are absent, case-insensitively, from the entire original IAM training partition. These are development holdouts, not a fresh independent benchmark after all prior project experimentation. Reserved official writer-validation and test pixels remain unopened.

Inspection artifacts: `candidates-1.png` through `candidates-3.png`, `writer-150-32.png`, `writer-153-32.png`, and `audit.json` in the same ignored directory. Seeded SHA-256 selection prevents choosing only attractive examples within each inspected writer. No dataset images were copied into Git.

## Revised implementation plan

1. Add an explicit writer embedding to the shared text encoder/image decoder and width predictor. Retain position inputs, masked canvases and text-only inference with predicted widths. Writer ID specifies a known learned style; it does not imply personal-handwriting imitation from a reference image.
2. Introduce a separate multi-writer checkpoint contract with an exact writer map, data digest and scope. Initialize the new experiment randomly. The rejected writer-000 checkpoint remains a comparison artifact, not a release candidate.
3. Train over the full proposed 2,337-image pool with deterministic coverage. Normalize sampling across the six writers, keep all training instances available, and record the sampling schedule. Inspect rare-letter coverage before declaring supported characters. Do not introduce fonts, external weights or a recognition loss from the weak reader.
4. Implement validation-based selection using the 246 seen-validation images, progress/sample exports and a hard owner-run budget established by a no-optimizer CPU preflight. Tiny-set fitting remains a pipeline diagnostic; it must not become the reported result or an indefinite training prerequisite.
5. Freeze evaluation prompts before the pilot. Generate all 65 distinct content-held-out words in target style 150 at the selected checkpoint, without target images or target widths. Report text accuracy by distinct word, character errors, length/rare-letter failures and visual defects. Include comparison output from the retained old word generator and real writer-150 training references shown separately as style examples. Style differences limit interpretation of that baseline comparison; do not attribute all preference differences to architectural quality.
6. Judge unseen output through a blinded transcription review, not the training-loss reader. Provisional pilot goal: at least 52/65 exact words (80%) plus acceptable stroke clarity and no clipping. This is a new generalization goal, not a retroactive pass or relaxation of the rejected 32-word experiment. Product release criteria stay unchanged. Report actual counts and all failures, and do not reuse this content holdout for endless tuning.

The 562-word training vocabulary is still small and may not support strong compositional generalization. Thin strokes and ambiguous labels in some real examples also remain risks. If this scope is inadequate, expand training coverage based on training-only evidence instead of merely adding updates. A style-conditioned deterministic regressor may still blur variation; changing the writer does not remove that modeling limitation.

## Current boundary

The inspection, reference sheets, proposed partition index and revised design are complete. The multi-writer model, broader trainer and evaluation workflow above are **not implemented yet**, so there is no new training command to run. No model-quality improvement is claimed. Next engineering work is that implementation and non-training preflight; the owner retains all optimization execution.
