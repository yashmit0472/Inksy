# Project instructions

- The agent runs project commands and model training directly as part of the requested work. Tell the owner when training starts and finishes. During longer runs, send a brief progress update about every 5–7 minutes with the latest completed steps or epochs when available. Distinguish training from setup, validation, and export.
- Data preparation, inspection, linting, and the default non-training test suite are allowed. Default tests must not update model weights; keep weight-update tests explicitly opt-in.
- Train the handwriting generator from scratch. Do not use pretrained handwriting weights, an external AI generation API, or a font fallback that is presented as generated handwriting.
- Keep datasets, checkpoints, secrets, and generated user content out of Git. Record source provenance, data partitions, settings, seeds, and checkpoint compatibility.
- Separate code correctness, smoke-run execution, reconstruction loss, and handwriting quality in reports. Do not claim an evaluated model until real requested-class and visual evaluations exist.
