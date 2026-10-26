# Project instructions

- The project owner wants to run all model training manually. Do not launch `ml.train`, optimizer-based experiments, or `pytest --run-training`. Prepare the code and exact commands, and tell the owner when a training run is needed.
- Data preparation, inspection, linting, and the default non-training test suite are allowed. Default tests must not update model weights; keep weight-update tests explicitly opt-in.
- Train the handwriting generator from scratch. Do not use pretrained handwriting weights, an external AI generation API, or a font fallback that is presented as generated handwriting.
- Keep datasets, checkpoints, secrets, and generated user content out of Git. Record source provenance, data partitions, settings, seeds, and checkpoint compatibility.
- Separate code correctness, smoke-run execution, reconstruction loss, and handwriting quality in reports. Do not claim an evaluated model until real requested-class and visual evaluations exist.
