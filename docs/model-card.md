# Inksy conditional character VAE - research model card

## Status

Architecture and training pipeline implemented. Any smoke checkpoint is explicitly marked as such; no release checkpoint or class-correctness score is claimed by this document. Evaluation results belong to the named run artifacts, not to generic architecture claims.

## Intended use and limitations

Generate isolated handwritten-style Latin alphanumerics for an educational handwriting studio. Not suitable for guaranteed faithful transcription, connected cursive, writer identification, signatures, impersonation or document authentication. Class conditioning is not a guarantee of the right glyph, and latent sampling is not a learned writer-style control.

## Model

Two stride-2 convolution blocks reduce 28x28 grayscale input to 7x7 features. A class embedding joins the encoder features, producing posterior mean/log-variance. A sampled latent vector and the same class embedding feed a decoder with two transposed-convolution blocks. Output logits convert to foreground intensities with sigmoid. All weights start randomly. No pretrained weights or external generation API are involved.

The default model uses 32 base channels, 16 label-embedding dimensions and 32 latent dimensions. The loss is pixel-summed BCE plus beta times dimension-summed KL, both averaged over the batch. KL warmup applies only to training; validation uses fixed full beta and fixed noise.

## Data

EMNIST ByClass: 62 classes, 28x28 grayscale, digits plus distinct uppercase/lowercase labels. The raw image is transposed to upright and normalized to [0,1], with bright ink on dark background internally. Rendering diagnostic grids inverts this to dark ink on white paper. Official training data supplies deterministic stratified training/validation subsets. Dataset provenance, checksums, mapping, split identity and class counts are saved in the preparation manifest.

Original dataset description: https://www.nist.gov/itl/products-and-services/emnist-dataset

Applicable source terms, attribution and redistribution conditions must be reviewed before publication. Dataset files and model weights are ignored in Git.

## Evaluation and reproducibility

Report requested-class correctness with a validated independent classifier and/or a documented human review, alongside diversity, full-page readability and performance. Reconstruction loss alone does not establish correctness. The evaluation CLI deliberately leaves accuracy null and exports review sheets for real annotation.

The final test set is not used for model selection. Numerical release targets are provisional until frozen after the baseline. Seeds, data split, preprocessing, training config, environment and checkpoint identity are recorded. A pinned CPU environment is the reproducibility reference; cross-device/version bitwise equality is not promised.
