"""Train a conditional VAE from scratch; keep final test data untouched."""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from ml.characters import ALPHABET
from ml.checkpoint import save_checkpoint
from ml.dataset import load_partitions
from ml.model import ConditionalVAE, ModelConfig, vae_loss
from ml.utils import environment_info, resolve_device, seed_everything, sha256_file, write_json
from ml.visualize import save_grid


@dataclass
class TrainingConfig:
    data_root: str = "data"
    prepared_dir: str = "artifacts/data"
    output_dir: str = "artifacts/baseline"
    seed: int = 42
    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 0.001
    latent_dim: int = 32
    label_dim: int = 16
    base_channels: int = 32
    beta: float = 1.0
    kl_warmup_epochs: int = 5
    patience: int = 5
    num_workers: int = 0
    cpu_threads: int = 4
    balanced_sampling: bool = True
    device: str = "auto"
    max_train_batches: int | None = None
    max_validation_batches: int | None = None
    progress: bool = True

    def __post_init__(self):
        for key in (
            "epochs",
            "batch_size",
            "latent_dim",
            "label_dim",
            "base_channels",
            "patience",
            "cpu_threads",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive.")
        if (
            self.learning_rate <= 0
            or self.beta < 0
            or self.num_workers < 0
            or self.kl_warmup_epochs < 0
        ):
            raise ValueError("Invalid training hyperparameters.")
        for value in (self.max_train_batches, self.max_validation_batches):
            if value is not None and value < 1:
                raise ValueError("Batch limits must be positive.")


def run_epoch(
    model,
    loader,
    device,
    beta,
    *,
    optimizer=None,
    max_batches=None,
    seed=43,
    show_progress=True,
    description=None,
):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "reconstruction", "kl")}
    examples = 0
    # Validation uses fixed noise each epoch, independently of the global training RNG.
    generator = torch.Generator(device=device).manual_seed(seed)
    batch_count = len(loader) if max_batches is None else min(len(loader), max_batches)
    start = time.perf_counter()
    with (
        torch.set_grad_enabled(training),
        tqdm(
            total=batch_count,
            desc=description or ("Train" if training else "Validation"),
            unit="batch",
            dynamic_ncols=True,
            ascii=True,
            mininterval=0.25,
            miniters=1,
            disable=not show_progress,
        ) as progress,
    ):
        for images, labels in islice(loader, batch_count):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                logits, mu, logvar = model(images, labels)
            else:
                mu, logvar = model.encode(images, labels)
                logits = model.decode(model.reparameterize(mu, logvar, generator), labels)
            losses = vae_loss(logits, images, mu, logvar, beta)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError("Non-finite loss. Inspect data/model before continuing.")
            if training:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                optimizer.step()
            count = len(images)
            examples += count
            for key in totals:
                totals[key] += losses[key].item() * count
            progress.set_postfix(
                loss=f"{losses['loss'].item():.3f}",
                avg=f"{totals['loss'] / examples:.3f}",
                examples=f"{examples:,}",
                refresh=False,
            )
            # Advance only after a successful batch (including the optimizer step for training).
            progress.update(1)
    if not examples:
        raise ValueError("The data loader contained no examples.")
    return {
        **{key: value / examples for key, value in totals.items()},
        "examples": examples,
        "seconds": time.perf_counter() - start,
    }


@torch.inference_mode()
def save_samples(model, validation, device, output: Path, epoch: int, seed: int):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    labels = torch.arange(len(ALPHABET), device=device).repeat_interleave(4)
    latent = torch.randn((len(labels), model.config.latent_dim), device=device, generator=generator)
    images = model.decode(latent, labels).sigmoid()
    names = [ALPHABET[int(label)] for label in labels.cpu()]
    save_grid(images, output / "samples" / f"prior-epoch-{epoch:03d}.png", names, columns=16)
    pairs, names = [], []
    # A deterministic cross-section of the validation partition, not cherry-picked outputs.
    for i in np.linspace(0, len(validation) - 1, 24, dtype=int):
        image, label = validation[int(i)]
        mu, _ = model.encode(image.unsqueeze(0).to(device), torch.tensor([label], device=device))
        reconstruction = model.decode(mu, torch.tensor([label], device=device)).sigmoid()[0]
        pairs.extend([image, reconstruction.cpu()])
        names.extend([f"{ALPHABET[label]} real", f"{ALPHABET[label]} recon"])
    save_grid(
        torch.stack(pairs),
        output / "samples" / f"reconstruction-epoch-{epoch:03d}.png",
        names,
        columns=12,
    )


def train(config: TrainingConfig):
    seed_everything(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = resolve_device(config.device)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run output {output} is not empty; choose a new output directory.")
    print(
        "Preparing data: checking source hashes and loading train/validation splits...", flush=True
    )
    dataset, training, validation, manifest = load_partitions(
        Path(config.data_root), Path(config.prepared_dir)
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", asdict(config))
    environment = environment_info(device)
    write_json(output / "environment.json", environment)
    write_json(output / "data-manifest.json", manifest)
    print(
        f"Device: {device}; GPU: {environment['gpu']}; "
        f"train={len(training):,}; validation={len(validation):,}",
        flush=True,
    )
    generator = torch.Generator().manual_seed(config.seed)
    sampler = None
    if config.balanced_sampling:
        targets = dataset.targets[training.indices]
        counts = torch.bincount(targets, minlength=len(ALPHABET)).double()
        if (counts == 0).any():
            raise ValueError("A supported class is missing from training.")
        sampler = WeightedRandomSampler(
            counts.reciprocal()[targets], len(training), replacement=True, generator=generator
        )
    loader = DataLoader(
        training,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        validation,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed + 1),
    )
    model = ConditionalVAE(
        ModelConfig(config.latent_dim, config.label_dim, config.base_channels)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    best, stale, history = math.inf, 0, []
    smoke = config.max_train_batches is not None or config.max_validation_batches is not None
    for epoch in range(1, config.epochs + 1):
        beta = config.beta * min(1, epoch / max(1, config.kl_warmup_epochs))
        train_metrics = run_epoch(
            model,
            loader,
            device,
            beta,
            optimizer=optimizer,
            max_batches=config.max_train_batches,
            show_progress=config.progress,
            description=f"Train {epoch}/{config.epochs}",
        )
        # Always compare validation at the final beta, including during KL warmup.
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            config.beta,
            max_batches=config.max_validation_batches,
            seed=config.seed + 1,
            show_progress=config.progress,
            description=f"Validation {epoch}/{config.epochs}",
        )
        row = {
            "epoch": epoch,
            "train_beta": beta,
            "validation_beta": config.beta,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(row)
        write_json(output / "history.json", history)
        print(
            f"epoch={epoch} train={train_metrics['loss']:.3f} validation={val_metrics['loss']:.3f} "
            f"kl={val_metrics['kl']:.3f} "
            f"seconds={train_metrics['seconds'] + val_metrics['seconds']:.1f}",
            flush=True,
        )
        metadata = {
            "epoch": epoch,
            "training_config": asdict(config),
            "environment": environment,
            "split_sha256": manifest["split_sha256"],
            "validation": val_metrics,
            "quality_status": "pipeline-smoke-only" if smoke else "unreviewed-research-checkpoint",
        }
        print("Saving checkpoints and sample grids...", flush=True)
        save_checkpoint(output / "last.pt", model, **metadata)
        if val_metrics["loss"] < best:
            best, stale = val_metrics["loss"], 0
            save_checkpoint(output / "best.pt", model, **metadata)
        else:
            stale += 1
        save_samples(model, validation, device, output, epoch, config.seed)
        if stale >= config.patience and epoch >= config.kl_warmup_epochs:
            print("Early stopping: validation did not improve.", flush=True)
            break
    write_json(
        output / "summary.json",
        {
            "best_validation_loss": best,
            "epochs_completed": len(history),
            "quality_status": metadata["quality_status"],
            "best_checkpoint_sha256": sha256_file(output / "best.pt"),
            "test_set_evaluated": False,
            "warning": (
                "Loss is not character accuracy or proof of usable handwriting. "
                "Inspect prior grids and evaluate class correctness."
            ),
        },
    )
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--output", type=str)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide live bars; keep epoch summaries for log files.",
    )
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("device", "epochs", "max_train_batches", "max_validation_batches"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    if args.output:
        values["output_dir"] = args.output
    if args.no_progress:
        values["progress"] = False
    train(TrainingConfig(**values))


if __name__ == "__main__":
    main()
