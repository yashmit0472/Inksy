"""Owner-run, bounded cGAN pilot on the existing EMNIST training partition."""

import argparse
import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from ml.characters import ALPHABET
from ml.checkpoint import save_checkpoint
from ml.dataset import load_partitions
from ml.gan import (
    ConditionalGenerator,
    GANModelConfig,
    ProjectionDiscriminator,
    discriminator_loss,
    generator_loss,
)
from ml.generate import sample_glyphs
from ml.utils import environment_info, resolve_device, seed_everything, sha256_file, write_json
from ml.visualize import save_grid


@dataclass
class GANTrainingConfig:
    data_root: str = "data"
    prepared_dir: str = "artifacts/data"
    output_dir: str = "artifacts/cgan-pilot"
    epochs: int = 3
    max_train_batches: int = 500
    batch_size: int = 128
    latent_dim: int = 32
    label_dim: int = 32
    base_channels: int = 32
    generator_lr: float = 0.0002
    discriminator_lr: float = 0.0002
    adam_beta1: float = 0.0
    adam_beta2: float = 0.9
    seed: int = 42
    cpu_threads: int = 4
    num_workers: int = 0
    device: str = "cpu"
    progress: bool = True

    def __post_init__(self):
        for name in (
            "epochs",
            "max_train_batches",
            "batch_size",
            "latent_dim",
            "label_dim",
            "base_channels",
            "cpu_threads",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")
        for value in (self.generator_lr, self.discriminator_lr):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Learning rates must be finite and positive.")
        if not all(math.isfinite(v) and 0 <= v < 1 for v in (self.adam_beta1, self.adam_beta2)):
            raise ValueError("Adam betas must be finite and in [0, 1).")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2^32).")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("Device must be cpu, cuda or auto.")


@contextmanager
def frozen_discriminator(discriminator):
    """Keep gradients to fake images, while freezing D weights AND spectral-norm buffers."""
    training = discriminator.training
    flags = [parameter.requires_grad for parameter in discriminator.parameters()]
    discriminator.eval()
    discriminator.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(discriminator.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)
        discriminator.train(training)


def run_gan_epoch(
    generator,
    discriminator,
    loader,
    generator_optimizer,
    discriminator_optimizer,
    device,
    noise,
    *,
    max_batches,
    show_progress=True,
    description="cGAN train",
):
    """One D update and one G update per batch. This function performs model training."""
    generator.train()
    discriminator.train()
    total_batches = min(len(loader), max_batches)
    totals = dict.fromkeys(
        ("d_loss", "g_loss", "real_score", "fake_score", "d_grad", "g_grad"), 0.0
    )
    examples = batches = 0
    start = time.perf_counter()
    with tqdm(
        total=total_batches,
        desc=description,
        unit="batch",
        ascii=True,
        dynamic_ncols=True,
        mininterval=0.25,
        disable=not show_progress,
    ) as progress:
        for images, labels in islice(loader, total_batches):
            images, labels = images.to(device), labels.to(device)
            if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise FloatingPointError("Real glyphs must be finite foreground masks in [0, 1].")
            count = len(images)
            discriminator_optimizer.zero_grad(set_to_none=True)
            generator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                latent = torch.randn(
                    (count, generator.config.latent_dim), device=device, generator=noise
                )
                fake = generator(latent, labels)
            # One call keeps the same spectral normalization for the real/fake comparison.
            scores = discriminator(torch.cat((images, fake)), labels.repeat(2))
            real_scores, fake_scores = scores.chunk(2)
            d_loss = discriminator_loss(real_scores, fake_scores)
            if not torch.isfinite(d_loss):
                raise FloatingPointError("Non-finite discriminator loss.")
            d_loss.backward()
            d_grad = torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), 10.0, error_if_nonfinite=True
            )
            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad(set_to_none=True)

            latent = torch.randn(
                (count, generator.config.latent_dim), device=device, generator=noise
            )
            with frozen_discriminator(discriminator):
                g_loss = generator_loss(discriminator(generator(latent, labels), labels))
                if not torch.isfinite(g_loss):
                    raise FloatingPointError("Non-finite generator loss.")
                g_loss.backward()
            g_grad = torch.nn.utils.clip_grad_norm_(
                generator.parameters(), 10.0, error_if_nonfinite=True
            )
            generator_optimizer.step()
            values = (d_loss, g_loss, real_scores.mean(), fake_scores.mean(), d_grad, g_grad)
            for key, value in zip(totals, values, strict=True):
                totals[key] += float(value.detach()) * count
            examples += count
            batches += 1
            progress.set_postfix(
                D=f"{float(d_loss.detach()):.3f}",
                G=f"{float(g_loss.detach()):.3f}",
                avg_D=f"{totals['d_loss'] / examples:.3f}",
                avg_G=f"{totals['g_loss'] / examples:.3f}",
                examples=examples,
            )
            progress.update(1)
    if not examples:
        raise ValueError("Training loader is empty.")
    return {
        **{key: value / examples for key, value in totals.items()},
        "examples": examples,
        "batches": batches,
        "generator_updates": batches,
        "discriminator_updates": batches,
        "seconds": time.perf_counter() - start,
    }


def train(config: GANTrainingConfig):
    seed_everything(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = resolve_device(config.device)
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError("Choose a new cGAN output directory; existing runs are preserved.")
    print("Checking existing dataset hashes and loading the training partition...", flush=True)
    dataset, training, validation, manifest = load_partitions(
        Path(config.data_root), Path(config.prepared_dir)
    )
    targets = dataset.targets[training.indices]
    counts = torch.bincount(targets, minlength=len(ALPHABET)).double()
    if (counts == 0).any():
        raise ValueError("Training partition must include all 62 classes.")
    sampling_rng = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(
        counts.reciprocal()[targets], len(training), replacement=True, generator=sampling_rng
    )
    loader = DataLoader(
        training,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed + 1),
    )
    model_config = GANModelConfig(config.latent_dim, config.label_dim, config.base_channels)
    generator = ConditionalGenerator(model_config).to(device)
    discriminator = ProjectionDiscriminator(model_config).to(device)
    betas = (config.adam_beta1, config.adam_beta2)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.generator_lr, betas=betas)
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config.discriminator_lr, betas=betas
    )
    noise = torch.Generator(device=device).manual_seed(config.seed + 101)
    budget = config.epochs * min(len(loader), config.max_train_batches)
    environment = environment_info(device)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", asdict(config))
    write_json(output / "environment.json", environment)
    write_json(output / "data-manifest.json", manifest)
    # Git HEAD alone misses uncommitted source changes in this working project.
    source_hashes = {
        name: sha256_file(Path(__file__).parent / name)
        for name in (
            "gan.py",
            "train_gan.py",
            "checkpoint.py",
            "dataset.py",
            "model.py",
            "utils.py",
            "characters.py",
            "generate.py",
            "visualize.py",
        )
    }
    write_json(output / "source-sha256.json", source_hashes)
    print(
        f"Fresh initialization; device={device}; train={len(training):,}; "
        f"validation={len(validation):,} (not optimized). Budget: {budget:,} batches, "
        f"one generator + one discriminator update per batch.\n"
        f"Parameters: G={sum(p.numel() for p in generator.parameters()):,}; "
        f"D={sum(p.numel() for p in discriminator.parameters()):,}. "
        "GAN losses are not accuracy or a checkpoint ranking.",
        flush=True,
    )
    history = []
    quality_status = "pipeline-smoke-only" if budget <= 16 else "unreviewed-bounded-gan-pilot"
    for epoch in range(1, config.epochs + 1):
        metrics = run_gan_epoch(
            generator,
            discriminator,
            loader,
            g_optimizer,
            d_optimizer,
            device,
            noise,
            max_batches=config.max_train_batches,
            show_progress=config.progress,
            description=f"cGAN {epoch}/{config.epochs}",
        )
        history.append({"epoch": epoch, **metrics})
        write_json(output / "history.json", history)
        print(
            f"epoch={epoch} D={metrics['d_loss']:.3f} G={metrics['g_loss']:.3f} "
            f"examples={metrics['examples']:,} seconds={metrics['seconds']:.1f}. "
            "Saving generator checkpoints and fixed-seed samples...",
            flush=True,
        )
        metadata = {
            "epoch": epoch,
            "training_config": asdict(config),
            "environment": environment,
            "split_sha256": manifest["split_sha256"],
            "quality_status": quality_status,
            "initialization": "from-scratch",
            "training_metrics": metrics,
            "source_sha256": source_hashes,
            "checkpoint_selection": "none; every epoch saved",
            "resume_supported": False,
        }
        epoch_path = output / f"epoch-{epoch:03d}.pt"
        save_checkpoint(epoch_path, generator, **metadata)
        save_checkpoint(output / "last.pt", generator, **metadata)
        generator.eval()
        images = sample_glyphs(generator, ALPHABET, config.seed, variations=4)
        names = [char for char in ALPHABET for _ in range(4)]
        save_grid(images, output / "samples" / f"prior-epoch-{epoch:03d}.png", names, columns=16)
        write_json(
            output / "summary.json",
            {
                "model_family": "cgan",
                "epochs_completed": epoch,
                "training_examples_drawn": sum(row["examples"] for row in history),
                "batches_completed": sum(row["batches"] for row in history),
                "batch_budget": budget,
                "quality_status": quality_status,
                "initialization": "from-scratch",
                "test_set_evaluated": False,
                "validation_used_for_optimization": False,
                "best_checkpoint": None,
                "last_checkpoint_sha256": sha256_file(output / "last.pt"),
                "checkpoints": [f"epoch-{i:03d}.pt" for i in range(1, epoch + 1)],
                "warning": (
                    "Bounded partial-data pilot, not a converged model or quality pass. "
                    "Losses are adversarial diagnostics, not comparable to VAE losses. "
                    "Generator-only checkpoints support inference, not training resume."
                ),
            },
        )
    return generator, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cgan-pilot.json"))
    parser.add_argument("--output", type=str)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("device", "epochs", "max_train_batches"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    if args.output:
        values["output_dir"] = args.output
    if args.no_progress:
        values["progress"] = False
    train(GANTrainingConfig(**values))


if __name__ == "__main__":
    main()
