"""Owner-run only: bounded real-word training for the future generator's content reader."""

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset
from tqdm import tqdm

from ml.iam import IAMWordDataset, collate_words
from ml.reader_balance import FrequencyBalancedCollator
from ml.utils import environment_info, resolve_device, seed_everything, sha256_file, write_json
from ml.word_reader import (
    READER_ROLE,
    WordReader,
    WordReaderConfig,
    adapt_reader,
    greedy_decode,
    load_reader,
    reading_metrics,
    save_reader,
    word_ctc_loss,
    write_predictions,
)


@dataclass
class ReaderTrainingConfig:
    prepared: str = "artifacts/iam-words"
    output: str = "artifacts/word-reader-pilot"
    epochs: int = 5
    batch_size: int = 16
    max_train_batches: int = 500
    max_validation_batches: int | None = 32
    learning_rate: float = 0.001
    channels: int = 24
    hidden: int = 128
    layers: int = 2
    seed: int = 314
    cpu_threads: int = 4
    device: str = "cpu"
    progress: bool = True
    init_checkpoint: str | None = None
    architecture: str = "mean"
    sampling: str = "shuffle"
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.005
    balance_word_frequency: bool = False

    def __post_init__(self):
        for name in ("epochs", "batch_size", "max_train_batches", "cpu_threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_validation_batches is not None and (
            not isinstance(self.max_validation_batches, int)
            or isinstance(self.max_validation_batches, bool)
            or self.max_validation_batches < 1
        ):
            raise ValueError("Validation limit must be a positive integer or null.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive.")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be in [0, 2^32).")
        if self.batch_size > 128 or self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("Invalid batch size or device.")
        if self.sampling not in {"shuffle", "coverage"}:
            raise ValueError("Sampling must be shuffle or coverage.")
        if not isinstance(self.balance_word_frequency, bool):
            raise ValueError("Word-frequency balancing must be a boolean.")
        if (
            not isinstance(self.early_stopping_patience, int)
            or isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience < 0
        ):
            raise ValueError("Early stopping patience must be a nonnegative integer.")
        if not math.isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0:
            raise ValueError("Early stopping minimum improvement must be finite and nonnegative.")
        WordReaderConfig(self.channels, self.hidden, self.layers, self.architecture)


@dataclass
class CERStopper:
    """Stop on selection CER stagnation; never inspect content-probe/test labels."""

    patience: int
    min_delta: float
    best: float = math.inf
    stale_checks: int = 0

    def observe(self, cer):
        if not math.isfinite(cer) or cer < 0:
            raise ValueError("Selection CER must be finite and nonnegative.")
        if cer < self.best and self.best - cer >= self.min_delta:
            self.best = cer
            self.stale_checks = 0
        else:
            self.stale_checks += 1
        return self.patience > 0 and self.stale_checks >= self.patience


class CoverageSampler(Sampler):
    """Continue a shuffled permutation across capped epochs before reshuffling.

    No workers/prefetch are used by this trainer, so yielded indices correspond to
    examples loaded. State is reported but optimizer/sampler resume is not supported.
    """

    def __init__(self, size, seed):
        if size < 1:
            raise ValueError("Coverage sampling needs a nonempty dataset.")
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = []
        self.position = 0
        self.draws = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        for _ in range(self.size):
            if self.position == len(self.order):
                self.order = torch.randperm(self.size, generator=self.generator).tolist()
                self.position = 0
            index = self.order[self.position]
            self.position += 1
            self.draws += 1
            yield index

    def report(self):
        return {
            "mode": "coverage",
            "draws": self.draws,
            "completed_data_passes": self.draws // self.size,
            "examples_into_next_pass": self.draws % self.size,
        }


def evaluation_loader(prepared, partition, batch_size, max_batches=None):
    if partition not in {"seen_validation", "content_validation"}:
        raise ValueError("Word-reader diagnostics use development data only.")
    if batch_size < 1 or (max_batches is not None and max_batches < 1):
        raise ValueError("Batch size and optional limit must be positive.")
    data = IAMWordDataset(prepared, partition)
    count = len(data) if max_batches is None else min(len(data), max_batches * batch_size)
    indices = np.linspace(0, len(data) - 1, count, dtype=int).tolist()
    identifiers = [data.rows[index]["id"] for index in indices]
    identity = {
        "partition": partition,
        "available_examples": len(data),
        "evaluated_examples": count,
        "partial": count < len(data),
        "selection": "evenly spaced source indices, fixed before fitting",
        "selected_ids_sha256": hashlib.sha256("\n".join(identifiers).encode()).hexdigest(),
        "records_sha256": data.manifest["records_sha256"],
    }
    return DataLoader(
        Subset(data, indices),
        batch_size=batch_size,
        collate_fn=collate_words,
        generator=torch.Generator().manual_seed(43),
    ), identity


def reader_epoch(
    model,
    loader,
    device,
    *,
    optimizer=None,
    max_batches=None,
    show_progress=True,
    description="Word-reader validation",
):
    training = optimizer is not None
    model.train(training)
    count = len(loader) if max_batches is None else min(len(loader), max_batches)
    rows, total_loss, characters, edits, examples = [], 0.0, 0, 0, 0
    start = time.perf_counter()
    with (
        torch.set_grad_enabled(training),
        tqdm(
            total=count,
            desc=description,
            unit="batch",
            ascii=True,
            dynamic_ncols=True,
            mininterval=0.25,
            disable=not show_progress,
        ) as progress,
    ):
        for batch in islice(loader, count):
            images = batch["images"].to(device)
            if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise FloatingPointError(
                    "Word images must contain finite foreground intensities in [0, 1]."
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
            probabilities, lengths = model(images, batch["widths"])
            loss = word_ctc_loss(
                probabilities,
                lengths,
                batch,
                sample_weights=batch.get("reader_loss_weights") if training else None,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite word CTC loss; alignment errors are not zeroed out."
                )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
            predictions = greedy_decode(probabilities, lengths)
            new_rows = [
                {"id": identifier, "expected": expected, "predicted": predicted}
                for identifier, expected, predicted in zip(
                    batch["ids"], batch["text"], predictions, strict=True
                )
            ]
            metrics = reading_metrics(new_rows)
            rows.extend(new_rows)
            examples += len(new_rows)
            total_loss += float(loss.detach()) * len(new_rows)
            characters += metrics["reference_characters"]
            edits += metrics["character_edits"]
            progress.set_postfix(
                CTC=f"{total_loss / examples:.3f}", CER=f"{edits / characters:.1%}", words=examples
            )
            progress.update(1)
    if not rows:
        raise ValueError("Word-reader epoch is empty.")
    return {
        "loss": total_loss / examples,
        **reading_metrics(rows),
        "batches": count,
        "optimizer_updates": count if training else 0,
        "seconds": time.perf_counter() - start,
    }, rows


def train(config: ReaderTrainingConfig):
    seed_everything(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = resolve_device(config.device)
    output, prepared = Path(config.output), Path(config.prepared)
    if output.exists():
        raise FileExistsError("Choose a new word-reader output directory.")
    print(
        "Loading verified IAM development data; reserved evaluation pixels stay closed...",
        flush=True,
    )
    training = IAMWordDataset(prepared, "train")
    training_collator = (
        FrequencyBalancedCollator(training.rows) if config.balance_word_frequency else collate_words
    )
    weighting = (
        training_collator.report()
        if config.balance_word_frequency
        else {"mode": "uniform-examples"}
    )
    sampler = CoverageSampler(len(training), config.seed) if config.sampling == "coverage" else None
    loader = DataLoader(
        training,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=training_collator,
        generator=torch.Generator().manual_seed(config.seed),
    )
    selection, selection_identity = evaluation_loader(
        prepared, "seen_validation", config.batch_size, config.max_validation_batches
    )
    content, content_identity = evaluation_loader(
        prepared, "content_validation", config.batch_size, config.max_validation_batches
    )
    target_model_config = WordReaderConfig(
        config.channels, config.hidden, config.layers, config.architecture
    )
    model = WordReader(target_model_config)
    initialization = {"kind": "from-scratch", "optimizer_restored": False}
    if config.init_checkpoint:
        model, metadata = load_reader(Path(config.init_checkpoint))
        if metadata["records_sha256"] != training.manifest["records_sha256"]:
            raise ValueError("Initialization reader uses different prepared data.")
        source_architecture = model.config.architecture
        model = adapt_reader(model, target_model_config)
        initialization = {
            "kind": "own-checkpoint",
            "path": config.init_checkpoint,
            "sha256": sha256_file(Path(config.init_checkpoint)),
            "source_epoch": metadata["epoch"],
            "optimizer_restored": False,
            "source_architecture": source_architecture,
            "target_architecture": config.architecture,
            "zero_initialized_spatial_adapter": source_architecture != config.architecture,
        }
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001)
    output.mkdir(parents=True, exist_ok=False)
    environment = environment_info(device)
    write_json(output / "config.json", asdict(config))
    write_json(output / "environment.json", environment)
    write_json(output / "data-manifest.json", training.manifest)
    write_json(output / "training-weighting.json", weighting)
    identities = {"selection": selection_identity, "content_probe": content_identity}
    write_json(output / "evaluation-partitions.json", identities)
    source_hashes = {
        name: sha256_file(Path(__file__).parent / name)
        for name in (
            "word_reader.py",
            "train_word_reader.py",
            "iam.py",
            "characters.py",
            "utils.py",
            "reader_balance.py",
        )
    }
    write_json(output / "source-sha256.json", source_hashes)
    budget = config.epochs * min(len(loader), config.max_train_batches)
    status = "pipeline-smoke-only" if budget <= 16 else "unreviewed-content-reader-pilot"
    print(
        f"Device={device}; parameters={sum(p.numel() for p in model.parameters()):,}; "
        f"budget={budget:,} updates; role={READER_ROLE}. No generator training.",
        flush=True,
    )

    def checkpoint_metadata(epoch, selection_metrics):
        return {
            "epoch": epoch,
            "training_config": asdict(config),
            "records_sha256": training.manifest["records_sha256"],
            "quality_status": status,
            "selection": selection_metrics,
            "selection_partition": selection_identity,
            "initialization": initialization,
            "source_sha256": source_hashes,
            "resume_supported": False,
            "independent_generator_evaluator": False,
            "training_sampling": sampler.report() if sampler else {"mode": "shuffle"},
            "training_weighting": weighting,
        }

    initial_selection, initial_predictions = reader_epoch(
        model,
        selection,
        device,
        show_progress=config.progress,
        description="Initial reader selection",
    )
    write_json(output / "initial-selection.json", initial_selection)
    write_predictions(output / "selection-epoch-000.csv", initial_predictions)
    save_reader(output / "best.pt", model, **checkpoint_metadata(0, initial_selection))
    history = []
    best = (initial_selection["character_error_rate"], initial_selection["loss"])
    best_epoch = 0
    stopper = CERStopper(config.early_stopping_patience, config.early_stopping_min_delta)
    stopper.observe(initial_selection["character_error_rate"])
    stop_reason = "budget_completed"
    for epoch in range(1, config.epochs + 1):
        train_metrics, _ = reader_epoch(
            model,
            loader,
            device,
            optimizer=optimizer,
            max_batches=config.max_train_batches,
            show_progress=config.progress,
            description=f"Word reader train {epoch}/{config.epochs}",
        )
        selection_metrics, predictions = reader_epoch(
            model,
            selection,
            device,
            show_progress=config.progress,
            description=f"Word reader selection {epoch}/{config.epochs}",
        )
        history.append({"epoch": epoch, "train": train_metrics, "selection": selection_metrics})
        write_json(output / "history.json", history)
        write_predictions(output / f"selection-epoch-{epoch:03d}.csv", predictions)
        metadata = checkpoint_metadata(epoch, selection_metrics)
        save_reader(output / "last.pt", model, **metadata)
        rank = (selection_metrics["character_error_rate"], selection_metrics["loss"])
        if rank < best:
            best, best_epoch = rank, epoch
            save_reader(output / "best.pt", model, **metadata)
        print(
            f"epoch={epoch} CTC={selection_metrics['loss']:.3f}; "
            f"selection CER={selection_metrics['character_error_rate']:.2%}; "
            f"exact words={selection_metrics['exact_word_accuracy']:.2%}; "
            f"seconds={train_metrics['seconds'] + selection_metrics['seconds']:.1f}",
            flush=True,
        )
        if stopper.observe(selection_metrics["character_error_rate"]):
            stop_reason = "selection_plateau"
            print(
                f"Stopped: no selection CER improvement of at least "
                f"{config.early_stopping_min_delta:.2%} for "
                f"{config.early_stopping_patience} checks. Reviewing the best checkpoint.",
                flush=True,
            )
            break
    best_model, _ = load_reader(output / "best.pt", device)
    content_metrics, predictions = reader_epoch(
        best_model,
        content,
        device,
        show_progress=config.progress,
        description="Unseen-word probe",
    )
    write_predictions(output / "content-predictions.csv", predictions)
    report = {
        "role": READER_ROLE,
        "quality_status": status,
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "early_stopping": asdict(stopper),
        "best_epoch": best_epoch,
        "selection_criterion": "Lowest CER, then lowest CTC loss",
        "best_selection_character_error_rate": best[0],
        "initial_selection": initial_selection,
        "content_probe": content_metrics,
        "evaluation_partitions": identities,
        "initialization": initialization,
        "actual_optimizer_updates": sum(row["train"]["optimizer_updates"] for row in history),
        "training_word_draws": sum(row["train"]["examples"] for row in history),
        "best_checkpoint_sha256": sha256_file(output / "best.pt"),
        "test_set_evaluated": False,
        "ready_for_generator_training": False,
        "training_sampling": sampler.report() if sampler else {"mode": "shuffle"},
        "training_weighting": weighting,
        "warning": "Reader competence needs review. This is a training-loss reader, "
        "not an independent generator evaluator. Capped probes are partial diagnostics.",
    }
    write_json(output / "summary.json", report)
    print(f"Saved {output / 'summary.json'}; reader readiness still requires review.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/word-reader-pilot.json"))
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8"))
    for key in (
        "output",
        "epochs",
        "max_train_batches",
        "max_validation_batches",
        "device",
        "init_checkpoint",
    ):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    if args.no_progress:
        values["progress"] = False
    train(ReaderTrainingConfig(**values))


if __name__ == "__main__":
    main()
