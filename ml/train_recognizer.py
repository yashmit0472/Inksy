"""OWNER-RUN ONLY: train the independent evaluator on real handwriting, not generated images."""

import argparse
import hashlib
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from ml.characters import ALPHABET, CONTRACT_VERSION, PREPROCESSING_VERSION
from ml.dataset import load_partitions, stratified_split
from ml.recognizer import RECOGNIZER_FORMAT, CharacterRecognizer, confusion_metrics, load_recognizer
from ml.utils import environment_info, resolve_device, seed_everything, sha256_file, write_json


def evaluator_partitions(dataset, validation, seed=915):
    """Separate classifier selection from its final real-data audit; never open official test."""
    labels = dataset.targets[validation.indices].numpy()
    selection, audit = stratified_split(labels, 0.5, seed)
    original = np.asarray(validation.indices, dtype=np.int64)
    selection, audit = original[selection], original[audit]
    identity = {
        "seed": seed,
        "selection_count": len(selection),
        "audit_count": len(audit),
        "selection_sha256": hashlib.sha256(selection.tobytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(audit.tobytes()).hexdigest(),
    }
    return Subset(dataset, selection.tolist()), Subset(dataset, audit.tolist()), identity


def recognizer_epoch(
    model,
    loader,
    device,
    *,
    optimizer=None,
    max_batches=None,
    description="Recognizer validation",
    show_progress=True,
):
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((len(ALPHABET), len(ALPHABET)), dtype=torch.long)
    total_loss, examples = 0.0, 0
    batch_count = len(loader) if max_batches is None else min(len(loader), max_batches)
    with (
        torch.set_grad_enabled(training),
        tqdm(
            total=batch_count,
            desc=description,
            unit="batch",
            ascii=True,
            dynamic_ncols=True,
            mininterval=0.25,
            disable=not show_progress,
        ) as progress,
    ):
        for images, labels in islice(loader, batch_count):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite recognizer loss.")
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.detach().argmax(1)
            bins = (labels * len(ALPHABET) + predictions).cpu()
            confusion += torch.bincount(bins, minlength=len(ALPHABET) ** 2).reshape_as(confusion)
            examples += len(images)
            total_loss += loss.item() * len(images)
            progress.set_postfix(
                loss=f"{total_loss / examples:.3f}",
                accuracy=f"{int(confusion.diag().sum()) / examples:.1%}",
                refresh=False,
            )
            progress.update(1)
    metrics = confusion_metrics(confusion)
    return {"loss": total_loss / examples, **metrics}


def train_recognizer(
    *,
    root=Path("data"),
    prepared=Path("artifacts/data"),
    output=Path("artifacts/recognizer"),
    epochs=3,
    batch_size=256,
    device="cpu",
    seed=123,
    max_train_batches=None,
):
    if epochs < 1 or batch_size < 1 or (max_train_batches is not None and max_train_batches < 1):
        raise ValueError("Epochs, batch size and any batch limit must be positive.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new or empty recognizer output directory.")
    seed_everything(seed)
    torch.set_num_threads(4)
    device = resolve_device(device)
    print(
        "Loading real data; reserving separate classifier selection and audit partitions...",
        flush=True,
    )
    dataset, training, validation, manifest = load_partitions(root, prepared)
    selection, audit, audit_identity = evaluator_partitions(dataset, validation)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "max_train_batches": max_train_batches,
    }
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment_info(device))
    write_json(output / "audit-partition.json", audit_identity)
    generator = torch.Generator().manual_seed(seed)
    targets = dataset.targets[training.indices]
    counts = torch.bincount(targets, minlength=len(ALPHABET)).double()
    if (counts == 0).any():
        raise ValueError("Missing training class.")
    sampler = WeightedRandomSampler(
        counts.reciprocal()[targets], len(training), replacement=True, generator=generator
    )
    loader = DataLoader(training, batch_size=batch_size, sampler=sampler, generator=generator)
    selection_loader = DataLoader(
        selection,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    model = CharacterRecognizer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    history, best = [], float("inf")
    for epoch in range(1, epochs + 1):
        training_metrics = recognizer_epoch(
            model,
            loader,
            device,
            optimizer=optimizer,
            max_batches=max_train_batches,
            description=f"Recognizer train {epoch}/{epochs}",
        )
        selection_metrics = recognizer_epoch(
            model, selection_loader, device, description=f"Recognizer selection {epoch}/{epochs}"
        )
        history.append({"epoch": epoch, "train": training_metrics, "selection": selection_metrics})
        write_json(output / "history.json", history)
        if selection_metrics["loss"] < best:
            best = selection_metrics["loss"]
            payload = {
                "format": RECOGNIZER_FORMAT,
                "alphabet": ALPHABET,
                "contract_version": CONTRACT_VERSION,
                "preprocessing_version": PREPROCESSING_VERSION,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "training_config": config,
                "split_sha256": manifest["split_sha256"],
                "audit_partition": audit_identity,
                "selection": selection_metrics,
                "quality_status": "smoke-only"
                if max_train_batches is not None
                else "awaiting-audit",
            }
            temporary = output / "recognizer.tmp"
            torch.save(payload, temporary)
            temporary.replace(output / "best.pt")
        print(
            f"Epoch {epoch}: real selection accuracy={selection_metrics['accuracy']:.2%}, "
            f"macro={selection_metrics['macro_accuracy']:.2%}",
            flush=True,
        )
    # Audit only once after selecting the best epoch. Do not tune on this audit's scores.
    selected, payload = load_recognizer(output / "best.pt", device)
    audit_loader = DataLoader(
        audit,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 2),
    )
    audit_metrics = recognizer_epoch(selected, audit_loader, device, description="Real-data audit")
    payload["real_audit"] = audit_metrics
    payload["quality_status"] = (
        "smoke-only" if max_train_batches is not None else "audited-research-scorer"
    )
    temporary = output / "recognizer.tmp"
    torch.save(payload, temporary)
    temporary.replace(output / "best.pt")
    write_json(
        output / "summary.json",
        {
            "best_epoch": payload["epoch"],
            "quality_status": payload["quality_status"],
            "real_audit": audit_metrics,
            "checkpoint_sha256": sha256_file(output / "best.pt"),
            "test_set_evaluated": False,
            "warning": (
                "Recognizer agreement is a proxy for identity, "
                "not human readability or stroke quality."
            ),
        },
    )
    print(
        f"Saved {output / 'best.pt'}; real audit accuracy={audit_metrics['accuracy']:.2%}. "
        "This trained the evaluator, not the handwriting generator.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/recognizer"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-train-batches", type=int)
    args = parser.parse_args()
    train_recognizer(**vars(args))


if __name__ == "__main__":
    main()
