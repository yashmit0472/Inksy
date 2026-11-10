"""Train the writer-conditioned generator on the frozen clear-writer pool."""

import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from ml.iam import IAMWordDataset
from ml.utils import environment_info, seed_everything, sha256_file, write_json
from ml.word_generator import WordGenerator, WordGeneratorConfig, paired_loss, save_generator
from ml.word_generator_data import image_batch
from ml.word_generator_preview import generate_images, save_grid

PARTITIONS = ("train", "seen_validation", "content_validation")


def load_pool(prepared, index):
    pool = json.loads(index.read_text(encoding="utf-8"))
    partitions = {}
    train_vocab = set()
    for partition in PARTITIONS:
        data = IAMWordDataset(prepared, partition)
        if data.manifest["records_sha256"] != pool["source_records_sha256"]:
            raise ValueError("IAM records differ from the pinned pool.")
        ids = pool["partitions"][partition]["ids"]
        selected = [
            (i, row)
            for i, row in enumerate(data.rows)
            if row["writer_id"] in pool["writers"] and re.fullmatch("[a-z]{2,10}", row["text"])
        ]
        if [row["id"] for _, row in selected] != ids or len(ids) != len(set(ids)):
            raise ValueError(f"Pool membership differs from prepared {partition}.")
        rows = [row for _, row in selected]
        if partition == "train":
            train_vocab = {row["text"].casefold() for row in data.rows}
        if partition == "content_validation":
            if any(row["text"].casefold() in train_vocab for row in rows):
                raise ValueError("Content holdout overlaps the IAM training vocabulary.")
            partitions[partition] = rows
        else:
            partitions[partition] = [(data[i], row["writer_id"]) for i, row in selected]
    return pool, partitions


def batches(samples, size, generator=None):
    order = list(range(len(samples)))
    if generator is not None:
        order = torch.randperm(len(samples), generator=generator).tolist()
    # Shuffle within width bands across epochs, then shuffle the resulting batches.
    groups = []
    for start in range(0, len(order), size * 16):
        band = sorted(
            order[start : start + size * 16], key=lambda i: samples[i][0]["image"].shape[-1]
        )
        groups.extend(band[i : i + size] for i in range(0, len(band), size))
    if generator is not None:
        groups = [groups[i] for i in torch.randperm(len(groups), generator=generator).tolist()]
    return [[samples[i] for i in group] for group in groups]


def batch_loss(model, group):
    items, writers = zip(*group, strict=True)
    images, widths, texts = image_batch(items)
    return paired_loss(model(texts, widths, writers=writers), images, widths)


@torch.inference_mode()
def evaluate(model, groups):
    model.eval()
    total = Counter()
    count = 0
    for group in groups:
        metrics = batch_loss(model, group)
        total.update({name: float(value) * len(group) for name, value in metrics.items()})
        count += len(group)
    return {name: value / count for name, value in total.items()}


def run(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    pool, data = load_pool(args.prepared, args.pool)
    writers = tuple(pool["writers"])
    model = WordGenerator(
        WordGeneratorConfig(channels=48, architecture="coordinate", writers=writers)
    )
    random = torch.Generator().manual_seed(args.seed)
    train, validation = data["train"], data["seen_validation"]
    validation_groups = batches(validation, args.batch_size)
    if args.preflight:
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        groups = batches(train, args.batch_size, random)
        timings = []
        for group in groups[:8]:
            model.zero_grad(set_to_none=True)
            start = time.perf_counter()
            batch_loss(model, group)["total"].backward()
            timings.append(time.perf_counter() - start)
        assert all(torch.equal(value, initial[key]) for key, value in model.state_dict().items())
        assert model.advance.weight.grad.norm().isfinite() and model.advance.weight.grad.norm() > 0
        print(
            {
                "train": len(train),
                "validation": len(validation),
                "updates_per_epoch": len(groups),
                "median_seconds": statistics.median(timings[1:]),
                "weights_unchanged": True,
            }
        )
        return

    args.output.mkdir(parents=True, exist_ok=False)
    holdout = sorted({row["text"] for row in data["content_validation"]})
    provenance = {
        "writers": writers,
        "records_sha256": pool["source_records_sha256"],
        "pool_sha256": sha256_file(args.pool),
        "partitions": pool["partitions"],
        "seed": args.seed,
        "initialization": "random-from-scratch",
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (Path(__file__), Path("ml/word_generator.py"), Path("ml/iam.py"))
        },
        "environment": environment_info(torch.device("cpu")),
    }
    write_json(args.output / "provenance.json", provenance)
    write_json(args.output / "holdout-prompts.json", holdout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    preview_words = sorted(
        {item["text"] for item, writer in validation if writer == "150"},
        key=lambda word: (len(word), word),
    )[::3][:16]
    history = []
    best, updates, seen = float("inf"), 0, 0
    reason = "epoch_cap"
    start = time.perf_counter()
    initial = evaluate(model, validation_groups)
    with tqdm(
        total=args.epochs * len(batches(train, args.batch_size)),
        desc="Train writer-conditioned generator",
        unit="update",
        ascii=True,
    ) as progress:
        for epoch in range(1, args.epochs + 1):
            model.train()
            for group in batches(train, args.batch_size, random):
                if time.perf_counter() - start >= args.seconds:
                    reason = "time_cap"
                    break
                optimizer.zero_grad(set_to_none=True)
                loss = batch_loss(model, group)["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                seen += len(group)
                progress.update()
                if updates % 20 == 0:
                    progress.set_postfix(loss=f"{loss.item():.3f}", epoch=epoch)
            metrics = evaluate(model, validation_groups)
            row = {"epoch": epoch, "updates": updates, "examples_seen": seen, "validation": metrics}
            history.append(row)
            save_generator(args.output / "last.pt", model, {**provenance, **row})
            if metrics["total"] < best:
                best = metrics["total"]
                save_generator(args.output / "best.pt", model, {**provenance, **row})
                save_grid(
                    args.output / "seen-validation.png",
                    generate_images(model, preview_words, writer="150"),
                    preview_words,
                    "Generated validation words | writer 150 | predicted widths",
                )
            write_json(args.output / "history.json", history)
            tqdm.write(f"Epoch {epoch}, updates {updates}, validation loss {metrics['total']:.4f}")
            if reason == "time_cap":
                break
    write_json(
        args.output / "summary.json",
        {
            "stop_reason": reason,
            "updates": updates,
            "examples_seen": seen,
            "seconds": time.perf_counter() - start,
            "initial_validation": initial,
            "best_validation_loss": best,
            "best_checkpoint_sha256": sha256_file(args.output / "best.pt"),
            "holdout_words": len(holdout),
            "readability_status": "awaiting review",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/iam-words"))
    parser.add_argument(
        "--pool", type=Path, default=Path("artifacts/writer-target-audit/proposed-pool.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/multi-writer-pilot"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seconds", type=int, default=720)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.seconds, args.threads, args.batch_size) < 1 or args.batch_size > 64:
        parser.error("Use positive budgets and a batch size of at most 64.")
    run(args)


if __name__ == "__main__":
    main()
