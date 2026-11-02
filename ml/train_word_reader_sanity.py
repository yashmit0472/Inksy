"""Owner-run, bounded memorization diagnostic. This does not train a generator."""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from ml.diagnose_word_reader import diagnose
from ml.iam import IAMWordDataset, collate_words
from ml.utils import environment_info, seed_everything, sha256_file, write_json
from ml.word_reader import load_reader, save_reader, word_ctc_loss, write_predictions


@dataclass
class SanityConfig:
    prepared: str = "artifacts/iam-words"
    checkpoint: str = "artifacts/word-reader-check-200/best.pt"
    output: str = "artifacts/word-reader-memorize-16"
    max_updates: int = 600
    check_every: int = 25
    consecutive_passes: int = 3
    learning_rate: float = 0.001
    seed: int = 314
    cpu_threads: int = 4

    def __post_init__(self):
        for name in ("max_updates", "check_every", "consecutive_passes", "cpu_threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_updates > 2000 or self.check_every > self.max_updates:
            raise ValueError("Memorization must have a bounded budget of at most 2,000 updates.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive.")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid seed.")


def sanity_batch(prepared):
    """Load exactly the pinned 16 training images, never development/test images."""
    data = IAMWordDataset(prepared, "train")
    path = prepared / "sanity-set.json"
    checksum = sha256_file(path)
    if checksum != data.manifest["sanity_set_sha256"]:
        raise ValueError("Sanity-set identity changed after data preparation.")
    specification = json.loads(path.read_text(encoding="utf-8"))
    identifiers = specification["word_ids"]
    if (
        specification["partition"] != "train"
        or len(set(identifiers)) != 16
        or len(identifiers) != 16
    ):
        raise ValueError("The sanity set must contain exactly 16 distinct training IDs.")
    indices = {row["id"]: index for index, row in enumerate(data.rows)}
    if any(identifier not in indices for identifier in identifiers):
        raise ValueError("Sanity-set ID is outside the training partition.")
    selected = [indices[identifier] for identifier in identifiers]
    if any(data.rows[index]["writer_id"] != specification["writer_id"] for index in selected):
        raise ValueError("Sanity set must use its declared single writer.")
    batch = collate_words([data[index] for index in selected])
    if len({text.casefold() for text in batch["text"]}) != 16:
        raise ValueError("Sanity set must contain 16 distinct word strings.")
    return batch, {
        "partition": "train",
        "word_ids": identifiers,
        "texts": batch["text"],
        "writer_id": specification["writer_id"],
        "records_sha256": data.manifest["records_sha256"],
        "sanity_set_sha256": checksum,
        "purpose": "Training-set memorization only; not generalization evaluation.",
    }


def passes_memorization(metrics):
    # Identical widths on blank controls reveal gross sequence-length shortcuts.
    # This is a local debugging gate, not evidence of general reading competence.
    return (
        metrics["real"]["exact_word_accuracy"] == 1.0
        and metrics["blank"]["exact_word_accuracy"] <= 0.25
        and metrics["real"]["ctc_loss"] < metrics["blank"]["ctc_loss"]
    )


def train(config):
    output = Path(config.output)
    if output.exists():
        raise FileExistsError("Choose a new memorization output directory.")
    seed_everything(config.seed)
    torch.set_num_threads(config.cpu_threads)
    batch, identity = sanity_batch(Path(config.prepared))
    model, original = load_reader(Path(config.checkpoint))
    if original["records_sha256"] != identity["records_sha256"]:
        raise ValueError("Initialization checkpoint and sanity data identities differ.")
    initialization = {
        "checkpoint": config.checkpoint,
        "sha256": sha256_file(Path(config.checkpoint)),
        "optimizer_restored": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", asdict(config))
    write_json(output / "data-identity.json", identity)
    write_json(output / "environment.json", environment_info(torch.device("cpu")))
    source_hashes = {
        name: sha256_file(Path(__file__).parent / name)
        for name in (
            "train_word_reader_sanity.py",
            "diagnose_word_reader.py",
            "word_reader.py",
            "iam.py",
            "utils.py",
        )
    }
    write_json(output / "source-sha256.json", source_hashes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001)
    history, streak, best = [], 0, (math.inf, math.inf)
    started = time.perf_counter()
    print("16 cached training words; bounded memorization only; no generator training.", flush=True)
    with tqdm(total=config.max_updates, desc="Memorize 16 words", unit="update", ascii=True) as bar:
        for step in range(config.max_updates + 1):
            if step:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logp, lengths = model(batch["images"], batch["widths"])
                loss = word_ctc_loss(logp, lengths, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite memorization loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                bar.update(1)
                bar.set_postfix(CTC=f"{float(loss.detach()):.3f}", passes=streak, refresh=False)
            if step % config.check_every and step != config.max_updates:
                continue
            metrics, rows = diagnose(model, [batch])
            passed = passes_memorization(metrics)
            streak = streak + 1 if passed else 0
            history.append({"updates": step, "conditions": metrics, "gate_passed": passed})
            write_json(output / "history.json", history)
            write_predictions(output / f"predictions-{step:04d}.csv", rows["real"])
            metadata = {
                "epoch": step,
                "optimizer_updates": step,
                "records_sha256": identity["records_sha256"],
                "quality_status": "training-set-memorization-only",
                "initialization": initialization,
                "training_config": asdict(config),
                "sanity_set": identity,
                "source_sha256": source_hashes,
                "ready_for_generator_training": False,
                "independent_generator_evaluator": False,
                "resume_supported": False,
            }
            rank = (metrics["real"]["character_error_rate"], metrics["real"]["ctc_loss"])
            if rank < best:
                best = rank
                save_reader(output / "best.pt", model, **metadata)
                write_predictions(output / "best-predictions.csv", rows["real"])
            save_reader(output / "last.pt", model, **metadata)
            tqdm.write(
                f"update={step}: exact={metrics['real']['exact_word_accuracy']:.0%}; "
                f"CER={metrics['real']['character_error_rate']:.1%}; "
                f"blank exact={metrics['blank']['exact_word_accuracy']:.0%}; "
                f"consecutive passes={streak}/{config.consecutive_passes}"
            )
            if streak >= config.consecutive_passes:
                break
    report = {
        "purpose": "Training-set memorization diagnostic, not handwriting generation.",
        "passed": streak >= config.consecutive_passes,
        "actual_optimizer_updates": step,
        "training_word_draws": step * 16,
        "seconds": time.perf_counter() - started,
        "final_conditions": metrics,
        "initialization": initialization,
        "best_checkpoint_sha256": sha256_file(output / "best.pt"),
        "ready_for_generator_training": False,
        "test_set_evaluated": False,
        "next_action": "Review the diagnostic before any larger training run.",
    }
    write_json(output / "summary.json", report)
    print(f"Memorization passed={report['passed']}. Saved {output / 'summary.json'}.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/word-reader-memorize-16.json"))
    parser.add_argument(
        "--inspect-only", action="store_true", help="Verify inputs without training."
    )
    args = parser.parse_args()
    config = SanityConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    if args.inspect_only:
        batch, identity = sanity_batch(Path(config.prepared))
        model, metadata = load_reader(Path(config.checkpoint))
        if metadata["records_sha256"] != identity["records_sha256"]:
            raise ValueError("Checkpoint and data identities differ.")
        print(
            {
                "shape": list(batch["images"].shape),
                "identity": identity,
                "max_updates": config.max_updates,
                "training_performed": False,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
        return
    train(config)


if __name__ == "__main__":
    main()
