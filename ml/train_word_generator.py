"""Owner-run bounded generator fit. --preflight never creates an optimizer."""

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from ml.utils import environment_info, seed_everything, sha256_file, write_json
from ml.word_generator import (
    WordGenerator,
    WordGeneratorConfig,
    adapt_generator,
    load_generator,
    paired_loss,
    save_generator,
)
from ml.word_generator_data import image_batch, load_scope
from ml.word_generator_preview import export_fit_review, generate_images, save_grid


@dataclass(frozen=True)
class FitConfig:
    prepared: str = "artifacts/iam-words"
    output: str = "artifacts/word-generator-fit-32"
    writer_id: str = "000"
    records_sha256: str = "1a4ad15ed16d2991ba04d5d77291b5191274a4882699b461e1f0d80177f5c01c"
    seed: int = 42
    max_updates: int = 1000
    max_seconds: int = 1200
    batch_size: int = 8
    check_every: int = 100
    learning_rate: float = 0.001
    cpu_threads: int = 4
    architecture: str = "plain"
    init_checkpoint: str | None = None
    init_sha256: str | None = None

    def __post_init__(self):
        for key, maximum in {
            "max_updates": 1000,
            "max_seconds": 1200,
            "batch_size": 32,
            "check_every": 1000,
            "cpu_threads": 16,
        }.items():
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{key} must be an integer in [1, {maximum}].")
        if self.check_every > self.max_updates or 32 % self.batch_size:
            raise ValueError("Check interval must fit the budget; batch size must divide 32.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid seed.")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.01:
            raise ValueError("Learning rate must be in (0, 0.01].")
        if not self.writer_id or len(self.records_sha256) != 64:
            raise ValueError("Writer identity and pinned data digest are required.")
        WordGeneratorConfig(architecture=self.architecture)
        if bool(self.init_checkpoint) != bool(self.init_sha256):
            raise ValueError("Warm starts require both a checkpoint path and its pinned SHA256.")
        if self.init_sha256 and (
            len(self.init_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.init_sha256)
        ):
            raise ValueError("Invalid initialization digest.")


def prepare(config):
    seed_everything(config.seed)
    torch.set_num_threads(config.cpu_threads)
    data, selected, _, identity = load_scope(
        config.prepared, config.writer_id, config.seed, config.records_sha256
    )
    items = [data[index] for index in selected]
    if config.init_checkpoint:
        path = Path(config.init_checkpoint)
        if sha256_file(path) != config.init_sha256:
            raise ValueError("Initialization checkpoint differs from its pinned SHA256.")
        model, checkpoint = load_generator(path)
        source = checkpoint["provenance"]
        if (
            source["writer_id"] != config.writer_id
            or source["records_sha256"] != identity["records_sha256"]
            or source.get("training_ids") != identity["diagnostic_ids"]
        ):
            raise ValueError("Initialization must use the same writer, data and 32 training IDs.")
        model = adapt_generator(model, config.architecture)
    else:
        model = WordGenerator(WordGeneratorConfig(architecture=config.architecture))
    return model, items, identity


@torch.inference_mode()
def evaluate_fit(model, items, batch_size):
    model.eval()
    totals = {}
    for start in range(0, len(items), batch_size):
        images, widths, texts = image_batch(items[start : start + batch_size])
        result = model(texts, widths)
        metrics = paired_loss(result, images, widths)
        for name, value in metrics.items():
            if not torch.isfinite(value):
                raise FloatingPointError(f"Non-finite fit metric: {name}")
            totals[name] = totals.get(name, 0) + float(value) * len(texts) / len(items)
    return totals


def source_hashes():
    return {
        name: sha256_file(Path(name))
        for name in (
            "ml/word_generator.py",
            "ml/word_generator_data.py",
            "ml/train_word_generator.py",
            "ml/word_generator_preview.py",
            "ml/review_word_generator.py",
            "ml/iam.py",
            "ml/utils.py",
        )
    }


def preflight(config, output):
    """Forward/backward timing and finite gradients, without any weight update."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new preflight output directory.")
    model, items, identity = prepare(config)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    images, widths, texts = image_batch(items[: config.batch_size])
    durations = []
    for _ in range(4):
        started = time.perf_counter()
        model.zero_grad(set_to_none=True)
        result = model(texts, widths)
        losses = paired_loss(result, images, widths)
        losses["total"].backward()
        durations.append(time.perf_counter() - started)
    gradients = {
        name: float(parameter.grad.norm())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    if not gradients or not all(math.isfinite(value) for value in gradients.values()):
        raise FloatingPointError("Non-finite or missing generator gradients.")
    unchanged = all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    if not unchanged:
        raise AssertionError("Preflight changed model weights.")
    metrics = evaluate_fit(model, items, config.batch_size)
    output.mkdir(parents=True)
    save_grid(
        output / "untrained-predicted-width.png",
        generate_images(model, texts),
        texts,
        "UNTRAINED RANDOM INITIALIZATION | layout check only | no quality claim",
    )
    report = {
        "training_performed": False,
        "optimizer_constructed": False,
        "weights_unchanged": unchanged,
        "config": asdict(config),
        "scope": identity,
        "source_sha256": source_hashes(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "forward_backward_seconds": durations,
        "median_warm_forward_backward_seconds": statistics.median(durations[1:]),
        "estimated_update_compute_seconds": statistics.median(durations[1:]) * config.max_updates,
        "timing_warning": "Fixed batch; excludes optimizer, exports and evaluation. Not an ETA.",
        "gradient_norms": gradients,
        "initial_training_fit_losses": metrics,
        "environment": environment_info(torch.device("cpu")),
        "quality_status": "untrained; execution and gradients only",
    }
    write_json(output / "preflight.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "training_performed",
                    "weights_unchanged",
                    "parameters",
                    "median_warm_forward_backward_seconds",
                    "estimated_update_compute_seconds",
                )
            },
            indent=2,
        )
    )
    return report


def train(config):
    """Only the owner invokes this optimizer-based path."""
    output = Path(config.output)
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing runs are never overwritten.")
    started = time.perf_counter()
    model, items, identity = prepare(config)
    output.mkdir(parents=True)
    write_json(output / "config.json", asdict(config))
    write_json(output / "scope.json", identity)
    sources = source_hashes()
    write_json(output / "source-sha256.json", sources)
    write_json(output / "environment.json", environment_info(torch.device("cpu")))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(config.seed)
    order, cursor = [], 0
    history, best, best_step, completed = [], math.inf, 0, 0

    def record(step):
        nonlocal best, best_step
        metrics = evaluate_fit(model, items, config.batch_size)
        provenance = {
            "writer_id": config.writer_id,
            "records_sha256": identity["records_sha256"],
            "training_ids": identity["diagnostic_ids"],
            "training_texts": identity["diagnostic_texts"],
            "seed": config.seed,
            "optimizer_updates": step,
            "initialization": {
                "kind": "own-checkpoint-warm-start"
                if config.init_checkpoint
                else "random-from-scratch",
                "checkpoint": config.init_checkpoint,
                "sha256": config.init_sha256,
                "optimizer_restored": False,
                "update_count": "Updates in this run; excludes parent checkpoint updates.",
            },
            "purpose": "32-word-training-fit-only",
            "quality_status": "awaiting-human-review",
            "resume_supported": False,
            "config": asdict(config),
            "source_sha256": sources,
        }
        save_generator(output / "last.pt", model, provenance)
        if metrics["total"] < best:
            best, best_step = metrics["total"], step
            save_generator(output / "best.pt", model, provenance)
        history.append(
            {"update": step, "seconds": time.perf_counter() - started, "training_fit": metrics}
        )
        write_json(output / "history.json", history)
        save_grid(
            output / f"predicted-width-{step:04d}.png",
            generate_images(model, identity["diagnostic_texts"]),
            identity["diagnostic_texts"],
            f"GENERATED TRAINING WORDS | predicted widths | update {step} | not quality-scored",
        )
        tqdm.write(
            f"update={step}; fit loss={metrics['total']:.4f}; "
            f"foreground MAE={metrics['foreground_mae']:.4f}; "
            f"edge error={metrics['edge']:.4f}; readability needs human review"
        )

    record(0)
    reason = "update_cap"
    try:
        with tqdm(
            total=config.max_updates,
            desc="Train word generator",
            unit="update",
            ascii=True,
            dynamic_ncols=True,
        ) as progress:
            for step in range(1, config.max_updates + 1):
                if time.perf_counter() - started >= config.max_seconds:
                    reason = "time_cap"
                    break
                if cursor >= len(order):
                    order = torch.randperm(len(items), generator=generator).tolist()
                    cursor = 0
                chosen = order[cursor : cursor + config.batch_size]
                cursor += config.batch_size
                images, widths, texts = image_batch([items[index] for index in chosen])
                model.train()
                optimizer.zero_grad(set_to_none=True)
                result = model(texts, widths)
                losses = paired_loss(result, images, widths)
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError("Non-finite training loss; stop and inspect this run.")
                losses["total"].backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5, error_if_nonfinite=True
                )
                optimizer.step()
                completed = step
                progress.update(1)
                progress.set_postfix(
                    loss=f"{float(losses['total'].detach()):.3f}",
                    grad=f"{float(norm):.2f}",
                    budget_s=max(0, int(config.max_seconds - (time.perf_counter() - started))),
                )
                if step % config.check_every == 0:
                    record(step)
    except KeyboardInterrupt:
        reason = "interrupted"
        # An interrupt may occur inside an update. Preserve the last complete saved check.
        model, checkpoint = load_generator(output / "last.pt")
        completed = checkpoint["provenance"]["optimizer_updates"]
        print("Interrupted. Keeping the last fully saved check; automatic resume is not supported.")
    if reason != "interrupted" and history[-1]["update"] != completed:
        record(completed)
    selected, selected_checkpoint = load_generator(output / "best.pt")
    best_step = selected_checkpoint["provenance"]["optimizer_updates"]
    best = evaluate_fit(selected, items, config.batch_size)["total"]
    export_fit_review(
        selected,
        items,
        output / "best-review",
        seed=config.seed + 1,
        checkpoint_sha256=sha256_file(output / "best.pt"),
    )
    report = {
        "purpose": "generator-training-set-fit; not generalization",
        "stop_reason": reason,
        "saved_optimizer_updates": completed,
        "best_update": best_step,
        "selection": "Lowest full 32-word training loss, including width loss; not visual quality",
        "seconds": time.perf_counter() - started,
        "best_training_fit_loss": best,
        "best_checkpoint_sha256": sha256_file(output / "best.pt"),
        "quality_status": "awaiting-human-review",
        "generalization_established": False,
        "test_set_evaluated": False,
        "training_word_draws_at_saved_update": completed * config.batch_size,
        "budget_note": "Time cap checked between updates. Final evaluation/export adds overhead.",
        "next_action": "Transcribe best-review/blind-predicted-width.png before revealing labels. "
        "Do not launch a larger run based solely on loss.",
    }
    write_json(output / "summary.json", report)
    print(f"Stopped: {reason}. Review {output / 'best-review' / 'blind-predicted-width.png'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/word-generator-fit-32.json"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--preflight-output", type=Path, default=Path("artifacts/word-generator-preflight")
    )
    args = parser.parse_args()
    config = FitConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    if args.preflight:
        preflight(config, args.preflight_output)
    else:
        train(config)


if __name__ == "__main__":
    main()
