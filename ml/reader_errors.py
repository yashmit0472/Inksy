"""Fixed-weight CTC decoding and error analysis; no language model or dictionary."""

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

from ml.iam import WORD_ALPHABET, ctc_min_steps
from ml.train_word_reader import evaluation_loader
from ml.utils import sha256_file, write_json
from ml.word_reader import greedy_decode, load_reader, reading_metrics


def logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def prefix_beam(logp, beam_size=8):
    """CTC prefix search, blank=0; returns token tuple and merged log probability.

    All tokens are considered. Only prefix count is pruned; no vocabulary, target
    length, language prior, or reference transcription enters decoding.
    """
    if beam_size < 1 or logp.ndim != 2 or not torch.isfinite(logp).all():
        raise ValueError("Expected finite T-by-C log probabilities and a positive beam size.")
    beam = {(): (0.0, -math.inf)}
    for frame in logp.detach().cpu().tolist():
        next_beam = defaultdict(lambda: [-math.inf, -math.inf])
        for prefix, (blank, nonblank) in beam.items():
            total = logadd(blank, nonblank)
            same = next_beam[prefix]
            same[0] = logadd(same[0], total + frame[0])
            for token, probability in enumerate(frame[1:], 1):
                if prefix and token == prefix[-1]:
                    same[1] = logadd(same[1], nonblank + probability)
                    extended = next_beam[prefix + (token,)]
                    extended[1] = logadd(extended[1], blank + probability)
                else:
                    extended = next_beam[prefix + (token,)]
                    extended[1] = logadd(extended[1], total + probability)
        beam = dict(
            sorted(next_beam.items(), key=lambda item: logadd(*item[1]), reverse=True)[:beam_size]
        )
    prefix, probabilities = max(beam.items(), key=lambda item: logadd(*item[1]))
    return prefix, logadd(*probabilities)


def edit_operations(expected, predicted):
    """One minimum-edit alignment; ties prefer substitution, deletion, insertion."""
    table = [[0] * (len(predicted) + 1) for _ in range(len(expected) + 1)]
    for i in range(len(expected) + 1):
        table[i][0] = i
    for j in range(len(predicted) + 1):
        table[0][j] = j
    for i, a in enumerate(expected, 1):
        for j, b in enumerate(predicted, 1):
            table[i][j] = min(
                table[i - 1][j - 1] + (a != b), table[i - 1][j] + 1, table[i][j - 1] + 1
            )
    i, j, operations = len(expected), len(predicted), []
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] + (expected[i - 1] != predicted[j - 1]):
            if expected[i - 1] != predicted[j - 1]:
                operations.append(("substitution", expected[i - 1], predicted[j - 1]))
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            operations.append(("deletion", expected[i - 1], ""))
            i -= 1
        else:
            operations.append(("insertion", "", predicted[j - 1]))
            j -= 1
    return operations


def summarize_errors(rows):
    operations = [
        operation
        for row in rows
        for operation in edit_operations(row["expected"], row["predicted"])
    ]
    by_length = {}
    for name, lower, upper in (("1-3", 1, 3), ("4-6", 4, 6), ("7+", 7, 100)):
        group = [row for row in rows if lower <= len(row["expected"]) <= upper]
        if group:
            by_length[name] = reading_metrics(group)
    return {
        **reading_metrics(rows),
        "edit_counts": dict(Counter(op[0] for op in operations)),
        "top_deleted_characters": dict(
            Counter(op[1] for op in operations if op[0] == "deletion").most_common(15)
        ),
        "top_substitutions": dict(
            Counter(f"{a}->{b}" for kind, a, b in operations if kind == "substitution").most_common(
                15
            )
        ),
        "by_reference_length": by_length,
        "mean_predicted_length": sum(len(r["predicted"]) for r in rows) / len(rows),
        "mean_reference_length": sum(len(r["expected"]) for r in rows) / len(rows),
    }


@torch.inference_mode()
def inspect(model, loader):
    model.eval()
    methods = {name: [] for name in ("greedy", "beam8", "blank_penalty_1")}
    frames, blank_frames, blank_sum, slack = 0, 0, 0.0, []
    for batch_index, batch in enumerate(loader):
        logp, lengths = model(batch["images"], batch["widths"])
        greedy = greedy_decode(logp, lengths)
        penalized = logp.clone()
        penalized[:, :, 0] -= 1.0
        penalized_text = greedy_decode(penalized, lengths)
        beam_text = []
        for index, length in enumerate(lengths.tolist()):
            sample = logp[:length, index]
            tokens, _ = prefix_beam(sample)
            beam_text.append("".join(WORD_ALPHABET[token - 1] for token in tokens))
            frames += length
            blank_frames += int((sample.argmax(-1) == 0).sum())
            blank_sum += float(sample[:, 0].exp().sum())
            slack.append(length - ctc_min_steps(batch["text"][index]))
        for method, predictions in zip(methods, (greedy, beam_text, penalized_text), strict=True):
            methods[method].extend(
                {"id": identifier, "expected": expected, "predicted": prediction}
                for identifier, expected, prediction in zip(
                    batch["ids"], batch["text"], predictions, strict=True
                )
            )
        print(f"Inspected batch {batch_index + 1}/{len(loader)}", flush=True)
    return {
        "methods": {name: summarize_errors(rows) for name, rows in methods.items()},
        "blank_argmax_frame_fraction": blank_frames / frames,
        "mean_blank_probability": blank_sum / frames,
        "minimum_ctc_alignment_slack": min(slack),
        "predictions": methods,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/iam-words"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new error analysis directory.")
    torch.set_num_threads(4)
    checksum = sha256_file(args.checkpoint)
    model, metadata = load_reader(args.checkpoint)
    model.requires_grad_(False)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    report = {}
    for partition in ("seen_validation", "content_validation"):
        loader, identity = evaluation_loader(args.prepared, partition, 16, 16)
        if metadata["records_sha256"] != identity["records_sha256"]:
            raise ValueError("Checkpoint and data identity differ.")
        report[partition] = {"identity": identity, **inspect(model, loader)}
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    assert sha256_file(args.checkpoint) == checksum
    write_json(
        args.output / "errors.json",
        {
            "checkpoint_sha256": checksum,
            "results": report,
            "source_sha256": sha256_file(Path(__file__)),
            "training_performed": False,
            "weights_unchanged": True,
            "test_set_evaluated": False,
            "warning": (
                "Development diagnostics only. No language model; beam search is approximate. "
                "Blank penalty is an ablation, not a selected production setting."
            ),
        },
    )
    for partition, result in report.items():
        print(
            partition,
            {
                name: {
                    key: value[key]
                    for key in (
                        "character_error_rate",
                        "exact_word_accuracy",
                        "edit_counts",
                        "mean_predicted_length",
                    )
                }
                for name, value in result["methods"].items()
            },
        )


if __name__ == "__main__":
    main()
