import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from ml.characters import ALPHABET, CONTRACT_VERSION, PREPROCESSING_VERSION
from ml.checkpoint import save_checkpoint
from ml.model import ConditionalVAE, ModelConfig
from ml.recognizer import (
    RECOGNIZER_FORMAT,
    CharacterRecognizer,
    confusion_metrics,
    load_recognizer,
)
from ml.score_generation import score_generation
from ml.train_recognizer import evaluator_partitions, recognizer_epoch


def test_exact_and_case_insensitive_scores_are_distinct():
    confusion = torch.zeros(62, 62, dtype=torch.long)
    confusion[0, 0], confusion[0, 24] = 3, 1  # 0 misread as O is still wrong ignoring case.
    confusion[10, 10], confusion[10, 36] = 2, 2  # A/a
    confusion[36, 10], confusion[36, 36] = 1, 1
    metrics = confusion_metrics(confusion)
    assert metrics["examples"] == 10
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["case_insensitive_accuracy"] == pytest.approx(0.9)
    assert metrics["macro_accuracy"] == pytest.approx((0.75 + 0.5 + 0.5) / 3)
    assert metrics["classes_evaluated"] == 3
    assert metrics["per_class"]["B"]["accuracy"] is None


def test_empty_confusion_cannot_report_accuracy():
    with pytest.raises(ValueError, match="empty"):
        confusion_metrics(torch.zeros(62, 62))


def test_recognizer_selection_and_audit_do_not_overlap():
    labels = torch.arange(62).repeat_interleave(10)
    data = TensorDataset(torch.zeros(620, 1, 28, 28), labels)
    data.targets = labels
    validation = Subset(data, list(range(620)))
    selection, audit, identity = evaluator_partitions(data, validation)
    assert not set(selection.indices) & set(audit.indices)
    assert set(selection.indices) | set(audit.indices) == set(range(620))
    assert torch.bincount(labels[audit.indices]).tolist() == [5] * 62
    second_selection, second_audit, second_identity = evaluator_partitions(data, validation)
    assert selection.indices == second_selection.indices
    assert audit.indices == second_audit.indices
    assert identity == second_identity


def test_recognizer_validation_is_inference_only():
    model = CharacterRecognizer()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    dataset = TensorDataset(torch.rand(3, 1, 28, 28), torch.tensor([0, 10, 61]))
    metrics = recognizer_epoch(
        model, DataLoader(dataset, batch_size=2), torch.device("cpu"), show_progress=False
    )
    assert metrics["examples"] == 3
    assert metrics["loss"] > 0
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


def recognizer_payload(model, status="smoke-only"):
    return {
        "format": RECOGNIZER_FORMAT,
        "alphabet": ALPHABET,
        "contract_version": CONTRACT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "state_dict": model.state_dict(),
        "quality_status": status,
        "split_sha256": "fixture",
    }


def test_recognizer_checkpoint_contract_is_separate_from_generator(tmp_path):
    path = tmp_path / "recognizer.pt"
    payload = recognizer_payload(CharacterRecognizer())
    torch.save(payload, path)
    model, metadata = load_recognizer(path)
    assert not model.training
    assert metadata["format"] == RECOGNIZER_FORMAT
    payload["preprocessing_version"] = "incorrect"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="preprocessing"):
        load_recognizer(path)


def test_unaudited_recognizer_cannot_produce_generation_score(tmp_path):
    generator = tmp_path / "generator.pt"
    recognizer = tmp_path / "recognizer.pt"
    model = ConditionalVAE(ModelConfig(latent_dim=4, label_dim=3, base_channels=4))
    save_checkpoint(generator, model, split_sha256="fixture")
    torch.save(recognizer_payload(CharacterRecognizer()), recognizer)
    with pytest.raises(ValueError, match="audit"):
        score_generation(generator, recognizer, tmp_path / "scores", samples_per_class=20)
    assert not (tmp_path / "scores").exists()


def test_scoring_exports_known_exact_and_case_confusions_without_training(tmp_path, monkeypatch):
    """Synthetic inference fixtures test reporting; no model weights are trained or evaluated."""
    import csv
    import importlib

    module = importlib.import_module("ml.score_generation")
    generator_path, recognizer_path = tmp_path / "generator.pt", tmp_path / "recognizer.pt"
    generator_path.write_bytes(b"synthetic generator identity")
    recognizer_path.write_bytes(b"synthetic recognizer identity")
    real_audit = confusion_metrics(torch.eye(62, dtype=torch.long))
    metadata = {
        "quality_status": "audited-research-scorer",
        "split_sha256": "fixture",
        "real_audit": real_audit,
        "audit_partition": {"fixture": True},
    }

    class DeterministicFixtureRecognizer:
        def __call__(self, images):
            labels = images[:, 0, 0, 0].long()
            # Deliberately confuse every uppercase A with lowercase a.
            predictions = torch.where(labels == 10, 36, labels)
            logits = torch.full((len(images), 62), -10.0)
            logits[torch.arange(len(images)), predictions] = 10
            return logits

    def fixture_glyphs(model, characters, seed):
        images = torch.zeros(len(characters), 1, 28, 28)
        images[:, 0, 0, 0] = ALPHABET.index(characters[0])
        return images

    monkeypatch.setattr(
        module, "load_checkpoint", lambda *args: (object(), {"split_sha256": "fixture"})
    )
    monkeypatch.setattr(
        module, "load_recognizer", lambda *args: (DeterministicFixtureRecognizer(), metadata)
    )
    monkeypatch.setattr(module, "sample_glyphs", fixture_glyphs)
    output = tmp_path / "scores"
    report = module.score_generation(generator_path, recognizer_path, output, samples_per_class=20)
    agreement = report["generated_agreement"]
    assert agreement["examples"] == 1240
    assert agreement["accuracy"] == pytest.approx(61 / 62)
    assert agreement["case_insensitive_accuracy"] == 1
    assert agreement["per_class"]["A"]["accuracy"] == 0
    assert report["largest_confusions"][0] == {"requested": "A", "predicted": "a", "count": 20}
    assert report["test_set_evaluated"] is False
    with (output / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 1240
