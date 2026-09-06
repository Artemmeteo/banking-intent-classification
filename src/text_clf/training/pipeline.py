import json
import math
import random
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from text_clf.domain.banking77 import BANKING77_LABELS
from text_clf.domain.ports import RunTracker
from text_clf.infrastructure.model_package import write_model_package_metadata
from text_clf.models.hf_models import get_hf_model_and_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _macro_f1(predictions: list[int], labels: list[int]) -> float:
    scores: list[float] = []
    for class_id in range(len(BANKING77_LABELS)):
        pairs = list(zip(predictions, labels, strict=True))
        true_positive = sum(p == class_id and y == class_id for p, y in pairs)
        false_positive = sum(p == class_id and y != class_id for p, y in pairs)
        false_negative = sum(p != class_id and y == class_id for p, y in pairs)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return statistics.fmean(scores)


def _ece(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    confidences, predictions = probabilities.max(dim=1)
    if labels.numel() == 0:
        return math.nan
    error = torch.tensor(0.0)
    for lower in torch.linspace(0, 1, bins + 1)[:-1]:
        upper = lower + 1 / bins
        selected = (confidences > lower) & (confidences <= upper)
        if selected.any():
            accuracy = (predictions[selected] == labels[selected]).float().mean()
            error += selected.float().mean() * torch.abs(accuracy - confidences[selected].mean())
    return float(error.item())


def _select_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    candidates = [0.5 + 0.1 * index for index in range(26)]
    losses = [
        float(torch.nn.functional.cross_entropy(logits / temperature, labels).item())
        for temperature in candidates
    ]
    return candidates[losses.index(min(losses))]


class _TokenizedDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    def __init__(self, dataset: Any, tokenizer: Any, max_length: int):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self._dataset[index]
        encoded = self._tokenizer(
            row["text"],
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def train_model(
    *,
    data_dir: Path,
    output_dir: Path,
    base_model: str,
    model_revision: str,
    dataset_revision: str,
    dvc_hash: str,
    model_version: str,
    tracker: RunTracker,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
    unknown_threshold: float,
    overwrite: bool = False,
) -> dict[str, float]:
    from datasets import load_from_disk

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_from_disk(str(data_dir))
    model, tokenizer = get_hf_model_and_tokenizer(
        base_model, revision=model_revision, num_labels=len(BANKING77_LABELS)
    )
    train_loader = DataLoader(
        _TokenizedDataset(dataset["train"], tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        _TokenizedDataset(dataset["validation"], tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    tags = {
        "dataset": "mteb/banking77",
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "dvc_hash": dvc_hash,
    }
    with tracker.run(run_name=model_version, tags=tags):
        tracker.log_params(
            {
                "base_model": base_model,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "seed": seed,
            }
        )
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                output = model(**batch)
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(output.loss.item())
            tracker.log_metrics({"train_loss": total_loss / max(len(train_loader), 1)}, step=epoch)

        model.eval()
        logits_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []
        latencies_ms: list[float] = []
        with torch.inference_mode():
            for batch in validation_loader:
                labels = batch.pop("labels")
                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                started = time.perf_counter()
                logits_batches.append(model(**batch).logits.cpu())
                latencies_ms.append((time.perf_counter() - started) * 1000 / len(labels))
                label_batches.append(labels)
        logits = torch.cat(logits_batches)
        labels = torch.cat(label_batches)
        temperature = _select_temperature(logits, labels)
        probabilities = torch.softmax(logits / temperature, dim=1)
        predictions = probabilities.argmax(dim=1).tolist()
        expected = labels.tolist()
        sorted_latency = sorted(latencies_ms)
        p95_index = min(int(0.95 * len(sorted_latency)), len(sorted_latency) - 1)
        metrics = {
            "macro_f1": _macro_f1(predictions, expected),
            "ece": _ece(probabilities, labels),
            "p95_inference_latency_ms": sorted_latency[p95_index],
            "temperature": temperature,
        }
        tracker.log_metrics(metrics)

        output_parent = output_dir.resolve().parent
        output_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_parent, prefix="model-package-") as temp_dir:
            package_dir = Path(temp_dir)
            model.save_pretrained(package_dir, safe_serialization=True)
            tokenizer.save_pretrained(package_dir)
            (package_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            write_model_package_metadata(
                package_dir,
                model_name=base_model,
                model_version=model_version,
                hf_revision=model_revision,
                dataset_revision=dataset_revision,
                dvc_hash=dvc_hash,
                temperature=temperature,
                ece=metrics["ece"],
                unknown_threshold=unknown_threshold,
                max_length=max_length,
            )
            if output_dir.exists():
                if not overwrite:
                    raise FileExistsError(f"output exists; pass --overwrite: {output_dir}")
                shutil.rmtree(output_dir)
            shutil.copytree(package_dir, output_dir)
        tracker.log_artifact(str(output_dir), artifact_path="model_package")
    return metrics
