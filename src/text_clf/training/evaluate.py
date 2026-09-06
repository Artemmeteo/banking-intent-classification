import json
import statistics
import time
from pathlib import Path
from typing import Any

from text_clf.infrastructure.hf_classifier import HuggingFaceClassifier
from text_clf.training.pipeline import _macro_f1


def evaluate_model_package(
    *, package_dir: Path, data_dir: Path, output_path: Path
) -> dict[str, float]:
    from datasets import load_from_disk

    classifier = HuggingFaceClassifier(package_dir)
    dataset: Any = load_from_disk(str(data_dir))["test"]
    predictions: list[int] = []
    labels: list[int] = []
    latencies: list[float] = []
    for row in dataset:
        started = time.perf_counter()
        prediction = classifier.predict(str(row["text"]))[0]
        latencies.append((time.perf_counter() - started) * 1000)
        predictions.append(prediction.class_id)
        labels.append(int(row["label"]))
    classifier.close()
    ordered_latency = sorted(latencies)
    p95_index = min(int(0.95 * len(ordered_latency)), len(ordered_latency) - 1)
    metrics = {
        "test_macro_f1": _macro_f1(predictions, labels),
        "test_p95_inference_latency_ms": ordered_latency[p95_index],
        "test_mean_inference_latency_ms": statistics.fmean(latencies),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
