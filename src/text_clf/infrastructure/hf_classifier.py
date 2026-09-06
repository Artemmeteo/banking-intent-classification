import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from text_clf.domain.entities import Prediction
from text_clf.domain.ports import Classifier
from text_clf.infrastructure.model_package import ModelManifest, validate_model_package


class HuggingFaceClassifier(Classifier):
    def __init__(self, package_dir: Path, *, device: str = "auto"):
        self._package_dir = package_dir.resolve()
        self._manifest: ModelManifest = validate_model_package(self._package_dir)
        self._labels = tuple(
            json.loads((self._package_dir / self._manifest.labels_file).read_text(encoding="utf-8"))
        )
        preprocessing = json.loads(
            (self._package_dir / self._manifest.preprocessing_file).read_text(encoding="utf-8")
        )
        self._max_length = int(preprocessing["max_length"])
        self._device = self._select_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._package_dir), local_files_only=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self._package_dir), local_files_only=True
        )
        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def _select_device(configured: str) -> torch.device:
        if configured != "auto":
            return torch.device(configured)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def model_version(self) -> str:
        return self._manifest.model_version

    @property
    def device(self) -> str:
        return str(self._device)

    def predict(self, text: str, top_k: int = 1) -> list[Prediction]:
        encoded: dict[str, Any] = self._tokenizer(
            text,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits / self._manifest.calibration.temperature
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)
            count = min(top_k, len(self._labels))
            confidences, class_ids = torch.topk(probabilities, k=count)
        predictions: list[Prediction] = []
        for confidence_tensor, class_id_tensor in zip(confidences, class_ids, strict=True):
            confidence = float(confidence_tensor.item())
            class_id = int(class_id_tensor.item())
            is_unknown = confidence < self._manifest.unknown_threshold
            predictions.append(
                Prediction(
                    class_id=class_id,
                    label="unknown" if is_unknown else self._labels[class_id],
                    confidence=confidence,
                    model_version=self.model_version,
                    is_unknown=is_unknown,
                )
            )
        return predictions

    def close(self) -> None:
        self._model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class PackageModelLoader:
    def __init__(self, resolver: Any, *, device: str = "auto"):
        self._resolver = resolver
        self._device = device

    def load(self, model_uri: str) -> HuggingFaceClassifier:
        return HuggingFaceClassifier(self._resolver.resolve(model_uri), device=self._device)
