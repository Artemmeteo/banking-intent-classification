import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from text_clf.api.app import create_app
from text_clf.domain.banking77 import BANKING77_LABELS
from text_clf.domain.errors import ArtifactValidationError
from text_clf.infrastructure.hf_classifier import HuggingFaceClassifier, PackageModelLoader
from text_clf.infrastructure.model_package import (
    LocalPackageResolver,
    validate_model_package,
)
from text_clf.settings import ModelSettings, Settings


@pytest.mark.integration
def test_package_loads_and_predicts_without_network(
    tiny_model_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    classifier = HuggingFaceClassifier(tiny_model_package, device="cpu")
    prediction = classifier.predict("lost card")[0]
    assert prediction.label in BANKING77_LABELS
    assert 0 <= prediction.confidence <= 1
    assert prediction.model_version == "fixture-v1"


def test_integrity_validation_detects_tampering(tiny_model_package: Path) -> None:
    validate_model_package(tiny_model_package)
    with (tiny_model_package / "labels.json").open("a", encoding="utf-8") as labels:
        labels.write(" ")
    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        validate_model_package(tiny_model_package)


@pytest.mark.integration
def test_real_api_prediction_is_offline(
    tiny_model_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    settings = Settings(model=ModelSettings(uri=str(tiny_model_package), device="cpu"))
    loader = PackageModelLoader(LocalPackageResolver(), device="cpu")
    with TestClient(create_app(settings=settings, model_loader=loader)) as client:
        response = client.post("/v1/predict", json={"text": "lost card"})
        assert response.status_code == 200
        assert response.json()["label"] in BANKING77_LABELS
        assert response.json()["model_version"] == "fixture-v1"
