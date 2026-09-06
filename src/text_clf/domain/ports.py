from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Protocol

from text_clf.domain.entities import Prediction


class Classifier(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def device(self) -> str: ...

    def predict(self, text: str, top_k: int = 1) -> list[Prediction]: ...

    def close(self) -> None: ...


class ModelLoader(Protocol):
    def load(self, model_uri: str) -> Classifier: ...


class RunTracker(Protocol):
    def run(self, *, run_name: str, tags: Mapping[str, str]) -> AbstractContextManager[None]: ...

    def log_params(self, params: Mapping[str, object]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None: ...

    def log_artifact(self, path: str, *, artifact_path: str | None = None) -> None: ...
