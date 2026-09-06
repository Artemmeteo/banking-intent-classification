import hashlib
import json
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from text_clf.domain.banking77 import BANKING77_LABELS
from text_clf.domain.errors import ArtifactValidationError


class DataLineage(BaseModel):
    dataset: str
    revision: str
    dvc_hash: str


class CalibrationMetadata(BaseModel):
    temperature: float = Field(gt=0)
    ece: float = Field(ge=0, le=1)


class ModelManifest(BaseModel):
    schema_version: Literal["1"] = "1"
    model_name: str
    model_version: str
    framework: Literal["transformers"] = "transformers"
    hf_revision: str
    created_at: datetime
    labels_file: str = "labels.json"
    preprocessing_file: str = "preprocessing.json"
    data_lineage: DataLineage
    calibration: CalibrationMetadata
    unknown_threshold: float = Field(ge=0, le=1)
    files: dict[str, str]

    @field_validator("hf_revision")
    @classmethod
    def immutable_revision(cls, value: str) -> str:
        if value.strip().lower() in {"", "main", "master", "latest"}:
            raise ValueError("hf_revision must be immutable, not a moving branch")
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model_package_metadata(
    package_dir: Path,
    *,
    model_name: str,
    model_version: str,
    hf_revision: str,
    dataset_revision: str,
    dvc_hash: str,
    temperature: float,
    ece: float,
    unknown_threshold: float,
    max_length: int,
) -> ModelManifest:
    package_dir = package_dir.resolve()
    package_dir.mkdir(parents=True, exist_ok=True)
    labels_path = package_dir / "labels.json"
    preprocessing_path = package_dir / "preprocessing.json"
    labels_path.write_text(
        json.dumps(list(BANKING77_LABELS), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    preprocessing_path.write_text(
        json.dumps(
            {
                "normalization": "collapse_whitespace_and_strip",
                "max_length": max_length,
                "truncation": True,
                "padding": "max_length",
                "tokenizer_location": ".",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    files = {
        str(path.relative_to(package_dir)): sha256_file(path)
        for path in sorted(package_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = ModelManifest(
        model_name=model_name,
        model_version=model_version,
        hf_revision=hf_revision,
        created_at=datetime.now(UTC),
        data_lineage=DataLineage(
            dataset="mteb/banking77", revision=dataset_revision, dvc_hash=dvc_hash
        ),
        calibration=CalibrationMetadata(temperature=temperature, ece=ece),
        unknown_threshold=unknown_threshold,
        files=files,
    )
    (package_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_model_package(package_dir: Path) -> ModelManifest:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactValidationError(f"missing model manifest: {manifest_path}")
    try:
        manifest = ModelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ArtifactValidationError(f"invalid model manifest: {error}") from error

    for relative_path, expected_digest in manifest.files.items():
        path = (package_dir / relative_path).resolve()
        if package_dir not in path.parents or not path.is_file() or path.is_symlink():
            raise ArtifactValidationError(f"unsafe or missing package file: {relative_path}")
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise ArtifactValidationError(f"checksum mismatch for {relative_path}")

    try:
        labels = json.loads((package_dir / manifest.labels_file).read_text(encoding="utf-8"))
        preprocessing = json.loads(
            (package_dir / manifest.preprocessing_file).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"invalid package metadata: {error}") from error
    if tuple(labels) != BANKING77_LABELS:
        raise ArtifactValidationError("model package must contain the canonical 77 labels")
    if preprocessing.get("tokenizer_location") != ".":
        raise ArtifactValidationError("tokenizer must be materialized inside the model package")
    if not any((package_dir / name).is_file() for name in ("tokenizer.json", "vocab.txt")):
        raise ArtifactValidationError("model package does not contain tokenizer files")
    if not any(
        (package_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    ):
        raise ArtifactValidationError("model package does not contain model weights")
    return manifest


class LocalPackageResolver:
    def resolve(self, uri: str) -> Path:
        package_dir = Path(uri).expanduser().resolve()
        if not package_dir.is_dir():
            raise ArtifactValidationError(f"local model package does not exist: {package_dir}")
        return package_dir


class MlflowPackageResolver:
    def __init__(self, tracking_uri: str, cache_dir: str, attempts: int = 3):
        self._tracking_uri = tracking_uri
        self._cache_dir = Path(cache_dir)
        self._attempts = attempts

    def resolve(self, uri: str) -> Path:
        import mlflow

        mlflow.set_tracking_uri(self._tracking_uri)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        def download() -> str:
            return str(
                mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(self._cache_dir))
            )

        downloaded = self._bounded_retry(download)
        root = Path(downloaded).resolve()
        candidates = (
            root / "artifacts" / "model_package",
            root / "artifacts" / "model-package",
            root / "model_package",
            root / "model-package",
            root,
        )
        for candidate in candidates:
            if (candidate / "manifest.json").is_file():
                return candidate
        artifact_root = root / "artifacts"
        manifests = list(artifact_root.rglob("manifest.json")) if artifact_root.is_dir() else []
        if len(manifests) == 1:
            return manifests[0].parent
        raise ArtifactValidationError(f"registry artifact does not contain a model package: {uri}")

    def _bounded_retry(self, operation: Callable[[], str]) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self._attempts + 1):
            try:
                return operation()
            except Exception as error:  # external boundary; re-raised after a bounded delay
                last_error = error
                if attempt < self._attempts:
                    time.sleep(min(0.25 * (2 ** (attempt - 1)), 1.0))
        raise ArtifactValidationError(
            f"model registry download failed: {last_error}"
        ) from last_error


def copy_package(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    shutil.copytree(source, destination)
