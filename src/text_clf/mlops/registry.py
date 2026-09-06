import json
from pathlib import Path

from text_clf.domain.banking77 import BANKING77_LABELS
from text_clf.domain.errors import QualityGateError
from text_clf.infrastructure.model_package import validate_model_package
from text_clf.mlops.quality import (
    QualityDecision,
    QualityMetrics,
    QualityThresholds,
    evaluate_quality,
)


def register_model_package(
    *,
    package_dir: Path,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    candidate_alias: str,
) -> str:
    import mlflow
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, Schema

    manifest = validate_model_package(package_dir)
    metrics = json.loads((package_dir / "metrics.json").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = mlflow.MlflowClient()
    for existing in client.search_model_versions(f"name = '{registered_model_name}'"):
        existing_run = client.get_run(existing.run_id)
        if existing_run.data.tags.get("model_package_version") == manifest.model_version:
            client.set_registered_model_alias(
                registered_model_name, candidate_alias, existing.version
            )
            return str(existing.version)
    tags = {
        "dataset": manifest.data_lineage.dataset,
        "dataset_revision": manifest.data_lineage.revision,
        "dvc_hash": manifest.data_lineage.dvc_hash,
        "model_revision": manifest.hf_revision,
        "model_package_version": manifest.model_version,
    }
    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "text", required=True)]),
        outputs=Schema(
            [
                ColSpec("long", "class_id"),
                ColSpec("string", "label"),
                ColSpec("double", "confidence"),
                ColSpec("boolean", "is_unknown"),
            ]
        ),
    )

    with mlflow.start_run(run_name=f"register-{manifest.model_version}", tags=tags) as run:
        mlflow.log_metrics(
            {
                "macro_f1": float(metrics["macro_f1"]),
                "ece": float(metrics["ece"]),
                "p95_inference_latency_ms": float(metrics["p95_inference_latency_ms"]),
                "label_count": float(len(BANKING77_LABELS)),
            }
        )
        mlflow.pyfunc.log_model(
            name="model",
            python_model=str(Path(__file__).with_name("pyfunc_model.py")),
            artifacts={"model_package": str(package_dir)},
            signature=signature,
            input_example={"text": ["I lost my card"]},
            registered_model_name=registered_model_name,
            pip_requirements=[
                "mlflow==3.14.0",
                "torch==2.12.1",
                "transformers==5.12.1",
                "text-classification==0.2.0",
            ],
        )
        run_id = run.info.run_id
    versions = client.search_model_versions(
        f"name = '{registered_model_name}' AND run_id = '{run_id}'"
    )
    if len(versions) != 1:
        raise RuntimeError("unable to resolve the registered model version")
    version = str(versions[0].version)
    client.set_registered_model_alias(registered_model_name, candidate_alias, version)
    return version


class MlflowRegistry:
    def __init__(self, *, tracking_uri: str):
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        self._client = mlflow.MlflowClient()

    def _metrics(self, model_name: str, version: str) -> QualityMetrics:
        model_version = self._client.get_model_version(model_name, version)
        run = self._client.get_run(model_version.run_id)
        metrics = run.data.metrics
        return QualityMetrics(
            macro_f1=metrics["macro_f1"],
            ece=metrics["ece"],
            label_count=int(metrics["label_count"]),
            p95_inference_latency_ms=metrics["p95_inference_latency_ms"],
        )

    def promote(
        self,
        *,
        model_name: str,
        candidate_alias: str,
        champion_alias: str,
        thresholds: QualityThresholds,
    ) -> QualityDecision:
        from mlflow.exceptions import MlflowException

        candidate = self._client.get_model_version_by_alias(model_name, candidate_alias)
        try:
            champion = self._client.get_model_version_by_alias(model_name, champion_alias)
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            champion = None
        decision = evaluate_quality(
            self._metrics(model_name, candidate.version),
            thresholds,
            self._metrics(model_name, champion.version) if champion else None,
        )
        if not decision.passed:
            raise QualityGateError("; ".join(decision.reasons))
        if champion:
            self._client.set_model_version_tag(
                model_name,
                candidate.version,
                "previous_champion_version",
                str(champion.version),
            )
        self._client.set_registered_model_alias(model_name, champion_alias, candidate.version)
        return decision

    def rollback(self, *, model_name: str, champion_alias: str) -> str:
        champion = self._client.get_model_version_by_alias(model_name, champion_alias)
        previous = champion.tags.get("previous_champion_version")
        if not previous:
            raise QualityGateError("champion has no recorded previous version")
        self._client.get_model_version(model_name, previous)
        self._client.set_registered_model_alias(model_name, champion_alias, previous)
        return str(previous)
