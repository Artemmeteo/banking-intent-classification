import importlib.util
from pathlib import Path

import pytest

from text_clf.infrastructure.model_package import MlflowPackageResolver, validate_model_package
from text_clf.mlops.registry import register_model_package


@pytest.mark.integration
def test_registration_signature_lineage_and_alias_are_idempotent(
    tiny_model_package: Path, tmp_path: Path
) -> None:
    if importlib.util.find_spec("pandas") is None:
        pytest.skip("full MLflow is intentionally isolated in the training dependency group")
    import mlflow

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    client.create_experiment("integration", artifact_location=(tmp_path / "mlartifacts").as_uri())

    first = register_model_package(
        package_dir=tiny_model_package,
        tracking_uri=tracking_uri,
        experiment_name="integration",
        registered_model_name="banking77-test",
        candidate_alias="candidate",
    )
    second = register_model_package(
        package_dir=tiny_model_package,
        tracking_uri=tracking_uri,
        experiment_name="integration",
        registered_model_name="banking77-test",
        candidate_alias="candidate",
    )

    assert first == second
    version = client.get_model_version_by_alias("banking77-test", "candidate")
    run = client.get_run(version.run_id)
    assert run.data.tags["dataset_revision"] == "1111111111111111111111111111111111111111"
    assert run.data.metrics["label_count"] == 77
    model_info = mlflow.models.get_model_info(version.source)
    assert model_info.signature is not None
    assert "confidence" in str(model_info.signature)
    resolved = MlflowPackageResolver(
        tracking_uri=tracking_uri,
        cache_dir=str(tmp_path / "registry-cache"),
        attempts=1,
    ).resolve("models:/banking77-test@candidate")
    assert validate_model_package(resolved).model_version == "fixture-v1"

    loaded = mlflow.pyfunc.load_model("models:/banking77-test@candidate")
    prediction = loaded.predict({"text": ["lost card"]})
    assert prediction.loc[0, "label"]
    assert 0 <= prediction.loc[0, "confidence"] <= 1
