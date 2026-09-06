from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

from airflow.sdk import dag, task


class TaskResult(TypedDict):
    status: str
    artifact: str


PROJECT_DIR = Path(os.environ.get("TEXT_CLF_PROJECT_DIR", "/opt/text-clf"))
PARAMS = PROJECT_DIR / "params.yaml"


def _environment(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _run(command: list[str], *, timeout_seconds: int) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return completed.stdout.strip()


def _params() -> dict[str, object]:
    import yaml

    return yaml.safe_load(PARAMS.read_text(encoding="utf-8"))


@dag(
    dag_id="banking77_train_register_promote",
    start_date=datetime(2024, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
    dagrun_timeout=timedelta(hours=6),
    tags=["banking77", "mlops"],
)
def banking77_pipeline():  # type: ignore[no-untyped-def]
    @task(execution_timeout=timedelta(minutes=30))
    def reproduce_data() -> TaskResult:
        _run(["dvc", "repro", "download"], timeout_seconds=1800)
        return {"status": "ready", "artifact": "data/processed"}

    @task(execution_timeout=timedelta(hours=4))
    def train(data_result: TaskResult) -> TaskResult:
        if data_result["status"] != "ready":
            raise ValueError("data task XCom contract violated")
        _run(["dvc", "repro", "train"], timeout_seconds=14_400)
        return {"status": "trained", "artifact": "artifacts/candidate-model"}

    @task(execution_timeout=timedelta(minutes=20))
    def register(training_result: TaskResult) -> TaskResult:
        if training_result["status"] != "trained":
            raise ValueError("training task XCom contract violated")
        mlflow_uri = os.environ["TEXT_CLF_MLFLOW__TRACKING_URI"]
        experiment_name = _environment(
            "TEXT_CLF_MLFLOW__EXPERIMENT_NAME", "banking77-classification"
        )
        registered_model_name = _environment(
            "TEXT_CLF_MLFLOW__REGISTERED_MODEL_NAME", "banking77-intent-classifier"
        )
        candidate_alias = _environment("TEXT_CLF_MLFLOW__CANDIDATE_ALIAS", "candidate")
        output = _run(
            [
                "text-clf-register",
                "--model",
                str(PROJECT_DIR / training_result["artifact"]),
                "--tracking-uri",
                mlflow_uri,
                "--experiment",
                experiment_name,
                "--registered-model",
                registered_model_name,
                "--candidate-alias",
                candidate_alias,
            ],
            timeout_seconds=1200,
        )
        return {"status": "registered", "artifact": output.splitlines()[-1]}

    @task(execution_timeout=timedelta(minutes=10))
    def quality_gate(registration_result: TaskResult) -> TaskResult:
        if registration_result["status"] != "registered":
            raise ValueError("registration task XCom contract violated")
        quality = _params()["quality"]
        assert isinstance(quality, dict)
        command = [
            "text-clf-promote",
            "--tracking-uri",
            os.environ["TEXT_CLF_MLFLOW__TRACKING_URI"],
            "--registered-model",
            _environment("TEXT_CLF_MLFLOW__REGISTERED_MODEL_NAME", "banking77-intent-classifier"),
            "--candidate-alias",
            _environment("TEXT_CLF_MLFLOW__CANDIDATE_ALIAS", "candidate"),
            "--champion-alias",
            _environment("TEXT_CLF_MLFLOW__CHAMPION_ALIAS", "champion"),
            "--macro-f1-min",
            str(quality["macro_f1_min"]),
            "--max-macro-f1-drop",
            str(quality["max_macro_f1_drop"]),
            "--ece-max",
            str(quality["ece_max"]),
            "--required-label-count",
            str(quality["required_label_count"]),
            "--p95-latency-max-ms",
            str(quality["p95_inference_latency_ms_max"]),
        ]
        _run(command, timeout_seconds=600)
        return {
            "status": "promoted",
            "artifact": json.dumps({"version": registration_result["artifact"]}),
        }

    quality_gate(register(train(reproduce_data())))


banking77_pipeline_dag = banking77_pipeline()
