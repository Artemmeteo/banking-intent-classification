from collections.abc import Iterator, Mapping
from contextlib import contextmanager


class NullRunTracker:
    @contextmanager
    def run(self, *, run_name: str, tags: Mapping[str, str]) -> Iterator[None]:
        del run_name, tags
        yield

    def log_params(self, params: Mapping[str, object]) -> None:
        del params

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        del metrics, step

    def log_artifact(self, path: str, *, artifact_path: str | None = None) -> None:
        del path, artifact_path


class MlflowRunTracker:
    def __init__(self, *, tracking_uri: str, experiment_name: str):
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._mlflow = mlflow

    @contextmanager
    def run(self, *, run_name: str, tags: Mapping[str, str]) -> Iterator[None]:
        with self._mlflow.start_run(run_name=run_name, tags=dict(tags)):
            yield

    def log_params(self, params: Mapping[str, object]) -> None:
        self._mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        self._mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, path: str, *, artifact_path: str | None = None) -> None:
        self._mlflow.log_artifacts(path, artifact_path=artifact_path)
