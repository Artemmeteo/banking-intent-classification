import importlib.util

import pytest


@pytest.mark.integration
def test_dag_parses_without_external_calls() -> None:
    if importlib.util.find_spec("airflow") is None:
        pytest.skip("Airflow is intentionally isolated from the default environment")
    from dags.banking77_pipeline import banking77_pipeline_dag

    assert banking77_pipeline_dag.dag_id == "banking77_train_register_promote"
    assert len(banking77_pipeline_dag.tasks) == 4
