import pytest

from text_clf.mlops.quality import QualityMetrics, QualityThresholds, evaluate_quality

THRESHOLDS = QualityThresholds(
    macro_f1_min=0.8,
    max_macro_f1_drop=0.02,
    ece_max=0.08,
    required_label_count=77,
    p95_inference_latency_ms_max=100,
)


def metrics(**overrides: float | int) -> QualityMetrics:
    values: dict[str, float | int] = {
        "macro_f1": 0.85,
        "ece": 0.04,
        "label_count": 77,
        "p95_inference_latency_ms": 50,
    }
    values.update(overrides)
    return QualityMetrics(**values)  # type: ignore[arg-type]


def test_quality_gate_passes_good_candidate() -> None:
    assert evaluate_quality(metrics(), THRESHOLDS, metrics(macro_f1=0.86)).passed


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"macro_f1": 0.79}, "macro_f1"),
        ({"ece": 0.09}, "ece"),
        ({"label_count": 76}, "label_count"),
        ({"p95_inference_latency_ms": 101}, "p95"),
    ],
)
def test_quality_gate_blocks_bad_candidate(overrides: dict[str, float | int], reason: str) -> None:
    decision = evaluate_quality(metrics(**overrides), THRESHOLDS)
    assert not decision.passed
    assert reason in decision.reasons[0]


def test_quality_gate_blocks_regression_against_champion() -> None:
    decision = evaluate_quality(metrics(macro_f1=0.85), THRESHOLDS, metrics(macro_f1=0.88))
    assert not decision.passed
    assert "champion" in decision.reasons[0]
