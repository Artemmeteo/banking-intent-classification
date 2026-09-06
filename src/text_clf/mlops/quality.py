from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    macro_f1: float
    ece: float
    label_count: int
    p95_inference_latency_ms: float


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    macro_f1_min: float
    max_macro_f1_drop: float
    ece_max: float
    required_label_count: int
    p95_inference_latency_ms_max: float


@dataclass(frozen=True, slots=True)
class QualityDecision:
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def evaluate_quality(
    candidate: QualityMetrics,
    thresholds: QualityThresholds,
    champion: QualityMetrics | None = None,
) -> QualityDecision:
    reasons: list[str] = []
    if candidate.macro_f1 < thresholds.macro_f1_min:
        reasons.append(f"macro_f1 {candidate.macro_f1:.4f} < {thresholds.macro_f1_min:.4f}")
    if champion and candidate.macro_f1 < champion.macro_f1 - thresholds.max_macro_f1_drop:
        reasons.append(f"macro_f1 drop against champion exceeds {thresholds.max_macro_f1_drop:.4f}")
    if candidate.ece > thresholds.ece_max:
        reasons.append(f"ece {candidate.ece:.4f} > {thresholds.ece_max:.4f}")
    if candidate.label_count != thresholds.required_label_count:
        reasons.append(f"label_count {candidate.label_count} != {thresholds.required_label_count}")
    if candidate.p95_inference_latency_ms > thresholds.p95_inference_latency_ms_max:
        reasons.append(
            "p95_inference_latency_ms "
            f"{candidate.p95_inference_latency_ms:.2f} > "
            f"{thresholds.p95_inference_latency_ms_max:.2f}"
        )
    return QualityDecision(passed=not reasons, reasons=tuple(reasons))
