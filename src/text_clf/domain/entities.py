from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prediction:
    class_id: int
    label: str
    confidence: float
    model_version: str
    is_unknown: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
