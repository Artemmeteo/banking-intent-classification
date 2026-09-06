from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from text_clf.domain.entities import Prediction

NormalizedText = Annotated[str, StringConstraints(min_length=2, max_length=1500)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"text": "I lost my credit card yesterday, what should I do?"}]
        },
    )

    text: NormalizedText = Field(description="Customer message to classify")

    @field_validator("text", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return " ".join(value.split())

    @field_validator("text")
    @classmethod
    def validate_text_content(cls, value: str) -> str:
        if not any(character.isalnum() for character in value):
            raise ValueError("text must contain at least one letter or digit")
        return value


class PredictionResponse(BaseModel):
    class_id: int = Field(ge=0, lt=77)
    label: str
    confidence: float = Field(ge=0, le=1)
    model_version: str
    is_unknown: bool

    @classmethod
    def from_entity(cls, prediction: Prediction) -> "PredictionResponse":
        return cls(
            class_id=prediction.class_id,
            label=prediction.label,
            confidence=prediction.confidence,
            model_version=prediction.model_version,
            is_unknown=prediction.is_unknown,
        )


class TopKPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None
    device: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
