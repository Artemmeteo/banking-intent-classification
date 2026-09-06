import re

from text_clf.domain.entities import Prediction
from text_clf.domain.errors import InvalidTextError
from text_clf.domain.ports import Classifier


class PredictIntentService:
    def __init__(self, classifier: Classifier, *, min_length: int = 2, max_length: int = 1500):
        self._classifier = classifier
        self._min_length = min_length
        self._max_length = max_length

    @property
    def model_version(self) -> str:
        return self._classifier.model_version

    @property
    def device(self) -> str:
        return self._classifier.device

    def close(self) -> None:
        self._classifier.close()

    def predict(self, text: str, *, top_k: int = 1) -> list[Prediction]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not self._min_length <= len(normalized) <= self._max_length:
            raise InvalidTextError(
                f"normalized text length must be in [{self._min_length}, {self._max_length}]"
            )
        if not any(character.isalnum() for character in normalized):
            raise InvalidTextError("text must contain at least one letter or digit")
        return self._classifier.predict(normalized, top_k=top_k)
