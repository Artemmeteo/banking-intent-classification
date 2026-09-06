from text_clf.application.predict import PredictIntentService
from text_clf.domain.entities import Prediction
from text_clf.domain.errors import InvalidTextError


class FakeClassifier:
    model_version = "fixture-v1"
    device = "fake"

    def __init__(self) -> None:
        self.seen_text: str | None = None

    def predict(self, text: str, top_k: int = 1) -> list[Prediction]:
        self.seen_text = text
        return [Prediction(0, "activate_my_card", 0.9, self.model_version)][:top_k]

    def close(self) -> None:
        pass


def test_service_normalizes_whitespace_without_frameworks() -> None:
    classifier = FakeClassifier()
    result = PredictIntentService(classifier).predict("  activate\n  my card ")
    assert classifier.seen_text == "activate my card"
    assert result[0].label == "activate_my_card"


def test_service_rejects_only_punctuation() -> None:
    classifier = FakeClassifier()
    try:
        PredictIntentService(classifier).predict("?!")
    except InvalidTextError as error:
        assert "letter or digit" in str(error)
    else:
        raise AssertionError("punctuation-only input must fail")
