import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from text_clf.api.app import create_app
from text_clf.domain.entities import Prediction
from text_clf.settings import ApiSettings, ModelSettings, Settings


class FakeClassifier:
    model_version = "fixture-v1"
    device = "cpu"

    def __init__(self, blocker: threading.Event | None = None) -> None:
        self.blocker = blocker
        self.entered = threading.Event()
        self.closed = False

    def predict(self, text: str, top_k: int = 1) -> list[Prediction]:
        self.entered.set()
        if self.blocker is not None:
            self.blocker.wait(timeout=2)
        return [Prediction(41, "lost_or_stolen_card", 0.75, self.model_version)] * top_k

    def close(self) -> None:
        self.closed = True


class FakeLoader:
    def __init__(self, classifier: FakeClassifier | None = None, error: Exception | None = None):
        self.classifier = classifier
        self.error = error

    def load(self, model_uri: str) -> FakeClassifier:
        assert model_uri == "/fixture/model"
        if self.error:
            raise self.error
        assert self.classifier is not None
        return self.classifier


def settings(*, concurrency: int = 2) -> Settings:
    return Settings(
        api=ApiSettings(max_concurrency=concurrency),
        model=ModelSettings(uri="/fixture/model"),
    )


def test_prediction_contract_and_request_id() -> None:
    classifier = FakeClassifier()
    with TestClient(create_app(settings=settings(), model_loader=FakeLoader(classifier))) as client:
        response = client.post(
            "/v1/predict",
            json={"text": "  lost  my\ncard "},
            headers={"x-request-id": "contract-1"},
        )
        assert response.status_code == 200
        assert response.headers["x-request-id"] == "contract-1"
        assert response.json() == {
            "class_id": 41,
            "label": "lost_or_stolen_card",
            "confidence": 0.75,
            "model_version": "fixture-v1",
            "is_unknown": False,
        }
        assert client.get("/live").status_code == 200
        assert client.get("/ready").json()["status"] == "ready"
        assert "text_clf_http_requests_total" in client.get("/metrics").text
    assert classifier.closed


def test_schema_examples_and_confidence_range() -> None:
    app = create_app(settings=settings(), model_loader=FakeLoader(FakeClassifier()))
    schema = app.openapi()
    request_schema = schema["components"]["schemas"]["PredictRequest"]
    response_schema = schema["components"]["schemas"]["PredictionResponse"]
    assert request_schema["examples"][0]["text"]
    confidence = response_schema["properties"]["confidence"]
    assert confidence["minimum"] == 0
    assert confidence["maximum"] == 1


def test_validation_boundaries_punctuation_and_extra_fields() -> None:
    with TestClient(
        create_app(settings=settings(), model_loader=FakeLoader(FakeClassifier()))
    ) as client:
        assert client.post("/v1/predict", json={"text": "x"}).status_code == 422
        assert client.post("/v1/predict", json={"text": "?!"}).status_code == 422
        assert client.post("/v1/predict", json={"text": "ok", "extra": 1}).status_code == 422
        assert client.post("/v1/predict", json={"text": "x" * 1500}).status_code == 200
        assert client.post("/v1/predict", json={"text": "x" * 1501}).status_code == 422


def test_not_ready_is_503_but_live_is_200() -> None:
    with TestClient(
        create_app(settings=settings(), model_loader=FakeLoader(error=RuntimeError("broken")))
    ) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/v1/predict", json={"text": "hello"}).status_code == 503


def test_saturation_returns_429() -> None:
    release = threading.Event()
    classifier = FakeClassifier(release)
    with TestClient(
        create_app(settings=settings(concurrency=1), model_loader=FakeLoader(classifier))
    ) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(client.post, "/v1/predict", json={"text": "first"})
            assert classifier.entered.wait(timeout=1)
            second = client.post("/v1/predict", json={"text": "second"})
            release.set()
            assert first.result(timeout=2).status_code == 200
        assert second.status_code == 429
