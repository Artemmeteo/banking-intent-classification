import pytest
from pydantic import ValidationError

from text_clf.settings import ModelSettings, Settings


def test_nested_environment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT_CLF_API__MAX_CONCURRENCY", "7")
    monkeypatch.setenv("TEXT_CLF_MODEL__URI", "/models/test")
    configured = Settings()
    assert configured.api.max_concurrency == 7
    assert configured.model.uri == "/models/test"


def test_registry_requires_explicit_tracking_uri() -> None:
    with pytest.raises(ValidationError, match="tracking_uri"):
        Settings(model=ModelSettings(source="registry", uri="models:/banking77@champion"))
