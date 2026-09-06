from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    min_text_length: int = Field(default=2, ge=1)
    max_text_length: int = Field(default=1500, ge=2, le=20_000)
    max_top_k: int = Field(default=10, ge=1, le=77)
    max_concurrency: int = Field(default=4, ge=1, le=256)
    shutdown_timeout_seconds: float = Field(default=20.0, gt=0)


class ModelSettings(BaseModel):
    source: Literal["local", "registry"] = "local"
    uri: str = "artifacts/model-package"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    local_files_only: bool = True
    registry_cache_dir: str = "/tmp/text-clf-model-cache"
    registry_download_attempts: int = Field(default=3, ge=1, le=5)


class MlflowSettings(BaseModel):
    tracking_uri: str | None = None
    experiment_name: str = "banking77-classification"
    registered_model_name: str = "banking77-intent-classifier"
    candidate_alias: str = "candidate"
    champion_alias: str = "champion"


class ObservabilitySettings(BaseModel):
    log_level: str = "INFO"
    json_logs: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TEXT_CLF_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: str = "development"
    api: ApiSettings = Field(default_factory=ApiSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    mlflow: MlflowSettings = Field(default_factory=MlflowSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @model_validator(mode="after")
    def validate_cross_field_settings(self) -> "Settings":
        if self.api.min_text_length > self.api.max_text_length:
            raise ValueError("api.min_text_length must not exceed api.max_text_length")
        if self.model.source == "registry":
            if not self.model.uri.startswith("models:/"):
                raise ValueError("registry model source requires a models:/ URI")
            if not self.mlflow.tracking_uri:
                raise ValueError("registry model source requires mlflow.tracking_uri")
        elif self.model.uri.startswith("models:/"):
            raise ValueError("local model source requires a filesystem path")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
