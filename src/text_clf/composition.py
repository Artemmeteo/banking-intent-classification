from text_clf.domain.ports import ModelLoader
from text_clf.infrastructure.hf_classifier import PackageModelLoader
from text_clf.infrastructure.model_package import LocalPackageResolver, MlflowPackageResolver
from text_clf.settings import Settings


def create_model_loader(settings: Settings) -> ModelLoader:
    resolver: LocalPackageResolver | MlflowPackageResolver
    if settings.model.source == "registry":
        tracking_uri = settings.mlflow.tracking_uri
        if tracking_uri is None:  # already guarded by Settings; keeps this boundary explicit
            raise ValueError("MLflow tracking URI is required for registry serving")
        resolver = MlflowPackageResolver(
            tracking_uri=tracking_uri,
            cache_dir=settings.model.registry_cache_dir,
            attempts=settings.model.registry_download_attempts,
        )
    else:
        resolver = LocalPackageResolver()
    return PackageModelLoader(resolver, device=settings.model.device)
