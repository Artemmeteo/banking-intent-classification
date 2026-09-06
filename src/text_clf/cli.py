import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from text_clf.infrastructure.model_package import (
    LocalPackageResolver,
    MlflowPackageResolver,
    copy_package,
    validate_model_package,
)


def _directory_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(directory)).encode())
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def data_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and deterministically split Banking77")
    parser.add_argument("command", choices=["download"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    from text_clf.data.prepare_data import prepare_and_save_data

    prepare_and_save_data(
        output_dir=args.output,
        dataset_name=args.dataset,
        revision=args.revision,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and package a Banking77 classifier")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unknown-threshold", type=float, default=0.45)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default="banking77-classification")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    from text_clf.mlops.tracking import MlflowRunTracker, NullRunTracker
    from text_clf.training.pipeline import train_model

    tracker = (
        MlflowRunTracker(tracking_uri=args.tracking_uri, experiment_name=args.experiment)
        if args.tracking_uri
        else NullRunTracker()
    )
    metrics = train_model(
        data_dir=args.data,
        output_dir=args.output,
        base_model=args.base_model,
        model_revision=args.model_revision,
        dataset_revision=args.dataset_revision,
        dvc_hash=_directory_digest(args.data),
        model_version=args.model_version,
        tracker=tracker,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        unknown_threshold=args.unknown_threshold,
        overwrite=args.overwrite,
    )
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


def evaluate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local model package")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from text_clf.training.evaluate import evaluate_model_package

    print(
        json.dumps(
            evaluate_model_package(
                package_dir=args.model, data_dir=args.data, output_path=args.output
            ),
            sort_keys=True,
        )
    )


def register_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Register a model package as the candidate")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--registered-model", required=True)
    parser.add_argument("--candidate-alias", default="candidate")
    args = parser.parse_args(argv)
    from text_clf.mlops.registry import register_model_package

    version = register_model_package(
        package_dir=args.model,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        registered_model_name=args.registered_model,
        candidate_alias=args.candidate_alias,
    )
    print(version)


def _quality_thresholds(args: Any) -> Any:
    from text_clf.mlops.quality import QualityThresholds

    return QualityThresholds(
        macro_f1_min=args.macro_f1_min,
        max_macro_f1_drop=args.max_macro_f1_drop,
        ece_max=args.ece_max,
        required_label_count=args.required_label_count,
        p95_inference_latency_ms_max=args.p95_latency_max_ms,
    )


def promote_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quality-gate and promote the candidate alias")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--registered-model", required=True)
    parser.add_argument("--candidate-alias", default="candidate")
    parser.add_argument("--champion-alias", default="champion")
    parser.add_argument("--macro-f1-min", type=float, required=True)
    parser.add_argument("--max-macro-f1-drop", type=float, required=True)
    parser.add_argument("--ece-max", type=float, required=True)
    parser.add_argument("--required-label-count", type=int, required=True)
    parser.add_argument("--p95-latency-max-ms", type=float, required=True)
    args = parser.parse_args(argv)
    from text_clf.mlops.registry import MlflowRegistry

    registry = MlflowRegistry(tracking_uri=args.tracking_uri)
    registry.promote(
        model_name=args.registered_model,
        candidate_alias=args.candidate_alias,
        champion_alias=args.champion_alias,
        thresholds=_quality_thresholds(args),
    )
    print("promoted")


def rollback_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rollback champion to its recorded predecessor")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--registered-model", required=True)
    parser.add_argument("--champion-alias", default="champion")
    args = parser.parse_args(argv)
    from text_clf.mlops.registry import MlflowRegistry

    version = MlflowRegistry(tracking_uri=args.tracking_uri).rollback(
        model_name=args.registered_model, champion_alias=args.champion_alias
    )
    print(version)


def materialize_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Materialize and verify a serving model package")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--cache-dir", default="/tmp/text-clf-model-cache")
    args = parser.parse_args(argv)
    resolver: LocalPackageResolver | MlflowPackageResolver
    if args.source.startswith("models:/"):
        if not args.tracking_uri:
            parser.error("--tracking-uri is required for models:/ URIs")
        resolver = MlflowPackageResolver(args.tracking_uri, args.cache_dir)
    else:
        resolver = LocalPackageResolver()
    source = resolver.resolve(args.source)
    validate_model_package(source)
    copy_package(source, args.destination)
    validate_model_package(args.destination)


def serve_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the HTTP API")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    import uvicorn

    from text_clf.settings import get_settings

    settings = get_settings()
    uvicorn.run(
        "text_clf.api.app:app",
        host=args.host or settings.api.host,
        port=args.port or settings.api.port,
        timeout_graceful_shutdown=int(settings.api.shutdown_timeout_seconds),
    )


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    help_text = (
        "usage: text-clf {data,train,evaluate,register,promote,rollback,materialize,serve} ..."
    )
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(help_text)
        return
    command, *rest = arguments
    commands = {
        "data": data_main,
        "train": train_main,
        "evaluate": evaluate_main,
        "register": register_main,
        "promote": promote_main,
        "rollback": rollback_main,
        "materialize": materialize_main,
        "serve": serve_main,
    }
    try:
        handler = commands[command]
    except KeyError as error:
        raise SystemExit(f"unknown command: {command}") from error
    handler(rest)


if __name__ == "__main__":
    main()
