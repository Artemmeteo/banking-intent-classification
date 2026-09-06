import json
from pathlib import Path
from typing import Any

from text_clf.data.contracts import validate_split_contract
from text_clf.domain.banking77 import BANKING77_LABELS


def prepare_and_save_data(
    *,
    output_dir: Path,
    dataset_name: str,
    revision: str,
    seed: int = 42,
    validation_fraction: float = 0.1,
) -> None:
    if revision.lower() in {"main", "master", "latest", ""}:
        raise ValueError("dataset revision must be an immutable commit")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty data directory: {output_dir}")

    from datasets import ClassLabel, DatasetDict, load_dataset

    raw: Any = load_dataset(dataset_name, revision=revision)
    if "train" not in raw or "test" not in raw:
        raise ValueError("dataset must contain train and test splits")

    def normalize(example: dict[str, Any]) -> dict[str, Any]:
        label_id = int(example["label"])
        label_text = str(example.get("label_text", BANKING77_LABELS[label_id]))
        return {"text": str(example["text"]), "label": label_id, "label_text": label_text}

    normalized_train = raw["train"].map(normalize, remove_columns=raw["train"].column_names)
    normalized_test = raw["test"].map(normalize, remove_columns=raw["test"].column_names)
    label_feature = ClassLabel(names=list(BANKING77_LABELS))
    normalized_train = normalized_train.cast_column("label", label_feature)
    normalized_test = normalized_test.cast_column("label", label_feature)
    split = normalized_train.train_test_split(
        test_size=validation_fraction,
        seed=seed,
        stratify_by_column="label",
    )
    processed = DatasetDict(train=split["train"], validation=split["test"], test=normalized_test)
    for name, dataset in processed.items():
        validate_split_contract(dataset[:], split_name=name)

    output_dir.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(output_dir))
    (output_dir / "lineage.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "revision": revision,
                "seed": seed,
                "validation_fraction": validation_fraction,
                "labels": list(BANKING77_LABELS),
                "counts": {name: len(dataset) for name, dataset in processed.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit("Use the text-clf-data entrypoint")
