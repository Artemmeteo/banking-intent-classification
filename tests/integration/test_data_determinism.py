from pathlib import Path

import pytest

from text_clf.domain.banking77 import BANKING77_LABELS


@pytest.mark.integration
def test_data_split_is_deterministic_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = pytest.importorskip("datasets")
    rows_per_label = 3
    labels = [index for index in range(77) for _ in range(rows_per_label)]
    features = datasets.Features(
        {
            "text": datasets.Value("string"),
            "label": datasets.ClassLabel(names=list(BANKING77_LABELS)),
            "label_text": datasets.Value("string"),
        }
    )
    train = datasets.Dataset.from_dict(
        {
            "text": [f"fixture-{index}" for index in range(len(labels))],
            "label": labels,
            "label_text": [BANKING77_LABELS[index] for index in labels],
        },
        features=features,
    )
    test_labels = list(range(77))
    test = datasets.Dataset.from_dict(
        {
            "text": [f"test-{index}" for index in test_labels],
            "label": test_labels,
            "label_text": [BANKING77_LABELS[index] for index in test_labels],
        },
        features=features,
    )
    fixture = datasets.DatasetDict(train=train, test=test)
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: fixture)

    from text_clf.data.prepare_data import prepare_and_save_data

    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_and_save_data(
        output_dir=first,
        dataset_name="fixture/banking77",
        revision="a" * 40,
        seed=42,
        validation_fraction=1 / rows_per_label,
    )
    prepare_and_save_data(
        output_dir=second,
        dataset_name="fixture/banking77",
        revision="a" * 40,
        seed=42,
        validation_fraction=1 / rows_per_label,
    )
    first_data = datasets.load_from_disk(str(first))
    second_data = datasets.load_from_disk(str(second))
    assert first_data["train"]["text"] == second_data["train"]["text"]
    assert first_data["validation"]["text"] == second_data["validation"]["text"]
    assert first_data["test"]["text"] == second_data["test"]["text"]
