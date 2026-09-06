import json
from pathlib import Path

import pytest


@pytest.fixture
def tiny_model_package(tmp_path: Path) -> Path:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    from text_clf.domain.banking77 import BANKING77_LABELS
    from text_clf.infrastructure.model_package import write_model_package_metadata

    package = tmp_path / "model-package"
    package.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "lost", "card"]
    (package / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(package / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(package)
    id2label = dict(enumerate(BANKING77_LABELS))
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=77,
            id2label=id2label,
            label2id={label: index for index, label in id2label.items()},
        )
    )
    model.save_pretrained(package, safe_serialization=True)
    (package / "metrics.json").write_text(
        json.dumps(
            {"macro_f1": 0.9, "ece": 0.03, "p95_inference_latency_ms": 20.0},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_model_package_metadata(
        package,
        model_name="tiny-bert-banking77",
        model_version="fixture-v1",
        hf_revision="0000000000000000000000000000000000000000",
        dataset_revision="1111111111111111111111111111111111111111",
        dvc_hash="md5:fixture",
        temperature=1.0,
        ece=0.03,
        unknown_threshold=0.0,
        max_length=8,
    )
    return package
