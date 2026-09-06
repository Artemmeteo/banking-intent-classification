from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from text_clf.domain.banking77 import BANKING77_LABELS


def get_hf_model_and_tokenizer(
    model_name: str,
    num_labels: int = 77,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    if num_labels != len(BANKING77_LABELS):
        raise ValueError("Banking77 models must expose exactly 77 labels")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, local_files_only=local_files_only
    )
    id2label = dict(enumerate(BANKING77_LABELS))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
        num_labels=num_labels,
        id2label=id2label,
        label2id={label: index for index, label in id2label.items()},
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer
