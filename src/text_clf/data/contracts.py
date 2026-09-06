from collections.abc import Mapping, Sequence
from typing import Any

from text_clf.domain.banking77 import BANKING77_LABELS


def validate_split_contract(split: Mapping[str, Sequence[Any]], *, split_name: str) -> None:
    missing = {"text", "label", "label_text"} - set(split)
    if missing:
        raise ValueError(f"{split_name} is missing required columns: {sorted(missing)}")
    lengths = {len(split[column]) for column in ("text", "label", "label_text")}
    if len(lengths) != 1:
        raise ValueError(f"{split_name} columns have inconsistent lengths")
    for label_id, label_text in zip(split["label"], split["label_text"], strict=True):
        if not isinstance(label_id, int) or not 0 <= label_id < len(BANKING77_LABELS):
            raise ValueError(f"{split_name} contains invalid label id")
        if label_text != BANKING77_LABELS[label_id]:
            raise ValueError(f"{split_name} label id/text mapping is inconsistent")

