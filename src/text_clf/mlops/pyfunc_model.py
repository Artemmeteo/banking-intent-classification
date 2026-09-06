from pathlib import Path
from typing import Any

import mlflow


class Banking77PyfuncModel(mlflow.pyfunc.PythonModel):  # type: ignore[misc]
    """MLflow model-from-code adapter around the integrity-checked package."""

    def load_context(self, context: Any) -> None:
        from text_clf.infrastructure.hf_classifier import HuggingFaceClassifier

        self._classifier = HuggingFaceClassifier(Path(context.artifacts["model_package"]))

    def predict(self, context: Any, model_input: Any, params: Any = None) -> Any:
        del context, params
        import pandas as pd

        rows = []
        for text in model_input["text"].tolist():
            prediction = self._classifier.predict(str(text))[0]
            rows.append(
                {
                    "class_id": prediction.class_id,
                    "label": prediction.label,
                    "confidence": prediction.confidence,
                    "is_unknown": prediction.is_unknown,
                }
            )
        return pd.DataFrame(rows)


mlflow.models.set_model(Banking77PyfuncModel())
