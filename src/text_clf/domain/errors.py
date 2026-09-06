class TextClassificationError(Exception):
    """Base exception safe to translate at an application boundary."""


class InvalidTextError(TextClassificationError):
    pass


class ModelNotReadyError(TextClassificationError):
    pass


class PredictionSaturatedError(TextClassificationError):
    pass


class ArtifactValidationError(TextClassificationError):
    pass


class QualityGateError(TextClassificationError):
    pass
