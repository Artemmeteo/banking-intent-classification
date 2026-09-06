from fastapi import Request

from text_clf.application.predict import PredictIntentService
from text_clf.domain.errors import ModelNotReadyError


def get_predict_service(request: Request) -> PredictIntentService:
    service: PredictIntentService | None = request.app.state.runtime.service
    if service is None:
        raise ModelNotReadyError("model is not loaded")
    return service
