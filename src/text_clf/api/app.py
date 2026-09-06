import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial

import anyio
from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram

from text_clf.api.dependencies import get_predict_service
from text_clf.api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    PredictRequest,
    TopKPredictionResponse,
)
from text_clf.application.predict import PredictIntentService
from text_clf.composition import create_model_loader
from text_clf.domain.errors import (
    InvalidTextError,
    ModelNotReadyError,
    PredictionSaturatedError,
)
from text_clf.domain.ports import ModelLoader
from text_clf.infrastructure.logging import configure_logging
from text_clf.settings import Settings, get_settings

LOGGER = logging.getLogger(__name__)


class ConcurrencyGate:
    def __init__(self, limit: int):
        self._limit = limit
        self._active = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._active >= self._limit:
                raise PredictionSaturatedError("prediction capacity is saturated")
            self._active += 1
        try:
            yield
        finally:
            async with self._lock:
                self._active -= 1


@dataclass(slots=True)
class RuntimeState:
    service: PredictIntentService | None = None
    load_error: str | None = None
    gate: ConcurrencyGate | None = None
    thread_limiter: anyio.CapacityLimiter | None = None


def _error_response(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers={"x-request-id": request_id},
    )


def create_app(
    *, settings: Settings | None = None, model_loader: ModelLoader | None = None
) -> FastAPI:
    configured = settings or get_settings()
    configure_logging(
        level=configured.observability.log_level,
        json_logs=configured.observability.json_logs,
    )
    runtime = RuntimeState()
    registry = CollectorRegistry()
    request_count = Counter(
        "text_clf_http_requests_total",
        "HTTP requests by method, route and status",
        ("method", "route", "status"),
        registry=registry,
    )
    request_latency = Histogram(
        "text_clf_http_request_duration_seconds",
        "HTTP request latency",
        ("method", "route"),
        registry=registry,
    )
    inflight_predictions = Gauge(
        "text_clf_inflight_predictions",
        "Predictions currently executing",
        registry=registry,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runtime.gate = ConcurrencyGate(configured.api.max_concurrency)
        runtime.thread_limiter = anyio.CapacityLimiter(configured.api.max_concurrency)
        loader = model_loader or create_model_loader(configured)
        try:
            classifier = await anyio.to_thread.run_sync(
                partial(loader.load, configured.model.uri), limiter=runtime.thread_limiter
            )
            runtime.service = PredictIntentService(
                classifier,
                min_length=configured.api.min_text_length,
                max_length=configured.api.max_text_length,
            )
            LOGGER.info("model loaded", extra={"model_version": classifier.model_version})
        except Exception as error:
            runtime.load_error = f"{type(error).__name__}: {error}"
            LOGGER.exception("model loading failed")
        yield
        if runtime.service is not None:
            await anyio.to_thread.run_sync(runtime.service.close)
        runtime.service = None

    app = FastAPI(
        title="Banking77 Intent Classification API",
        description="Offline-first intent classification with an integrity-checked model package.",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.state.settings = configured

    @app.middleware("http")
    async def observe_request(request: Request, call_next):  # type: ignore[no-untyped-def]
        supplied_id = request.headers.get("x-request-id", "")
        request_id = supplied_id if 0 < len(supplied_id) <= 128 else str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        request_count.labels(request.method, route_path, str(response.status_code)).inc()
        request_latency.labels(request.method, route_path).observe(time.perf_counter() - started)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        messages = []
        for item in error.errors():
            location = ".".join(str(part) for part in item["loc"])
            messages.append(f"{location}: {item['msg']}")
        return _error_response(request, 422, "validation_error", "; ".join(messages))

    @app.exception_handler(InvalidTextError)
    async def invalid_text(request: Request, error: InvalidTextError) -> JSONResponse:
        return _error_response(request, 422, "invalid_text", str(error))

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready(request: Request, error: ModelNotReadyError) -> JSONResponse:
        return _error_response(request, 503, "model_not_ready", str(error))

    @app.exception_handler(PredictionSaturatedError)
    async def saturated(request: Request, error: PredictionSaturatedError) -> JSONResponse:
        return _error_response(request, 429, "prediction_saturated", str(error))

    @app.get("/live", response_model=HealthResponse, tags=["system"])
    async def live() -> HealthResponse:
        return HealthResponse(status="alive")

    @app.get(
        "/ready",
        response_model=HealthResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["system"],
    )
    async def ready(request: Request) -> HealthResponse | JSONResponse:
        if runtime.service is None:
            message = runtime.load_error or "model is not loaded"
            return _error_response(request, 503, "model_not_ready", message)
        return HealthResponse(
            status="ready",
            model_version=runtime.service.model_version,
            device=runtime.service.device,
        )

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        from prometheus_client import generate_latest

        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    async def execute_prediction(
        request: PredictRequest, top_k: int, service: PredictIntentService
    ) -> list[PredictionResponse]:
        if runtime.gate is None or runtime.thread_limiter is None:
            raise ModelNotReadyError("runtime is not initialized")
        async with runtime.gate.slot():
            inflight_predictions.inc()
            try:
                predictions = await anyio.to_thread.run_sync(
                    partial(service.predict, request.text, top_k=top_k),
                    limiter=runtime.thread_limiter,
                )
            finally:
                inflight_predictions.dec()
        return [PredictionResponse.from_entity(item) for item in predictions]

    @app.post(
        "/v1/predict",
        response_model=PredictionResponse,
        responses={422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
        tags=["inference"],
    )
    async def predict(
        request: PredictRequest,
        service: PredictIntentService = Depends(get_predict_service),  # noqa: B008
    ) -> PredictionResponse:
        return (await execute_prediction(request, 1, service))[0]

    @app.post(
        "/v1/predict/top-k",
        response_model=TopKPredictionResponse,
        responses={422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
        tags=["inference"],
    )
    async def predict_top_k(
        request: PredictRequest,
        k: int = Query(default=3, ge=1, le=configured.api.max_top_k),
        service: PredictIntentService = Depends(get_predict_service),  # noqa: B008
    ) -> TopKPredictionResponse:
        return TopKPredictionResponse(predictions=await execute_prediction(request, k, service))

    return app


app = create_app()
