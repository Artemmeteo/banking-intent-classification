FROM python:3.12.11-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.12.11-slim-bookworm AS runtime
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface
WORKDIR /app
RUN groupadd --gid 10001 app && \
    useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["text-clf-serve"]
