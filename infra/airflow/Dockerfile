FROM apache/airflow:3.1.8-python3.12

USER root
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /bin/
WORKDIR /opt/text-clf-build
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN UV_PROJECT_ENVIRONMENT=/opt/airflow/text-clf-venv \
    uv sync --frozen --group mlops --no-dev --no-editable

ENV PATH=/opt/airflow/text-clf-venv/bin:$PATH
USER 50000
