# Banking user intent classification

README.md that properly corresponds in proper to this project will be soon



## Требования

- Python 3.11–3.13;
- `uv` 0.11.23 или совместимая версия;
- Docker 28+ и Docker Compose 2.39+ для контейнерного запуска;
- доступ в интернет при первой загрузке Banking77 и DistilBERT;
- желательно не менее 8 ГБ свободной RAM для локального обучения на CPU.

Установка обычного окружения API и тестов:

```bash
uv sync --frozen
```

Установка зависимостей для обучения:

```bash
uv sync --frozen --group training
```

Установка обучения вместе с DVC/S3:

```bash
uv sync --frozen --group mlops
```





## MLOps-запуск

В этом режиме используются PostgreSQL, MinIO, MLflow, Airflow и API с моделью из Registry.

### 1. Заполнить секреты

### 2. Запустить инфраструктуру без classifier API

```bash
docker compose \
  -f compose.yaml \
  -f compose.mlops.yaml \
  --profile mlops \
  --env-file .env \
  up --build -d \
  postgres minio minio-init mlflow airflow-init airflow-api airflow-scheduler
```

Проверить контейнеры:

```bash
docker compose \
  -f compose.yaml \
  -f compose.mlops.yaml \
  --profile mlops \
  --env-file .env \
  ps
```

Интерфейсы:

- MLflow: <http://127.0.0.1:5000>;
- Airflow: <http://127.0.0.1:8080>.

PostgreSQL и MinIO не публикуются на хост и доступны только во внутренней Docker-сети.

### 3A. Запустить pipeline через Airflow

DAG называется `banking77_train_register_promote`. Новые DAG создаются приостановленными,
поэтому сначала активируйте его:

```bash
docker compose \
  -f compose.yaml \
  -f compose.mlops.yaml \
  --profile mlops \
  --env-file .env \
  exec airflow-scheduler \
  airflow dags unpause banking77_train_register_promote
```

Запустить обучение:

```bash
docker compose \
  -f compose.yaml \
  -f compose.mlops.yaml \
  --profile mlops \
  --env-file .env \
  exec airflow-scheduler \
  airflow dags trigger banking77_train_register_promote
```

DAG выполняет подготовку данных, обучение, регистрацию `candidate`, проверку quality gates
и перенос alias `champion`, если все проверки пройдены.

### 3B. Альтернатива: выполнить pipeline вручную

Подготовить данные, обучить модель и оценить её:

```bash
uv sync --frozen --group mlops
uv run --group mlops dvc repro
uv run --group mlops dvc metrics show
```

Зарегистрировать candidate в работающем MLflow:

```bash
uv run --group training text-clf-register \
  --model artifacts/candidate-model \
  --tracking-uri http://127.0.0.1:5000 \
  --experiment banking77-classification \
  --registered-model banking77-intent-classifier \
  --candidate-alias candidate
```

Проверить quality gates и назначить champion:

```bash
uv run --group training text-clf-promote \
  --tracking-uri http://127.0.0.1:5000 \
  --registered-model banking77-intent-classifier \
  --candidate-alias candidate \
  --champion-alias champion \
  --macro-f1-min 0.80 \
  --max-macro-f1-drop 0.02 \
  --ece-max 0.08 \
  --required-label-count 77 \
  --p95-latency-max-ms 100
```

Если gate не пройден, `champion` не изменяется. Не снижайте пороги только ради успешного
запуска: сначала изучите `artifacts/evaluation.json` и MLflow run.

### 4. Отправить DVC-артефакты в MinIO

MinIO доступен по имени `minio` только внутри Compose-сети. Поэтому `dvc push` запускается
из контейнера Airflow:

```bash
docker compose \
  -f compose.yaml \
  -f compose.mlops.yaml \
  --profile mlops \
  --env-file .env \
  run --rm airflow-scheduler dvc push
```

В Git следует сохранить только метаданные воспроизводимости:

```bash
git add dvc.yaml dvc.lock params.yaml .dvc/config
git commit -m "Зафиксировать версию данных и модели"
```

Большие данные и модель остаются в DVC/MinIO и не попадают в Git.

### 5. Запустить classifier API с champion-моделью

API следует запускать только после появления alias `champion`:

```bash
docker compose \
  -f compose.yaml \
  -f compose.mlops.yaml \
  --profile mlops \
  --env-file .env \
  up -d api
```

При старте API один раз скачивает и проверяет champion-пакет через MLflow, после чего
обрабатывает запросы локально. Во время каждого prediction обращений к MLflow или MinIO
нет.

Проверка выполняется теми же `curl`-командами из standalone-раздела.

### 6. Остановить MLOps-контур

Сохранить данные PostgreSQL, MinIO и логи Airflow:

```bash
docker compose \
  -f compose.yaml \
  -f compose.mlops.yaml \
  --profile mlops \
  --env-file .env \
  down
```

Удалить также named volumes со всеми локальными данными инфраструктуры:

```bash
docker compose \
  -f compose.yaml \
  -f compose.mlops.yaml \
  --profile mlops \
  --env-file .env \
  down --volumes
```

Последняя команда необратимо удаляет локальные базы MLflow/Airflow и артефакты MinIO.

## Rollback модели

Вернуть alias `champion` на ранее записанную версию:

```bash
uv run --group training text-clf-rollback \
  --tracking-uri http://127.0.0.1:5000 \
  --registered-model banking77-intent-classifier \
  --champion-alias champion
```

После rollback пересоздайте API-контейнер, чтобы он заново разрешил alias:

```bash
docker compose \
  -f compose.yaml \
  -f compose.mlops.yaml \
  --profile mlops \
  --env-file .env \
  up -d --force-recreate api
```

## Проверка проекта

```bash
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
docker compose --env-file .env -f compose.yaml config --quiet
docker compose --env-file .env -f compose.yaml -f compose.mlops.yaml --profile mlops config --quiet
```

Сборка API-образа:

```bash
docker build --progress=plain -t text-clf-api:0.2.0 .
```

Образы и инструменты закреплены конкретными версиями. Не заменяйте их на `latest` без
пересборки lock-файла, тестов, smoke-проверок и проверки образа.

## Структура конфигурации

- `compose.yaml` — standalone API;
- `compose.mlops.yaml` — MLOps override и инфраструктура;
- `.env.example` — безопасный шаблон переменных;
- `params.yaml` — параметры данных, обучения и quality gates;
- `dvc.yaml` — граф pipeline;
- `dvc.lock` — фактические хеши конкретного запуска, появляется после `dvc repro`;
- `docs/adr/` — принятые архитектурные решения;
- `docs/runbook.md` — восстановление после эксплуатационных сбоев.
