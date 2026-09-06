# Классификация намерений Banking77

Сервис классификации банковских обращений на 77 классов и воспроизводимый MLOps-контур:

- FastAPI для онлайн-инференса;
- Transformers и PyTorch для обучения;
- DVC для воспроизводимого pipeline данных и моделей;
- MLflow Model Registry для версий `candidate` и `champion`;
- MinIO для артефактов MLflow и DVC;
- PostgreSQL для метаданных MLflow и Airflow;
- Airflow для оркестрации обучения, регистрации и promotion модели.

Проект поддерживает два независимых режима:

1. **Standalone** — API загружает готовый пакет модели с локального диска. MLflow, MinIO,
   PostgreSQL и Airflow не нужны.
2. **MLOps** — модель берётся из MLflow Registry по alias `champion`, а обучение может
   запускаться через Airflow.

Проект нельзя считать готовым к production, пока настоящая модель не прошла quality gate
и smoke-тесты в целевом окружении.

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

Источником зависимостей являются `pyproject.toml` и `uv.lock`.

## Настройка `.env`

Создайте локальный файл конфигурации:

```bash
cp .env.example .env
```

В репозитории уже может находиться локальный `.env` с пустыми полями секретов. Заполните
их перед запуском полного MLOps-контура. Compose остановится с понятной ошибкой, если
обязательное значение осталось пустым.

Сгенерировать URL-safe пароль PostgreSQL:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Сгенерировать MinIO access key и secret key:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(18))"
python3 -c "import secrets; print(secrets.token_urlsafe(40))"
```

Сгенерировать Fernet key и JWT secret для Airflow:

```bash
python3 -c "import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

Вставьте результаты в соответствующие поля `.env`: `POSTGRES_PASSWORD`,
`MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, `AIRFLOW_FERNET_KEY` и
`AIRFLOW_JWT_SECRET`.

Пароль PostgreSQL включается в SQLAlchemy URI. Используйте только символы `A-Z`, `a-z`,
`0-9`, `_` и `-`. Команда выше генерирует подходящее значение.

`.env` исключён из Git и Docker build context. `.env.example` содержит только безопасные
заглушки и может храниться в Git. Для production вместо `.env` следует использовать
Docker Secrets, Vault или secret manager платформы.

По умолчанию опубликованные порты слушают только `127.0.0.1`. Не меняйте
`BIND_ADDRESS=127.0.0.1` на `0.0.0.0`, если сервисы не должны быть доступны из локальной
сети.

## Пакет модели

API принимает не одиночный файл весов, а полный неизменяемый пакет:

```text
model-package/
├── manifest.json
├── labels.json
├── preprocessing.json
├── metrics.json
├── config.json
├── model.safetensors
└── tokenizer files
```

Manifest содержит версию модели, зафиксированные revisions датасета и базовой модели,
digest данных, temperature calibration, unknown threshold и SHA-256 файлов. Если пакет
неполный или изменён, `/ready` возвращает ошибку и inference не запускается.

Старые каталоги внутри `models/` не являются корректными пакетами: в них нет tokenizer и
используются placeholder-метки `LABEL_n`. Их нельзя подключать к API.

## Быстрый путь: обучить и запустить standalone API

Этот вариант не требует MLflow, MinIO, PostgreSQL и Airflow.

### 1. Подготовить окружение

```bash
uv sync --frozen --group mlops
```

Перед первым DVC-запуском переместите старый `data/processed` в резервную копию или
удалите его. Legacy-данные имеют несовместимое сопоставление идентификаторов классов.

### 2. Запустить подготовку данных, обучение и оценку

```bash
uv run --group mlops dvc repro
```

Pipeline последовательно выполнит:

```text
download → train → evaluate
```

Результаты:

```text
data/processed/
artifacts/candidate-model/
artifacts/train-metrics.json
artifacts/evaluation.json
dvc.lock
```

Проверить состояние и метрики:

```bash
uv run --group mlops dvc status
uv run --group mlops dvc metrics show
```

Параметры датасета, модели, обучения и quality gate находятся в `params.yaml`.

### 3. Подготовить пакет для standalone API

```bash
uv run text-clf materialize \
  --source artifacts/candidate-model \
  --destination artifacts/model-package
```

### 4A. Запустить API локально

```bash
uv run text-clf-serve
```

### 4B. Или запустить API в Docker

```bash
docker compose --env-file .env up --build -d
docker compose --env-file .env ps
```

Проверка:

```bash
curl -fsS http://127.0.0.1:8000/live
curl -fsS http://127.0.0.1:8000/ready
curl -fsS -X POST http://127.0.0.1:8000/v1/predict \
  -H 'content-type: application/json' \
  -d '{"text":"I lost my card","top_k":3}'
```

Swagger UI доступен по адресу <http://127.0.0.1:8000/docs>, Prometheus-метрики — по
адресу <http://127.0.0.1:8000/metrics>.

Остановить standalone API:

```bash
docker compose --env-file .env down
```

## Полный MLOps-запуск

В этом режиме используются PostgreSQL, MinIO, MLflow, Airflow и API с моделью из Registry.

### 1. Заполнить секреты

Заполните в `.env` как минимум поля `POSTGRES_PASSWORD`, `MINIO_ROOT_USER`,
`MINIO_ROOT_PASSWORD`, `AIRFLOW_FERNET_KEY` и `AIRFLOW_JWT_SECRET`.

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

Подробнее о миграции legacy-данных написано в
[`docs/adr/0004-data-versioning.md`](docs/adr/0004-data-versioning.md), а действия при
сбоях — в [`docs/runbook.md`](docs/runbook.md).
