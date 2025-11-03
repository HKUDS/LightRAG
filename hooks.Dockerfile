FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/neuro-inc/LightRAG"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY README.md poetry.lock pyproject.toml .
RUN pip --no-cache-dir install poetry && poetry install --no-root --no-cache

COPY .apolo .apolo
RUN poetry install --only-root --no-cache

ENTRYPOINT ["app-types"]
