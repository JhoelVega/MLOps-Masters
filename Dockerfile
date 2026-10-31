FROM python:3.12-slim AS builder

ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1
WORKDIR /app

RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

FROM python:3.12-slim AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PREFECT_HOME=/prefect
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY load_data.py ./load_data.py

EXPOSE 4200 5000
