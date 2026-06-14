FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLM_MODE=off

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY config ./config
COPY data ./data
COPY app ./app
COPY tests ./tests
COPY Makefile ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

RUN mkdir -p outputs

CMD ["python", "-m", "provider_intelligence.cli", "run-all"]
