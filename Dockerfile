FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY tests /app/tests
COPY docs /app/docs
COPY Makefile /app/Makefile

RUN python -m pip install --upgrade pip && python -m pip install -e ".[dev]"

CMD ["python", "-m", "robot_bench.cli", "run-all", "--config", "configs/synthetic.yaml", "--profile", "small"]
