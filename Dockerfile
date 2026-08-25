# syntax=docker/dockerfile:1.7

# ------------------------------------------------------------
# Builder
# ------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1

WORKDIR /build

COPY requirements.txt .

# BuildKit cache keeps downloaded wheels between builds.
# --prefer-binary + --only-binary prevent pip from silently spending
# a very long time compiling scientific/ML packages from source.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade pip \
    && python -m pip install \
        --prefer-binary \
        --only-binary=:all: \
        --prefix=/install \
        -r requirements.txt

# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app/ ./app/
COPY data/ ./data/

RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
