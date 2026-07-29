#Multistage build

#Stage 1: зависимости
FROM python:3.12-slim AS builder

WORKDIR /app

# устанавливаем только то что нужно для сборки зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# устанавливаем в отдельную папку чтобы скопировать в финальный образ
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


#Stage 2: финальный образ
FROM python:3.12-slim AS runtime

WORKDIR /app

# копируем только установленные пакеты из builder
COPY --from=builder /install /usr/local

# копируем код приложения
COPY app/ ./app/
COPY data/ ./data/

# не запускаем от root
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

# healthcheck — Docker перезапустит контейнер если /health не отвечает
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]