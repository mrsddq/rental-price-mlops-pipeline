FROM python:3.10.14-slim-bookworm

LABEL maintainer="laraibks@gmail.com"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/outputs \
    && chown -R appuser:appuser /app/outputs
USER appuser

EXPOSE 8000

CMD ["python", "main.py", "--compile-only", "--output", "outputs/rental_price_prediction_pipeline.yaml"]
