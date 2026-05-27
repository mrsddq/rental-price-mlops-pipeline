FROM python:3.10.14-slim-bookworm

LABEL maintainer="laraibks@gmail.com"

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

CMD ["python", "main.py", "--compile-only"]
