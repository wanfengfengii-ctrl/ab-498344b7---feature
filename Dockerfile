# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    API_PORT=8000

WORKDIR /app

COPY app ./app
COPY tests ./tests
COPY verify ./verify

# Build step: byte-compile everything so syntax errors fail the image build.
RUN python -m compileall -q app tests verify \
    && useradd --create-home --uid 10001 appuser

USER appuser

EXPOSE 8000

CMD ["python", "-m", "app.server"]
