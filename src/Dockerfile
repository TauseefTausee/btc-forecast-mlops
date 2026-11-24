# syntax=docker/dockerfile:1.7
# Training / ETL image used by the Kubeflow pipeline components.
# Build from the repository root:  docker build -f src/Dockerfile -t btc-forecast-trainer .

FROM python:3.10-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

FROM python:3.10-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MODEL_DIR=/app/models
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src/ ./src/
RUN mkdir -p /app/models && chown app:app /app/models
USER 10001
# Each pipeline step picks its module, e.g. `python -m src.etl_fetch --output ...`
ENTRYPOINT ["python", "-m"]
CMD ["src.bootstrap"]
