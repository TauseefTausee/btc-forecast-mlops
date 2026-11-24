# syntax=docker/dockerfile:1.7
# Production image for the BTC forecast Flask API.
# Build from the repository root:  docker build -f app/Dockerfile -t btc-forecast-api .

########################  Stage 1: build dependencies  ########################
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Isolated virtualenv that is copied as-is into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements first so this layer is cached until they change.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt \
 && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
 && find /opt/venv -type d -name "tests" -path "*site-packages/*" -prune -exec rm -rf {} +

########################  Stage 2: slim runtime  ##############################
FROM python:3.10-slim AS runtime

LABEL org.opencontainers.image.title="btc-forecast-api" \
      org.opencontainers.image.description="7-day Bitcoin close forecast REST API (Flask + Gunicorn)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MODEL_DIR=/app/models \
    PORT=5000

# Unprivileged user with a fixed UID so K8s securityContext can match it.
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src/ ./src/
COPY --chown=app:app app/app.py app/gunicorn.conf.py ./
RUN mkdir -p /app/models && chown app:app /app/models

USER 10001
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT','5000'), timeout=4)" || exit 1

CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
