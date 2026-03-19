FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt \
    POETRY_VERSION=1.8.4 \
    POETRY_NO_INTERACTION=1 \
    TOKENIZERS_PARALLELISM=false

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    bash=5* \
    dumb-init=1.2.5* \
    netcat-openbsd; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    mkdir /home/ase-admin; \
    groupadd -r ase; \
    useradd -r -g ase ase-admin

SHELL ["/bin/bash", "-c"]

RUN pip install poetry==$POETRY_VERSION

WORKDIR /opt/server

COPY *.lock *.py *.toml *.yaml ./
COPY annotations annotations/
COPY config_classes config_classes/
COPY data_classes data_classes/
COPY inference inference/
COPY metrics metrics/
COPY scripts scripts/

RUN set -eux; \
    poetry config virtualenvs.create false; \
    poetry install --no-ansi --no-dev

RUN set -eux; \
    mkdir -p data/datapoints data/predictions data/scores; \
    chown -R ase-admin:ase /home/ase-admin /opt/server

EXPOSE 8000

USER ase-admin

ENTRYPOINT ["dumb-init", "--"]
CMD [ \
    "uvicorn", "app:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "4", \
    "--log-config", "logging.yaml" \
]

HEALTHCHECK \
    --start-period=10s \
    --interval=60s \
    --timeout=30s \
    --retries=5 \
CMD nc -z 127.0.0.1 8000 || exit 1
