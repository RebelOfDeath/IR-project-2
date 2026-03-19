FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.4 \
    POETRY_NO_INTERACTION=1

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    bash=5* \
    dumb-init=1.2.5*; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    mkdir /home/ase-admin; \
    groupadd -r ase; \
    useradd -r -g ase ase-admin

SHELL ["/bin/bash", "-c"]

RUN pip install poetry==$POETRY_VERSION

WORKDIR /opt/worker

COPY *.lock *.py *.toml *.yaml ./

RUN set -eux; \
    poetry config virtualenvs.create false; \
    poetry install --no-ansi --no-dev

RUN set -eux; \
    chown -R ase-admin:ase /home/ase-admin /opt/worker

USER ase-admin

ENTRYPOINT ["dumb-init", "--"]
CMD ["python", "main.py"]
