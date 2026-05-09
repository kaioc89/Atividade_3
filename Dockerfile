FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && install -d /usr/share/postgresql-common/pgdg \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc -o /tmp/postgresql.asc \
    && gpg --dearmor --yes -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.gpg /tmp/postgresql.asc \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.gpg] https://apt.postgresql.org/pub/repos/apt trixie-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client-18 \
    && rm -rf /var/lib/apt/lists/* /tmp/postgresql.asc

COPY pyproject.toml README.md backup_atividade_2_reset.sql ./
COPY src ./src
COPY outputs/audit ./outputs/audit

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e .

RUN mkdir -p outputs/audit outputs/backup

EXPOSE 8000

CMD ["uvicorn", "atividade_2.web:app", "--host", "0.0.0.0", "--port", "8000"]
