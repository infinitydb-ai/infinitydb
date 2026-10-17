FROM python:3.11-slim

LABEL org.opencontainers.image.title="InfinityDB" \
      org.opencontainers.image.description="High-performance neurosymbolic agent memory engine" \
      org.opencontainers.image.url="https://github.com/infinitydb-ai/infinitydb" \
      org.opencontainers.image.source="https://github.com/infinitydb-ai/infinitydb" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Create persistent storage directory for SQLite WAL databases
RUN mkdir -p /data && chmod 777 /data
ENV INFINITYDB_PATH=/data/infinity.db
ENV PORT=8080

# Install dependencies and infinitydb package
COPY pyproject.toml .
COPY infinitydb/ ./infinitydb/
COPY dashboard/ ./dashboard/

RUN pip install --no-cache-dir .

EXPOSE 8080
VOLUME ["/data"]

CMD ["python", "dashboard/server.py"]
