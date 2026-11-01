# API security and tenant identity

The v2 HTTP adapter requires an API key for every data-bearing endpoint. The
public `GET /health` route is a liveness probe only and does not disclose
database statistics.

## Configure credentials

Set `INFINITYDB_API_KEYS` to a JSON object mapping API tokens to tenant IDs.
Use high-entropy tokens and keep them in a secret manager or process
environment; never commit real tokens.

Linux/macOS:

```bash
export INFINITYDB_API_KEYS='{"replace-with-a-long-random-token":"acme"}'
export INFINITYDB_NEXT_PATH=/var/lib/infinitydb/memory.sqlite3
uvicorn infinitydb_next.api:app --host 127.0.0.1 --port 8000
```

PowerShell:

```powershell
$env:INFINITYDB_API_KEYS='{"replace-with-a-long-random-token":"acme"}'
$env:INFINITYDB_NEXT_PATH="$PWD\memory.sqlite3"
uvicorn infinitydb_next.api:app --host 127.0.0.1 --port 8000
```

Send the token in the `X-API-Key` header. The server resolves that token to
one tenant. Tenant IDs supplied in request JSON are ignored; clients cannot
select another tenant by changing a payload field. Use a distinct random token
for each tenant and rotate credentials by replacing the mapping and restarting
the process.

If authentication is not configured or the JSON mapping is invalid, protected
routes fail closed with HTTP 503. Missing or invalid credentials return HTTP
401. The mapping is process configuration, not a database-backed identity
provider; there is no token issuance, expiry, RBAC, per-user identity, or
in-process hot reload.

## Deployment boundary

This is a first security boundary, not a complete internet-facing deployment
certification. Terminate TLS at a trusted reverse proxy, restrict network
access, protect the environment/secret store, configure backups and filesystem
permissions, and add rate limiting and operational monitoring before exposing
the service publicly. The public health route returns liveness only.
