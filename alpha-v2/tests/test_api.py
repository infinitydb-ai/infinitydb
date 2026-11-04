import json

import pytest
from fastapi.testclient import TestClient

from infinitydb_next.store import MemoryStore
import infinitydb_next.api as api


@pytest.fixture
def client(tmp_path, monkeypatch):
    api.store.close()
    monkeypatch.setattr(api, "store", MemoryStore(tmp_path / "api.sqlite3"))
    monkeypatch.setenv(
        "INFINITYDB_API_KEYS",
        json.dumps({"tenant-a-secret": "tenant-a", "tenant-b-secret": "tenant-b"}),
    )
    with TestClient(api.app) as test_client:
        yield test_client
    api.store.close()


def test_health_is_public_and_does_not_expose_database_stats(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("get", "/v1/stats", None),
        ("post", "/v1/memories", {"content": "private fact"}),
        ("post", "/v1/search", {"query": "private fact"}),
        ("put", "/v1/memories/not-owned/embedding", {"embedding": [1.0]}),
        ("post", "/v1/search/vector", {"embedding": [1.0]}),
        ("post", "/v1/search/hybrid", {"query": "private fact", "embedding": [1.0]}),
        ("post", "/v1/context", {"query": "private fact"}),
    ],
)
def test_every_data_route_requires_authentication(client, method, path, payload):
    request = getattr(client, method)
    response = request(path, json=payload) if payload is not None else request(path)
    assert response.status_code == 401


def test_memory_api_rejects_invalid_key(client):
    response = client.post(
        "/v1/memories",
        headers={"X-API-Key": "wrong"},
        json={"content": "private fact"},
    )
    assert response.status_code == 401


def test_api_key_binds_requests_to_tenant_not_payload(client):
    created = client.post(
        "/v1/memories",
        headers={"X-API-Key": "tenant-a-secret"},
        json={"content": "tenant a secret fact", "tenant_id": "tenant-b"},
    )
    assert created.status_code == 200

    tenant_a = client.post(
        "/v1/search",
        headers={"X-API-Key": "tenant-a-secret"},
        json={"query": "tenant a secret"},
    )
    tenant_b = client.post(
        "/v1/search",
        headers={"X-API-Key": "tenant-b-secret"},
        json={"query": "tenant a secret"},
    )
    assert len(tenant_a.json()) == 1
    assert tenant_b.json() == []


def test_stats_are_scoped_to_authenticated_tenant(client):
    created = client.post(
        "/v1/memories",
        headers={"X-API-Key": "tenant-a-secret"},
        json={"content": "tenant a only"},
    )
    assert created.status_code == 200

    tenant_a = client.get("/v1/stats", headers={"X-API-Key": "tenant-a-secret"})
    tenant_b = client.get("/v1/stats", headers={"X-API-Key": "tenant-b-secret"})
    assert tenant_a.json()["memories"] == 1
    assert tenant_b.json()["memories"] == 0


def test_embedding_cannot_be_written_across_tenants(client):
    created = client.post(
        "/v1/memories",
        headers={"X-API-Key": "tenant-a-secret"},
        json={"content": "tenant a embedding target"},
    )
    memory_id = created.json()["id"]

    cross_tenant = client.put(
        f"/v1/memories/{memory_id}/embedding",
        headers={"X-API-Key": "tenant-b-secret"},
        json={"embedding": [1.0, 0.0]},
    )
    assert cross_tenant.status_code == 400

    owner_update = client.put(
        f"/v1/memories/{memory_id}/embedding",
        headers={"X-API-Key": "tenant-a-secret"},
        json={"embedding": [1.0, 0.0]},
    )
    assert owner_update.status_code == 200

    tenant_b = client.post(
        "/v1/search/vector",
        headers={"X-API-Key": "tenant-b-secret"},
        json={"embedding": [1.0, 0.0]},
    )
    assert tenant_b.json() == []


@pytest.mark.parametrize("invalid_config", ["not-json", "[]", "{}", "null", "123"])
def test_invalid_auth_configuration_fails_closed(client, monkeypatch, invalid_config):
    monkeypatch.setenv("INFINITYDB_API_KEYS", invalid_config)
    response = client.post("/v1/search", json={"query": "anything"})
    assert response.status_code == 503
