import importlib
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


def test_memory_api_requires_key_and_rejects_invalid_key(client):
    assert client.post("/v1/memories", json={"content": "private fact"}).status_code == 401
    assert client.post(
        "/v1/memories", headers={"X-API-Key": "wrong"},
        json={"content": "private fact"},
    ).status_code == 401


def test_api_key_binds_requests_to_tenant_not_payload(client):
    created = client.post(
        "/v1/memories",
        headers={"X-API-Key": "tenant-a-secret"},
        json={"content": "tenant a secret fact", "tenant_id": "tenant-b"},
    )
    assert created.status_code == 200

    tenant_a = client.post(
        "/v1/search", headers={"X-API-Key": "tenant-a-secret"},
        json={"query": "tenant a secret"},
    )
    tenant_b = client.post(
        "/v1/search", headers={"X-API-Key": "tenant-b-secret"},
        json={"query": "tenant a secret"},
    )
    assert len(tenant_a.json()) == 1
    assert tenant_b.json() == []


def test_stats_endpoint_is_authenticated_and_tenant_scoped(client):
    assert client.get("/v1/stats").status_code == 401
    response = client.get("/v1/stats", headers={"X-API-Key": "tenant-a-secret"})
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_invalid_auth_configuration_fails_closed(client, monkeypatch):
    monkeypatch.setenv("INFINITYDB_API_KEYS", "not-json")
    response = client.post("/v1/search", json={"query": "anything"})
    assert response.status_code == 503
