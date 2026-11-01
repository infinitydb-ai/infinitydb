import pytest

from infinitydb_next.embeddings import (
    EmbeddingError,
    EmbeddingModel,
    OllamaEmbeddingProvider,
)


def test_ollama_adapter_batches_inputs_and_records_model_dimensions():
    calls = []

    def transport(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"embeddings": [[1, 0, 0], [0, 1, 0]]}

    provider = OllamaEmbeddingProvider(
        "nomic-embed-text", base_url="http://localhost:11434/",
        timeout=7, transport=transport,
    )
    assert provider.embed(["first", "second"]) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert calls == [(
        "http://localhost:11434/api/embed",
        {"model": "nomic-embed-text", "input": ["first", "second"]},
        7,
    )]
    assert provider.model_info == EmbeddingModel("ollama", "nomic-embed-text", 3)


def test_ollama_adapter_supports_single_input():
    provider = OllamaEmbeddingProvider(
        "test-model", transport=lambda *_: {"embeddings": [[0.5, 0.5]]}
    )
    assert provider.embed_one("hello") == [0.5, 0.5]


def test_ollama_adapter_does_not_call_transport_for_empty_batch():
    def unexpected(*_):
        raise AssertionError("transport should not be called")

    assert OllamaEmbeddingProvider("model", transport=unexpected).embed([]) == []


@pytest.mark.parametrize("response", [
    {},
    {"embeddings": []},
    {"embeddings": [[1.0]]},
    {"embeddings": [[]]},
    {"embeddings": [[0.0, 0.0]]},
    {"embeddings": [[float("nan")]]},
    {"embeddings": [[1.0], [1.0, 2.0]]},
    {"embeddings": [["not-a-number"]]},
])
def test_ollama_adapter_rejects_malformed_vectors(response):
    provider = OllamaEmbeddingProvider(
        "model", transport=lambda *_: response
    )
    with pytest.raises(EmbeddingError):
        provider.embed(["first", "second"] if len(response.get("embeddings", [])) == 2 else ["first"])


def test_ollama_adapter_enforces_configured_dimensions():
    provider = OllamaEmbeddingProvider(
        "model", dimensions=3,
        transport=lambda *_: {"embeddings": [[1.0, 0.0]]},
    )
    with pytest.raises(EmbeddingError, match="dimension mismatch"):
        provider.embed_one("hello")


def test_ollama_adapter_validates_configuration_and_inputs():
    with pytest.raises(ValueError, match="model"):
        OllamaEmbeddingProvider(" ")
    with pytest.raises(ValueError, match="timeout"):
        OllamaEmbeddingProvider("model", timeout=0)
    provider = OllamaEmbeddingProvider("model")
    with pytest.raises(ValueError, match="non-empty"):
        provider.embed([""])
