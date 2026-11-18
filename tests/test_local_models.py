"""The optional ONNX helpers must fail with an install hint, not a stack trace.

Nothing here imports fastembed: the tests stub the module so they run on a bare
install, which is the configuration the core package promises.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest

from infinitydb import local_models


def test_missing_extra_names_the_install_command(monkeypatch):
    def missing(name):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(local_models.importlib, "import_module", missing)
    with pytest.raises(ImportError) as excinfo:
        local_models.local_embedder()
    assert "infinitydb-ai[onnx]" in str(excinfo.value)


def test_embedder_wraps_a_fastembed_model(monkeypatch):
    calls = {}

    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir=None, threads=None):
            calls["init"] = (model_name, cache_dir, threads)

        def embed(self, texts):
            calls["texts"] = list(texts)
            return [iter([1, 2.5]) for _ in texts]

    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(
        local_models, "importlib", importlib, raising=False
    )

    embed, model = local_models.local_embedder(
        model="BAAI/bge-small-en-v1.5", cache_dir="/tmp/models", threads=2
    )
    vectors = embed(["a", "b"])

    assert model == "BAAI/bge-small-en-v1.5"
    assert calls["init"] == ("BAAI/bge-small-en-v1.5", "/tmp/models", 2)
    assert calls["texts"] == ["a", "b"]
    assert vectors == [[1.0, 2.5], [1.0, 2.5]]


def test_reranker_returns_one_score_per_passage(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, model_name, cache_dir=None, threads=None):
            pass

        def rerank(self, query, documents):
            assert query == "q"
            return [0.9 if "yes" in d else 0.1 for d in documents]

    rerank_module = types.ModuleType("fastembed.rerank.cross_encoder")
    rerank_module.TextCrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", rerank_module)

    score = local_models.local_reranker()
    assert score("q", ["yes", "no"]) == [0.9, 0.1]
    assert score("q", []) == []
