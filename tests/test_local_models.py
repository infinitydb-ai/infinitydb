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


def _stub_quantized_stack(monkeypatch, tmp_path, quantized_exists):
    """Stub every heavy import the INT8 path touches, and count quantizations."""
    state = {"quantized": 0, "truncation": None, "inputs": None}

    class FakeEncoder:
        def __init__(self, text):
            self.ids = [1, len(text)]
            self.attention_mask = [1, 1]
            self.type_ids = [0, 0]

    class FakeTokenizer:
        @classmethod
        def from_file(cls, _path):
            return cls()

        def enable_truncation(self, max_length):
            state["truncation"] = max_length

        def enable_padding(self):
            pass

        def encode_batch(self, pairs):
            state["pairs"] = list(pairs)
            return [FakeEncoder(doc) for _query, doc in pairs]

    class FakeInput:
        def __init__(self, name):
            self.name = name

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            state["model"] = path

        def get_inputs(self):
            return [FakeInput("input_ids"), FakeInput("attention_mask")]

        def get_outputs(self):
            return [FakeInput("logits")]

        def run(self, _outputs, inputs):
            state["inputs"] = inputs
            # Two-column head: (irrelevant, relevant).
            return [[[0.0, float(sum(doc))] for doc in inputs["input_ids"]]]

    def fake_load(module, attribute):
        table = {
            ("huggingface_hub", "hf_hub_download"): lambda repo_id, filename, cache_dir: (
                str(tmp_path / filename)
            ),
            ("tokenizers", "Tokenizer"): FakeTokenizer,
            ("onnxruntime", "InferenceSession"): FakeSession,
            ("onnxruntime", "SessionOptions"): lambda: types.SimpleNamespace(),
            ("onnxruntime", "GraphOptimizationLevel"): types.SimpleNamespace(
                ORT_ENABLE_ALL="all"
            ),
            ("onnxruntime.quantization", "quantize_dynamic"): _fake_quantize(state),
            ("onnxruntime.quantization", "QuantType"): types.SimpleNamespace(QInt8="int8"),
        }
        return table[(module, attribute)]

    monkeypatch.setattr(local_models, "_load", fake_load)
    monkeypatch.setattr(local_models, "_cache_root", lambda cache_dir: tmp_path)
    (tmp_path / "Xenova--ms-marco-MiniLM-L-6-v2.int8.onnx").write_bytes(
        b"onnx" if quantized_exists else b""
    )
    if not quantized_exists:
        (tmp_path / "Xenova--ms-marco-MiniLM-L-6-v2.int8.onnx").unlink()
    return state


def _fake_quantize(state):
    def quantize_dynamic(source, target, weight_type):
        state["quantized"] += 1
        state["weight_type"] = weight_type
        with open(target, "wb") as handle:
            handle.write(b"onnx")

    return quantize_dynamic


def test_quantized_reranker_quantizes_once_and_scores_every_pair(monkeypatch, tmp_path):
    state = _stub_quantized_stack(monkeypatch, tmp_path, quantized_exists=False)

    score = local_models.local_reranker(quantize=True, cache_dir=str(tmp_path))
    first = score("q", ["short", "a much longer passage"])
    second = score("q", ["short"])

    assert state["quantized"] == 1, "the INT8 build must be cached, not redone"
    assert state["truncation"] == 512
    assert len(first) == 2 and first[0] < first[1], "relevance is the last logit"
    assert len(second) == 1
    assert set(state["inputs"]) == {"input_ids", "attention_mask"}
    assert score("q", []) == []


def test_quantized_reranker_reuses_an_existing_int8_file(monkeypatch, tmp_path):
    state = _stub_quantized_stack(monkeypatch, tmp_path, quantized_exists=True)

    local_models.local_reranker(quantize=True, cache_dir=str(tmp_path))

    assert state["quantized"] == 0
