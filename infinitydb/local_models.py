"""Optional local ONNX models, for callers who opt into a semantic signal.

The engine itself ships no model and no runtime dependency: ``recall(...)``
accepts any callable embedder or reranker, and this module is one convenient
way to build them. It is the only place that imports ``fastembed``, and nothing
imports it automatically, so ``pip install infinitydb-ai`` still installs zero
third-party packages.

    pip install "infinitydb-ai[onnx]"

    from infinitydb import InfinityDB
    from infinitydb.local_models import local_embedder, local_reranker

    db = InfinityDB("memory.db")
    embed, model = local_embedder()              # BAAI/bge-small-en-v1.5
    db.index_dense(embed, model=model)           # embed each turn window once
    hits = db.recall("where did the user move", user_id="alice",
                     dense=(embed, model, 0.3))

    cross_encoder = local_reranker()             # ms-marco MiniLM cross-encoder
    hits = db.recall("where did the user move", user_id="alice",
                     rerank=cross_encoder)

Both models run on CPU through ONNX Runtime, need no API key and make no
network call after the one-time model download. Neither is on by default,
because both cost latency and the measured trade-off is published in
``benchmarks/README.md``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

_INSTALL_HINT = 'pip install "infinitydb-ai[onnx]"'
# Cross-encoder inputs are one (query, passage) pair; MiniLM truncates at 512.
_MAX_PAIR_TOKENS = 512


def _load(module: str, attribute: str) -> Any:
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised by test stub
        raise ImportError(
            f"{attribute} needs the optional local-model extra: {_INSTALL_HINT}"
        ) from exc
    target = getattr(mod, attribute, None)
    if target is None:  # pragma: no cover - depends on the fastembed version
        raise ImportError(
            f"{module}.{attribute} is missing; upgrade the extra: {_INSTALL_HINT}"
        )
    return target


def local_embedder(
    model: str = DEFAULT_EMBED_MODEL,
    cache_dir: str | None = None,
    threads: int | None = None,
) -> tuple[Callable[[list[str]], list[list[float]]], str]:
    """Return ``(embed, model_name)`` for use with ``index_dense`` / ``recall``.

    ``embed`` takes a list of strings and returns one float vector per string,
    which is exactly what ``InfinityDB.index_dense`` and ``recall(dense=...)``
    expect. The model name is returned with it because the index is keyed by
    model: switching models means re-embedding, not mixing vector spaces.
    """
    TextEmbedding = _load("fastembed", "TextEmbedding")
    engine = TextEmbedding(model_name=model, cache_dir=cache_dir, threads=threads)

    def embed(texts: list[str]) -> list[list[float]]:
        return [[float(value) for value in vector] for vector in engine.embed(list(texts))]

    return embed, model


def _cache_root(cache_dir: str | None) -> Path:
    if cache_dir:
        return Path(cache_dir)
    get_cache_dir = _load("fastembed.common.utils", "get_cache_dir")
    return Path(get_cache_dir())


def _quantized_reranker(
    model: str,
    cache_dir: str | None = None,
    threads: int | None = None,
) -> Callable[[str, Sequence[str]], list[float]]:
    """A cross-encoder whose weights were dynamically quantized to INT8.

    FastEmbed ships the fp32 ONNX file only, so the INT8 copy is produced once
    with ``onnxruntime.quantization.quantize_dynamic`` and cached next to the
    model, keyed by model name. The tokenizer comes from the same repository and
    truncates each pair at 512 tokens, the limit this class of cross-encoder was
    trained on.
    """
    hf_hub_download = _load("huggingface_hub", "hf_hub_download")
    Tokenizer = _load("tokenizers", "Tokenizer")
    InferenceSession = _load("onnxruntime", "InferenceSession")
    SessionOptions = _load("onnxruntime", "SessionOptions")
    GraphOptimizationLevel = _load("onnxruntime", "GraphOptimizationLevel")
    try:
        quantize_dynamic = _load("onnxruntime.quantization", "quantize_dynamic")
        QuantType = _load("onnxruntime.quantization", "QuantType")
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"INT8 quantization needs onnx ({exc}); reinstall the extra: {_INSTALL_HINT}"
        ) from exc

    root = _cache_root(cache_dir)
    hub_cache = root / "hf"
    quantized = root / f"{model.replace('/', '--')}.int8.onnx"
    if not quantized.exists():
        source = hf_hub_download(
            repo_id=model, filename="onnx/model.onnx", cache_dir=str(hub_cache)
        )
        quantize_dynamic(str(source), str(quantized), weight_type=QuantType.QInt8)
    tokenizer = Tokenizer.from_file(
        hf_hub_download(repo_id=model, filename="tokenizer.json", cache_dir=str(hub_cache))
    )
    tokenizer.enable_truncation(max_length=_MAX_PAIR_TOKENS)
    # Every pair in a batch becomes one row of the same tensor, so the shorter
    # passages must be padded to the longest one in the call.
    tokenizer.enable_padding()
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        options.intra_op_num_threads = int(threads)
    session = InferenceSession(
        str(quantized), sess_options=options, providers=["CPUExecutionProvider"]
    )
    accepted = {item.name for item in session.get_inputs()}
    output_name = session.get_outputs()[0].name

    def rerank(query: str, passages: Sequence[str]) -> list[float]:
        documents = [str(passage) for passage in passages]
        if not documents:
            return []
        encoded = tokenizer.encode_batch([(str(query), doc) for doc in documents])
        columns = {
            "input_ids": [item.ids for item in encoded],
            "attention_mask": [item.attention_mask for item in encoded],
            "token_type_ids": [item.type_ids for item in encoded],
        }
        inputs = {name: values for name, values in columns.items() if name in accepted}
        logits = session.run([output_name], inputs)[0]
        rows = logits.tolist() if hasattr(logits, "tolist") else logits
        scores: list[float] = []
        for row in rows:
            # One-column logits are the relevance score; two-column heads are
            # (irrelevant, relevant), so the last column is the relevance score.
            if isinstance(row, (list, tuple)):
                scores.append(float(row[-1]))
            else:
                scores.append(float(row))
        return scores

    return rerank


def local_reranker(
    model: str = DEFAULT_RERANK_MODEL,
    cache_dir: str | None = None,
    threads: int | None = None,
    quantize: bool = False,
) -> Callable[[str, Sequence[str]], list[float]]:
    """Return ``rerank(query, passages) -> scores`` for ``recall(rerank=...)``.

    Higher is better; one score per passage, in input order. The scores are the
    cross-encoder's logits, not probabilities — only their order matters to the
    engine.

    ``quantize=True`` switches to an INT8 copy of the same model, built once and
    cached, which is several times cheaper per pair on CPU. It is off by
    default: fp32 is the reference, and the measured difference between the two
    is published in ``benchmarks/README.md``.
    """
    if quantize:
        return _quantized_reranker(model=model, cache_dir=cache_dir, threads=threads)
    TextCrossEncoder = _load("fastembed.rerank.cross_encoder", "TextCrossEncoder")
    engine = TextCrossEncoder(model_name=model, cache_dir=cache_dir, threads=threads)

    def rerank(query: str, passages: Sequence[str]) -> list[float]:
        documents = [str(passage) for passage in passages]
        if not documents:
            return []
        return [float(score) for score in engine.rerank(query, documents)]

    return rerank


def local_dense_index(
    db: Any,
    model: str = DEFAULT_EMBED_MODEL,
    cache_dir: str | None = None,
    threads: int | None = None,
    **scope: Any,
) -> tuple[Callable[[list[str]], list[list[float]]], str]:
    """Build an embedder and index every turn window in ``db`` with it.

    Convenience for the common case; pass ``user_id`` / ``session_id`` through
    ``**scope`` to index only that scope. Returns the ``(embed, model)`` pair to
    hand to ``recall(dense=(embed, model, weight))``.
    """
    embed, name = local_embedder(model=model, cache_dir=cache_dir, threads=threads)
    db.index_dense(embed, model=name, **scope)
    return embed, name


__all__ = [
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_RERANK_MODEL",
    "local_dense_index",
    "local_embedder",
    "local_reranker",
]
