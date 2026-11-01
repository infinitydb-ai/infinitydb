"""Provider-neutral embedding interfaces and a local Ollama adapter."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot produce valid vectors."""


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    provider: str
    model: str
    dimensions: int | None = None


class EmbeddingProvider(Protocol):
    @property
    def model_info(self) -> EmbeddingModel: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


Transport = Callable[[str, dict[str, object], float], dict[str, object]]


def _http_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    request = Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise EmbeddingError(f"Ollama embedding request failed: {exc}") from exc
    if not isinstance(result, dict):
        raise EmbeddingError("Ollama returned a non-object response")
    return result


class OllamaEmbeddingProvider:
    """Local-first adapter for Ollama POST /api/embed; no call until embed()."""

    def __init__(
        self, model: str, *, base_url: str = "http://127.0.0.1:11434",
        timeout: float = 30.0, dimensions: int | None = None,
        transport: Transport = _http_json,
    ) -> None:
        if not model or not model.strip():
            raise ValueError("model must not be empty")
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be a finite positive number")
        if dimensions is not None and dimensions < 2:
            raise ValueError("dimensions must be at least 2")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dimensions = dimensions
        self._transport = transport

    @property
    def model_info(self) -> EmbeddingModel:
        return EmbeddingModel("ollama", self.model, self._dimensions)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        response = self._transport(
            f"{self.base_url}/api/embed",
            {"model": self.model, "input": list(texts)}, self.timeout,
        )
        raw = response.get("embeddings")
        if not isinstance(raw, list) or len(raw) != len(texts):
            raise EmbeddingError("Ollama response must contain one embedding per input")
        vectors: list[list[float]] = []
        observed_dimensions: int | None = None
        for vector in raw:
            if not isinstance(vector, list) or len(vector) < 2:
                raise EmbeddingError("Ollama returned an embedding with fewer than 2 dimensions")
            try:
                values = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise EmbeddingError("Ollama embedding contains non-numeric values") from exc
            if not all(math.isfinite(value) for value in values):
                raise EmbeddingError("Ollama embedding contains non-finite values")
            if not any(value != 0 for value in values):
                raise EmbeddingError("Ollama embedding has zero norm")
            if observed_dimensions is None:
                observed_dimensions = len(values)
            elif len(values) != observed_dimensions:
                raise EmbeddingError("Ollama returned inconsistent embedding dimensions")
            vectors.append(values)
        if self._dimensions is not None and observed_dimensions != self._dimensions:
            raise EmbeddingError(
                f"embedding dimension mismatch: expected {self._dimensions}, got {observed_dimensions}"
            )
        self._dimensions = observed_dimensions
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
