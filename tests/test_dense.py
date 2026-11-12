"""Dense fusion: index bookkeeping and lexical+dense candidate generation.

Uses a hand-built embedder, so the test needs no model download and stays
deterministic.
"""

from __future__ import annotations

import pytest

from infinitydb.core.engine import InfinityDB

# Anything about wheels/cars lands on the same axis, which is the point: the
# query never says "car", so only the dense ranking can reach it.
AXES = {
    "vehicle": [1.0, 0.0, 0.0, 0.0],
    "car": [1.0, 0.0, 0.0, 0.0],
    "drive": [1.0, 0.0, 0.0, 0.0],
    "pasta": [0.0, 1.0, 0.0, 0.0],
    "recipe": [0.0, 1.0, 0.0, 0.0],
    "meeting": [0.0, 0.0, 1.0, 0.0],
}


def embedder(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        vector = [0.0, 0.0, 0.0, 0.1]
        for word in text.lower().replace(":", " ").split():
            axis = AXES.get(word.strip(".,!?"))
            if axis:
                vector = [a + b for a, b in zip(vector, axis)]
        vectors.append(vector)
    return vectors


@pytest.fixture()
def db():
    database = InfinityDB(":memory:")
    yield database
    database.close()


def _seed(db: InfinityDB) -> None:
    db.add("Ana: I drive a battered car to work every day.", user_id="u", ts=1000.0)
    db.add("Ana: My pasta recipe needs more garlic.", user_id="u", ts=1001.0)
    db.add("Ana: The meeting moved to Thursday.", user_id="u", ts=1002.0)


def test_index_dense_is_incremental_and_scoped(db):
    _seed(db)
    first = db.index_dense(embedder, model="fake")
    assert first == 3
    assert db.index_dense(embedder, model="fake") == 0
    assert db.index_dense(embedder, model="other") == 3, "a new model re-indexes everything"


def test_dense_fusion_retrieves_a_paraphrase(db):
    _seed(db)
    db.index_dense(embedder, model="fake")
    lexical = db.recall("vehicle", n=3, user_id="u", reinforce=False)
    fused = db.recall(
        "vehicle", n=3, user_id="u", reinforce=False, dense=(embedder, "fake", 0.5)
    )
    assert "car" not in " ".join(item["fact"] for item in lexical["results"]).lower()
    assert "car" in " ".join(item["fact"] for item in fused["results"]).lower()


def test_dense_requires_an_index(db):
    _seed(db)
    with pytest.raises(RuntimeError, match="index_dense"):
        db.recall("vehicle", n=3, user_id="u", dense=(embedder, "missing", 0.3))


def test_dense_arguments_are_validated(db):
    with pytest.raises(TypeError):
        db.recall("x", dense="not-a-tuple")
    with pytest.raises(ValueError):
        db.recall("x", dense=(embedder, "m", 9.0))
