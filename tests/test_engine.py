from infinitydb import InfinityDB


def test_put_and_get():
    db = InfinityDB()
    db.put("user:1", "likes Python")
    assert db.get("user:1").content == "likes Python"


def test_delete():
    db = InfinityDB()
    db.put("x", "value")
    assert db.delete("x")
    assert db.get("x") is None


def test_empty_key_rejected():
    db = InfinityDB()
    try:
        db.put(" ", "value")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
