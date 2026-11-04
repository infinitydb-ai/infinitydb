from benchmarks.vendor_neutral.runner import percentile, vector_for


def test_feature_hash_is_stable_and_normalized():
    first = vector_for("InfinityDB memory")
    second = vector_for("InfinityDB memory")
    assert first == second
    assert len(first) == 128
    assert abs(sum(value * value for value in first) ** 0.5 - 1.0) < 1e-9


def test_percentile_handles_empty_and_ordered_values():
    assert percentile([], .95) == 0.0
    assert percentile([4.0, 1.0, 3.0, 2.0], .50) == 2.0
    assert percentile([4.0, 1.0, 3.0, 2.0], .95) == 4.0
