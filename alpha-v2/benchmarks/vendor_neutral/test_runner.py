from benchmarks.vendor_neutral.runner import percentile, run, vector_for


class FakeAdapter:
    def __init__(self):
        self.rows = {}

    def add(self, memory_id, tenant_id, content, vector):
        self.rows[memory_id] = (tenant_id, content)

    def search(self, tenant_id, query, vector, limit):
        return [key for key, (owner, content) in self.rows.items()
                if owner == tenant_id and content == query][:limit]

    def metadata(self):
        return {"system": "test"}

    def close(self):
        pass


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


def test_runner_uses_identical_tenant_scoped_queries_and_reports_recall():
    result = run(FakeAdapter(), records=20, queries=10, tenants=2, top_k=3)
    assert result["query"]["recall_at_k"] == 1.0
    assert result["query"]["cross_tenant_id_leaks"] == 0
    assert result["workload"]["records"] == 20
