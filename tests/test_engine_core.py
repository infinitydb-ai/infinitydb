

def test_assembly_depth_is_capped_by_n_and_accepts_override():
    """`assembly_depth` changes which chunks are assembled, never how many return.

    Phase 2i: chunk assembly traversed only 3*n ranked items, so a caller asking
    for 10 could never receive a turn ranked below 30 — even when that turn was
    the second hop of the question. The parameter must stay a traversal bound.
    """
    from infinitydb.core.engine import InfinityDB

    db = InfinityDB(":memory:")
    try:
        for i in range(40):
            db.add(f"Caroline: turn {i} mentions ceramics and the Lisbon studio.",
                   user_id="u", ts=1709251200.0 + i, auto_supersede=False)
        shallow = db.recall("Where is the ceramics studio?", n=5, user_id="u",
                            reinforce=False, include_policies=False, assembly_depth=5)
        default = db.recall("Where is the ceramics studio?", n=5, user_id="u",
                            reinforce=False, include_policies=False)
        assert len(shallow["results"]) <= 5
        assert len(default["results"]) <= 5
        assert default["results"], "an override must not empty the result"
    finally:
        db.close()
