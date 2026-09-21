"""Ebbinghaus decay scorer."""
def decay_score(delta_t, stability):
    import math
    return math.exp(-delta_t / max(stability, 0.1))
