"""Aggregate evidence quality into a simple confidence estimate."""
from collections.abc import Sequence


def estimate(relevance_scores: Sequence[float], sufficient: bool = True) -> float:
    if not relevance_scores:
        return 0.0
    scores = [max(0.0, min(1.0, float(score))) for score in relevance_scores]
    mean = sum(scores) / len(scores)
    return mean if sufficient else mean * 0.5
