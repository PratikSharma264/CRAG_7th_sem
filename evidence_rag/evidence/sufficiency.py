"""Determine whether evidence has enough query-term coverage."""
from ..evaluation.retrieval_metrics import tokenize


def assess(query: str, evidence: str, threshold: float = 0.5) -> dict:
    terms = set(tokenize(query))
    coverage = len(terms & set(tokenize(evidence))) / len(terms) if terms else 0.0
    return {"sufficient": coverage >= threshold, "coverage": coverage}
