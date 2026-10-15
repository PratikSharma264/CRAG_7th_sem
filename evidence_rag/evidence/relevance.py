"""Lightweight lexical relevance scoring."""
from ..evaluation.retrieval_metrics import overlap_score


def score(query: str, evidence: str) -> float:
    return overlap_score(query, evidence)
