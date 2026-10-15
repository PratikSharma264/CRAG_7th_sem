"""Metrics for evidence coverage and relevance."""
from ..evidence.relevance import score as relevance_score
from ..evidence.sufficiency import assess as assess_sufficiency


def evaluate_evidence(query: str, evidence: list[str]) -> dict:
    scores = [relevance_score(query, item) for item in evidence]
    sufficiency = assess_sufficiency(query, " ".join(evidence))
    return {"relevance_scores": scores,
            "mean_relevance": sum(scores) / len(scores) if scores else 0.0,
            "sufficiency": sufficiency}
