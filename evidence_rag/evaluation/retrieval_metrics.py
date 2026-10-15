"""Dependency-free retrieval metrics, matching the legacy evaluation API."""
from __future__ import annotations
import math
import re
from typing import Iterable, List, Sequence


def normalize_text(text: str) -> str:
    value = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def overlap_score(query: str, document: str) -> float:
    q, d = set(tokenize(query)), set(tokenize(document))
    return len(q & d) / len(q | d) if q and d else 0.0


def precision_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0: return 0.0
    rel = {normalize_text(item) for item in relevant}
    return len(rel & {normalize_text(item) for item in retrieved[:k]}) / k


def recall_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    rel = {normalize_text(item) for item in relevant}
    return len(rel & {normalize_text(item) for item in retrieved[:k]}) / len(rel) if k > 0 and rel else 0.0


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def mean_reciprocal_rank(relevant: Iterable[str], retrieved: Sequence[str]) -> float:
    rel = {normalize_text(item) for item in relevant}
    return next((1 / rank for rank, item in enumerate(retrieved, 1) if normalize_text(item) in rel), 0.0)


def ndcg_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    rel = {normalize_text(item) for item in relevant}
    if k <= 0 or not rel: return 0.0
    seen, dcg = set(), 0.0
    for rank, item in enumerate(retrieved[:k], 1):
        item = normalize_text(item)
        if item in rel and item not in seen:
            dcg += 1 / math.log2(rank + 1)
            seen.add(item)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(rel)) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_retrieval(query: str, retrieved_docs: Sequence[str], relevant_docs: Sequence[str], top_k: int) -> dict:
    p, r = precision_at_k(relevant_docs, retrieved_docs, top_k), recall_at_k(relevant_docs, retrieved_docs, top_k)
    top = retrieved_docs[:top_k]
    return {"query": normalize_text(query), "precision_at_k": p, "recall_at_k": r,
            "f1_score": f1_score(p, r), "mrr": mean_reciprocal_rank(relevant_docs, top),
            "ndcg_at_k": ndcg_at_k(relevant_docs, top, top_k), "retrieved_count": len(top),
            "relevant_count": len({normalize_text(item) for item in relevant_docs})}
