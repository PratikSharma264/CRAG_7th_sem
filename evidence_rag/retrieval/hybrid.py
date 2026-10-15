"""Hybrid BM25 and dense-result rank fusion."""
from collections.abc import Sequence
from .bm25 import rank as bm25_rank
from .fusion import reciprocal_rank_fusion


def hybrid_rank(query: str, documents: Sequence[str], dense_ids: Sequence[str], top_k: int = 3) -> list[str]:
    lexical_ids = [str(index) for index, _ in bm25_rank(query, documents, top_k=max(top_k, len(documents)))]
    dense_order = [str(item) for item in dense_ids]
    fused = reciprocal_rank_fusion([dense_order, lexical_ids])
    return [doc_id for doc_id, _ in fused[:top_k]]
