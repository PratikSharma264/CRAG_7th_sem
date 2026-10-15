"""Reciprocal-rank fusion for combining retriever result lists."""
from collections import defaultdict
from typing import Hashable, Iterable, Sequence


def reciprocal_rank_fusion(rankings: Iterable[Sequence[Hashable]], k: int = 60) -> list[tuple[Hashable, float]]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))
