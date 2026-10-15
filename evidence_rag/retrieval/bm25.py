"""Small dependency-free BM25 ranker for a list of text documents."""
import math
import re
from collections import Counter
from typing import Sequence


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rank(query: str, documents: Sequence[str], top_k: int = 3,
         k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
    """Return (document index, BM25 score), highest score first."""
    if not documents or top_k <= 0:
        return []
    terms = _tokens(query)
    corpus = [_tokens(doc) for doc in documents]
    avgdl = sum(map(len, corpus)) / len(corpus) or 1.0
    df = Counter(term for doc in corpus for term in set(doc))
    scores = []
    for index, words in enumerate(corpus):
        counts = Counter(words)
        score = 0.0
        for term in terms:
            freq = counts[term]
            if freq:
                idf = math.log(1 + (len(corpus) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(words) / avgdl))
        scores.append((index, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))[:top_k]
