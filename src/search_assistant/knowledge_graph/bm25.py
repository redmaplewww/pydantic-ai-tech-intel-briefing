from __future__ import annotations

import math
import re
from collections import Counter


_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+._-]{1,}|[\u4e00-\u9fff]{2,}")


def tokenize(text: str) -> list[str]:
    """Split text into BM25 tokens (lowercased latin terms and CJK bigrams).

    CJK text is chunked into 2-character bigrams so that a query like
    "持久化记忆" can match entity summaries even without word segmentation.
    """
    tokens: list[str] = []
    for chunk in re.split(r"[\s,，。；;:：/|()（）]+", text.lower()):
        chunk = chunk.strip()
        if not chunk:
            continue
        for match in _TOKEN_PATTERN.finditer(chunk):
            token = match.group(0)
            if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9+._-]{1,}", token):
                tokens.append(token)
            elif len(token) >= 2:
                # CJK bigrams preserve ordering information cheaply.
                tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
    return tokens


class BM25Index:
    """A small in-memory BM25 index over a fixed set of documents.

    Only depends on the standard library, so it can live in a lightweight
    knowledge-graph service without pulling in a search engine dependency.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, documents: list[str]):
        self._documents = [tokenize(document) for document in documents]
        self._avgdl = (
            sum(len(doc) for doc in self._documents) / len(self._documents)
            if self._documents
            else 0.0
        )
        # term -> number of documents containing the term
        self._doc_freq: dict[str, int] = Counter()
        for doc in self._documents:
            self._doc_freq.update(set(doc))
        self._idf: dict[str, float] = {
            term: math.log(1 + (len(self._documents) - freq + 0.5) / (freq + 0.5))
            for term, freq in self._doc_freq.items()
        }

    def score(self, query: str, document_index: int) -> float:
        """Return the BM25 score of one document for a query string."""
        if document_index < 0 or document_index >= len(self._documents):
            return 0.0
        doc = self._documents[document_index]
        doc_len = len(doc)
        if doc_len == 0:
            return 0.0
        term_freq = Counter(doc)
        score = 0.0
        for term in set(tokenize(query)):
            if term not in self._idf:
                continue
            freq = term_freq.get(term, 0)
            if freq == 0:
                continue
            denom = freq + self.K1 * (1 - self.B + self.B * doc_len / self._avgdl)
            score += self._idf[term] * freq * (self.K1 + 1) / denom
        return score

    def score_all(self, query: str) -> list[float]:
        """Return BM25 scores for every document."""
        return [self.score(query, index) for index in range(len(self._documents))]


def rrf_rank(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion over one or more ranked document-id lists.

    Each entry in ``rank_lists`` is a list of document indices ordered from
    best to worst. The fused score of a document is ``sum(1 / (k + rank))``
    across all lists, where rank is 1-based.
    """
    fused: dict[int, float] = {}
    for ranked in rank_lists:
        for position, doc_index in enumerate(ranked, start=1):
            fused[doc_index] = fused.get(doc_index, 0.0) + 1.0 / (k + position)
    return fused
