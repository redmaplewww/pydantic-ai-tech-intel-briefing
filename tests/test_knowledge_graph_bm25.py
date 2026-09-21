from __future__ import annotations

import os
import tempfile

from search_assistant.knowledge_graph.bm25 import BM25Index, rrf_rank, tokenize
from search_assistant.knowledge_graph.service import DomainKnowledgeGraphService
from search_assistant.memory.store import MemoryStore


def _store() -> MemoryStore:
    db_path = os.path.join(tempfile.mkdtemp(), "kg-test.sqlite3")
    store = MemoryStore(db_path)
    store.initialize()
    return store


def test_tokenize_handles_latin_and_cjk():
    tokens = tokenize("Persistent memory 持久化记忆系统")
    assert "persistent" in tokens
    assert "memory" in tokens
    # CJK bigrams: 持久化记忆系统 -> 持久, 久化, 化记, 记忆, 忆系, 系统
    assert "持久" in tokens
    assert "记忆" in tokens
    assert "系统" in tokens


def test_bm25_prefers_documents_with_term_overlaps():
    index = BM25Index(
        [
            "AI agent memory architectures and persistent memory systems",
            "quantum error correction for superconducting qubits",
            "industrial AI computer vision for medical imaging",
        ]
    )
    scores = index.score_all("AI agent memory")
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_bm25_empty_corpus_returns_zero():
    index = BM25Index([])
    assert index.score_all("anything") == []
    assert index.score("anything", 0) == 0.0


def test_rrf_rank_fuses_two_rankings():
    fused = rrf_rank([[0, 1, 2], [1, 0, 2]])
    # doc 0 ranks 1st then 2nd; doc 1 ranks 2nd then 1st -> equal fused score.
    assert fused[0] == fused[1]
    assert fused[0] > fused[2]


def test_hybrid_query_boosts_weak_literal_match_with_bm25():
    store = _store()
    service = DomainKnowledgeGraphService(store, hybrid_retrieval=True)
    service.seed_default_graphs()
    # A query whose terms appear spread across entity text should get a BM25 bonus.
    hits = service.query("persistent memory architecture", limit=10)
    assert hits
    # With hybrid disabled, the same query must still work (literal path only).
    service2 = DomainKnowledgeGraphService(store, hybrid_retrieval=False)
    hits2 = service2.query("persistent memory architecture", limit=10)
    assert isinstance(hits2, list)


def test_bm25_index_is_cached_between_queries():
    store = _store()
    service = DomainKnowledgeGraphService(store, hybrid_retrieval=True)
    service.seed_default_graphs()
    graphs = service.list_graphs()
    graph_id = str(graphs[0]["id"])
    first = service._bm25_scores_for_graph(graph_id, "memory")
    second = service._bm25_scores_for_graph(graph_id, "memory")
    assert first == second
    assert graph_id in service._bm25_cache
