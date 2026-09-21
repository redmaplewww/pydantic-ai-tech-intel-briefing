from __future__ import annotations

import os
import tempfile

from search_assistant.briefing.service import DailyBriefingService
from search_assistant.contracts import LayeredMemoryItem
from search_assistant.memory.store import MemoryStore


def _store() -> MemoryStore:
    db_path = os.path.join(tempfile.mkdtemp(), "routing-test.sqlite3")
    store = MemoryStore(db_path)
    store.initialize()
    return store


def _service(store: MemoryStore) -> DailyBriefingService:
    class _StubSearch:
        def search(self, query: str, limit: int = 5):
            return []

    return DailyBriefingService(store, _StubSearch())


def _insert_trajectory(store: MemoryStore, question: str, source: str = "cli") -> None:
    import json
    from datetime import UTC, datetime

    payload = json.dumps(
        {
            "question": question,
            "final_answer": f"analysis of {question}",
            "answer_strategy": {"primary": "search-then-synthesize"},
        },
        ensure_ascii=False,
    )
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO trajectory_logs (id, question_id, user_id, chat_id, source, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"traj-{abs(hash(question)) % 100000}",
                f"q-{abs(hash(question)) % 100000}",
                "u-1",
                "c-1",
                source,
                payload,
                datetime.now(UTC).isoformat(),
            ),
        )


def test_knowledge_context_routes_episodic_trajectories():
    store = _store()
    _insert_trajectory(store, "AI agent memory architectures in production")
    _insert_trajectory(store, "quantum error correction for qubits")
    service = _service(store)

    context = service._knowledge_context("AI agent memory architectures")
    episodic = context["episodic_trajectories"]
    assert isinstance(episodic, list)
    assert len(episodic) >= 1
    matched_terms = episodic[0]["matched_terms"]
    assert any(term in ("ai", "agent", "memory") for term in matched_terms)
    assert "memory" in episodic[0]["summary"]


def test_knowledge_context_routes_procedural_rules():
    store = _store()
    store.add_layered_memory_item(
        layer="run_experience",
        kind="search_strategy",
        content="For AI agent topics, prefer searching arXiv and GitHub before general web.",
        source_id="traj-1",
        metadata={"topic": "AI agent"},
    )
    store.add_layered_memory_item(
        layer="domain_knowledge",
        kind="verified_fact",
        content="Persistent memory systems pair a vector store with a compaction policy.",
        source_id="brief-1",
        metadata={"topic": "AI agent memory"},
    )
    store.add_layered_memory_item(
        layer="preference",
        kind="user_pref",
        content="User prefers concise answers under 300 words.",
        source_id="u-1",
    )
    service = _service(store)

    context = service._knowledge_context("AI agent memory architectures")
    rules = context["procedural_rules"]
    assert isinstance(rules, list)
    assert len(rules) >= 1
    layers = {rule["layer"] for rule in rules}
    assert "run_experience" in layers or "domain_knowledge" in layers
    # preference layer is not a procedural rule and must not be returned.
    assert all(rule["layer"] in ("run_experience", "domain_knowledge") for rule in rules)


def test_knowledge_context_metadata_counts_new_routes():
    store = _store()
    _insert_trajectory(store, "AI agent memory architectures")
    store.add_layered_memory_item(
        layer="run_experience",
        kind="search_strategy",
        content="AI agent searches should start on arXiv.",
        source_id="traj-1",
    )
    service = _service(store)

    context = service._knowledge_context("AI agent memory")
    metadata = service._knowledge_context_metadata(context)
    assert metadata["episodic_trajectories"] >= 1
    assert metadata["procedural_rules"] >= 1
    assert "seeded_graphs" in metadata
    assert "reviewed_graph_hits" in metadata


def test_episodic_routing_ignores_unrelated_trajectories():
    store = _store()
    _insert_trajectory(store, "quantum error correction and surface codes")
    _insert_trajectory(store, "Rust async runtime scheduling")
    service = _service(store)

    context = service._knowledge_context("AI agent memory architectures")
    episodic = context["episodic_trajectories"]
    for item in episodic:
        assert "memory" in item["summary"] or "agent" in item["summary"]
