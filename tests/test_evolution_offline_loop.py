"""Tests for the offline evolution loop (three-layer verification, distillation, release monitor)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

from search_assistant.contracts import (
    BriefingSynthesis,
    BriefingTheme,
    CollectedSource,
    DailyBriefing,
    DomainKnowledgeCandidate,
    DomainKnowledgeEvidence,
)
from search_assistant.evolution.offline_loop import OfflineEvolutionLoop
from search_assistant.evolution.service import DomainKnowledgeCandidateService
from search_assistant.memory.store import MemoryStore


def _insert_trajectory(store: MemoryStore, trajectory_id: str, payload: dict) -> None:
    payload.setdefault("question_id", f"q-{trajectory_id}")
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO trajectory_logs (id, question_id, user_id, chat_id, source, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trajectory_id,
                payload["question_id"],
                "u-1",
                "c-1",
                "test",
                json.dumps(payload, ensure_ascii=False),
                "2026-08-06T00:00:00Z",
            ),
        )


def _clean_payload() -> dict:
    return {
        "question": "How does distributed inference work?",
        "classification": "hard",
        "search_record": {
            "executed": True,
            "engines": ["bing"],
            "queries": ["distributed inference"],
            "sources": [
                {
                    "title": "Distributed inference overview",
                    "url": "https://example.com/distributed-inference",
                    "snippet": "Nodes cooperate over interconnect for decode.",
                }
            ],
        },
        "draft_answer": "draft",
        "calibration": {"ran": True, "revision": "answer"},
        "review": {"ran": True, "approved": True, "issues": [], "revision": "answer"},
        "final_answer": (
            "Distributed inference splits the model across nodes that cooperate over "
            "interconnect. See https://example.com/distributed-inference for the mechanism."
        ),
        "verified_claims": [],
        "unverified_claims": [],
        "active_skills": [],
        "answer_strategy": {},
        "runtime_metadata": {},
        "execution_flags": {},
    }


def _blocked_payload() -> dict:
    payload = _clean_payload()
    payload["final_answer"] = "我已阻断本轮生成，因为证据不足。"
    payload["review"] = {"ran": True, "approved": False, "issues": ["evidence gap"], "revision": "answer"}
    payload["execution_flags"] = {"review_failed": True}
    return payload


def test_offline_loop_verifies_trajectories_and_persists_evaluations(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    _insert_trajectory(store, "traj-clean", _clean_payload())
    _insert_trajectory(store, "traj-blocked", _blocked_payload())

    loop = OfflineEvolutionLoop(store, data_dir=tmp_path)
    report = loop.run()

    assert report["pending_trajectories"] == 2
    assert report["verified"] == 2
    assert report["failures"] == 1
    assert report["judge"] == "rule"
    evaluations = store.list_trajectory_evaluations()
    assert len(evaluations) == 2
    by_trajectory = {item["trajectory_id"]: item for item in evaluations}
    assert by_trajectory["traj-blocked"]["process_verification"]["passed"] is False
    assert "blocked_answer" in by_trajectory["traj-blocked"]["process_verification"]["flags"]
    assert by_trajectory["traj-clean"]["result_verification"]["passed"] is True
    report_file = tmp_path / "evolution" / "offline-evolution-report.json"
    assert report_file.exists()
    gates = store.list_gate_records(gate_type="evolution_offline_run")
    assert len(gates) == 1
    assert gates[0]["result"] == "failed"
    ledger = store.list_project_ledger_entries(entry_type="offline_evolution_run")
    assert len(ledger) == 1
    assert ledger[0]["status"] == "failed"
    # The blocked trajectory produced a reviewable experience item.
    experiences = store.list_experience_items()
    assert any("Offline evolution quality issue" in item["title"] for item in experiences)


def test_offline_loop_is_idempotent(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    _insert_trajectory(store, "traj-clean", _clean_payload())

    first = OfflineEvolutionLoop(store, data_dir=tmp_path).run()
    second = OfflineEvolutionLoop(store, data_dir=tmp_path).run()

    assert first["pending_trajectories"] == 1
    assert second["pending_trajectories"] == 0
    assert second["verified"] == 0
    assert len(store.list_trajectory_evaluations()) == 1


def test_offline_loop_distills_two_trajectory_candidates(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    service.capture_briefing(_briefing("briefing-1", "run-1"))
    service.capture_briefing(_briefing("briefing-2", "run-2"))

    report = OfflineEvolutionLoop(store, data_dir=tmp_path).run()

    assert report["distillation"]["considered"] == 1
    assert report["distillation"]["validated"] == 1
    assert report["distillation"]["weak_signal"] == 0
    candidates = store.list_domain_knowledge_candidates()
    # Distillation promotes a passing candidate to pending_user_confirm;
    # it becomes validated only after explicit user confirmation.
    assert candidates[0]["status"] == "pending_user_confirm"
    assert service.list_candidates(layer="validated_knowledge")[0]["id"] == candidates[0]["id"]


def test_offline_loop_pending_candidate_is_confirmed_to_validated(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing("briefing-1", "run-1"))[0]
    OfflineEvolutionLoop(store, data_dir=tmp_path).run()
    assert store.get_domain_knowledge_candidate(candidate_id)["status"] == "pending_user_confirm"

    confirmed = service.confirm_candidate(candidate_id, reviewer="user-1")
    assert confirmed["confirmed"] is True
    assert store.get_domain_knowledge_candidate(candidate_id)["status"] == "validated"
    assert store.list_project_ledger_entries(entry_type="knowledge_release")
    assert store.list_gate_records(gate_type="domain_knowledge_candidate_human_review")


def test_offline_loop_single_trajectory_well_sourced_reaches_pending(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    service.capture_briefing(_briefing("briefing-1", "run-1"))

    report = OfflineEvolutionLoop(store, data_dir=tmp_path).run()

    assert report["distillation"]["weak_signal"] == 0
    assert report["distillation"]["validated"] == 1
    assert store.list_domain_knowledge_candidates()[0]["status"] == "pending_user_confirm"


def test_offline_loop_release_monitor_reflects_eval_report(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    evaluations_dir = tmp_path / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    (evaluations_dir / "evaluation-report.json").write_text(
        json.dumps(
            {
                "total_questions": 1,
                "items": [],
                "summary": {
                    "flagged_answers": 1,
                    "review_rejected_answers": 0,
                    "result_failed_answers": 0,
                    "process_flagged_answers": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    report = OfflineEvolutionLoop(store, data_dir=tmp_path).run()

    assert report["release_monitor"]["checked"] is True
    assert report["release_monitor"]["passed"] is False
    assert report["release_monitor"]["summary"]["flagged_answers"] == 1


def test_offline_loop_rollback_monitor_flags_regressed_candidate(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    now = "2026-08-06T00:00:00Z"
    candidate = DomainKnowledgeCandidate(
        id="knowledge-regressed",
        topic="industrial AI",
        claim="Review-gated execution is the dominant pattern for MES write-back.",
        applies_when="Use when researching industrial AI agents.",
        evidence=[
            DomainKnowledgeEvidence(
                title="Industrial AI workflow",
                url="https://example.com/industrial-ai/1",
                provider="example",
                retrieved_at=now,
            ),
            DomainKnowledgeEvidence(
                title="Industrial AI workflow 2",
                url="https://example.com/industrial-ai/2",
                provider="example",
                retrieved_at=now,
            ),
        ],
        contradictions=[],
        confidence="medium",
        status="validated",
        source_ids=["briefing-1"],
        fingerprint="regressed-candidate-fingerprint",
        created_at=now,
        updated_at=now,
    )
    store.add_domain_knowledge_candidate(candidate)

    report = OfflineEvolutionLoop(store, data_dir=tmp_path).run()

    regressed = report["rollback_monitor"]["regressed_candidates"]
    assert [item["candidate_id"] for item in regressed] == ["knowledge-regressed"]
    assert regressed[0]["independent_trajectory_count"] == 1
    # The loop never auto-deprecates.
    assert store.get_domain_knowledge_candidate("knowledge-regressed")["status"] == "validated"


def _briefing(briefing_id: str, url_prefix: str) -> DailyBriefing:
    now = datetime.now(UTC)
    sources = [
        CollectedSource(
            id=f"source-{url_prefix}-{index}",
            topic_id="topic-1",
            user_id="user-1",
            title=f"Industrial AI source {index}",
            # Alternate host so multi-source briefings span at least two domains.
            url=f"https://{'example.org' if index % 2 == 0 else 'example.com'}/{url_prefix}/{index}",
            snippet="A source-backed workflow with review gates.",
            platform="web",
            provider=f"provider-{index}",
            query="industrial AI workflow",
            relevance_score=0.9,
            importance_score=0.9,
            retrieved_at=(now - timedelta(days=index)).isoformat().replace("+00:00", "Z"),
        )
        for index in range(1, 3)
    ]
    return DailyBriefing(
        id=briefing_id,
        topic_id="topic-1",
        user_id="user-1",
        chat_id="chat-1",
        topic="industrial AI",
        run_date=now.date(),
        search_directions=["industrial AI workflow"],
        keywords=["industrial AI"],
        sources=sources,
        synthesis=BriefingSynthesis(
            search_content_summary="Sources describe guarded industrial AI workflows.",
            short_summary="The workflow keeps human review before execution.",
            detailed_summary="Evidence supports a review-gated execution pattern.",
            themes=[
                BriefingTheme(
                    name="Review-gated execution",
                    analysis=(
                        "Industrial AI workflows keep a human review gate before writing work orders "
                        "to MES / Manufacturing Execution System production execution."
                    ),
                    source_urls=[source.url for source in sources],
                )
            ],
            key_signal_interpretation="Review gates recur across the sources.",
            analysis_judgment="The pattern is useful planning context but still requires fresh source checks.",
            next_search_directions=["Find deployment metrics", "Check failure recovery"],
            landing_suggestions=["Start with review-only mode", "Record rollback outcomes"],
        ),
        markdown="# Industrial AI",
        created_at=now.isoformat().replace("+00:00", "Z"),
    )
