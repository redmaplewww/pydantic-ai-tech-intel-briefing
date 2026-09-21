import sqlite3

import pytest

from search_assistant.contracts import (
    AnswerPackage,
    IncomingMessage,
    ProviderTraceEvent,
    SearchRecord,
    SourceCandidate,
    VerifiedClaim,
)
from search_assistant.memory.store import MemoryStore


def test_records_interaction_answer_and_evidence(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    message = IncomingMessage(
        message_id="m-1",
        event_id="e-1",
        user_id="u-1",
        chat_id="c-1",
        text="What is the latest FastAPI version?",
        source="cli",
    )

    question_id = store.record_interaction(message)
    package = AnswerPackage(
        question_id=question_id,
        answer_text="FastAPI is current only after checking PyPI.",
        classification="research",
        confidence="medium",
        verified_claims=[],
        unverified_claims=["FastAPI latest version"],
        calibration={"ran": True, "critique": "Needs package source", "revision": "Marked unverified"},
        memory_updates=[],
    )
    store.record_answer(package)

    assert store.has_interaction("e-1")
    assert store.latest_answer_for_dedupe_key("e-1").question_id == question_id


def test_answer_recording_appends_an_immutable_trajectory(tmp_path):
    database_path = tmp_path / "assistant.sqlite3"
    store = MemoryStore(database_path)
    store.initialize()
    question_id = store.record_interaction(
        IncomingMessage(
            message_id="m-trajectory",
            event_id="e-trajectory",
            user_id="u-trajectory",
            chat_id="c-trajectory",
            text="What changed in the current API?",
            source="cli",
        )
    )
    package = AnswerPackage(
        question_id=question_id,
        answer_text="The answer is traceable to the recorded search.",
        classification="research",
        confidence="medium",
        search_record=SearchRecord(executed=True, queries=["current API official docs"]),
        trajectory_context={
            "draft_answer": "Initial draft",
            "active_skills": [{"name": "verification"}],
            "runtime_metadata": {"runtime_class": "FakeAgentRuntime"},
        },
    )

    store.record_answer(package)

    trajectories = store.list_trajectory_logs("u-trajectory", "c-trajectory")
    assert len(trajectories) == 1
    assert trajectories[0]["question_id"] == question_id
    assert trajectories[0]["payload"]["question"] == "What changed in the current API?"
    assert trajectories[0]["payload"]["draft_answer"] == "Initial draft"
    assert trajectories[0]["payload"]["search_record"]["executed"] is True
    assert trajectories[0]["payload"]["active_skills"] == [{"name": "verification"}]

    with sqlite3.connect(database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="trajectory logs are immutable"):
            connection.execute(
                "UPDATE trajectory_logs SET source = 'changed' WHERE question_id = ?",
                (question_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="trajectory logs are immutable"):
            connection.execute("DELETE FROM trajectory_logs WHERE question_id = ?", (question_id,))


def test_updates_answer_verification_and_deduplicates_evidence(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    package = AnswerPackage(
        question_id="q-verify",
        answer_text="NVIDIA DGX Spark 官方规格确认 128GB 统一内存。",
        classification="research",
        confidence="medium",
    )
    claim = VerifiedClaim(
        claim="NVIDIA DGX Spark 官方规格确认 128GB 统一内存",
        verdict="verified",
        source="https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
        checked_at="2026-07-03T00:00:00Z",
        notes="unit",
    )
    store.record_answer(package)

    first_added = store.update_answer_verification(
        "q-verify",
        verified_claims=[claim],
        unverified_claims=[],
    )
    second_added = store.update_answer_verification(
        "q-verify",
        verified_claims=[claim],
        unverified_claims=[],
    )

    stored = store.list_answers()[0]
    evidence = store.list_evidence()
    assert first_added == 1
    assert second_added == 0
    assert [verified.claim for verified in stored.verified_claims] == [
        "NVIDIA DGX Spark 官方规格确认 128GB 统一内存"
    ]
    assert stored.unverified_claims == []
    assert len(evidence) == 1
    assert evidence[0]["source"] == "https://www.nvidia.com/en-us/products/workstations/dgx-spark/"


def test_update_answer_verification_removes_stale_evidence(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    package = AnswerPackage(
        question_id="q-stale-evidence",
        answer_text="A claim was once verified.",
        classification="research",
        confidence="medium",
    )
    stale_claim = VerifiedClaim(
        claim="A broad claim that is no longer verified",
        verdict="verified",
        source="https://example.test/source",
        checked_at="2026-07-03T00:00:00Z",
        notes="old verifier",
    )
    store.record_answer(package)
    store.update_answer_verification(
        "q-stale-evidence",
        verified_claims=[stale_claim],
        unverified_claims=[],
    )

    store.update_answer_verification(
        "q-stale-evidence",
        verified_claims=[],
        unverified_claims=["A broad claim that is no longer verified"],
    )

    stored = store.list_answers()[0]
    assert stored.verified_claims == []
    assert stored.unverified_claims == ["A broad claim that is no longer verified"]
    assert store.list_evidence() == []


def test_lists_latest_interactions_first(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    first_id = store.record_interaction(
        IncomingMessage(
            message_id="m-1",
            event_id="e-1",
            user_id="u-1",
            chat_id="c-1",
            text="first",
            source="feishu-long-connection",
        )
    )
    second_id = store.record_interaction(
        IncomingMessage(
            message_id="m-2",
            event_id="e-2",
            user_id="u-1",
            chat_id="c-1",
            text="second",
            source="feishu-long-connection",
        )
    )

    rows = store.list_interactions(limit=1)

    assert rows[0]["id"] == second_id
    assert rows[0]["message_id"] == "m-2"
    assert first_id != second_id


def test_memory_supersession_keeps_old_record(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()

    old_id = store.add_memory_item("preference", "likes concise answers", "manual")
    new_id = store.add_memory_item("preference", "likes concise answers with sources", "manual", supersedes_id=old_id)

    rows = store.list_memory_items()
    assert [row["id"] for row in rows] == [old_id, new_id]
    assert rows[1]["supersedes_id"] == old_id


def test_records_feishu_reply_attempts(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()

    attempt_id = store.record_reply_attempt(
        question_id="q-1",
        message_id="om_1",
        channel="feishu-long-connection",
        status="success",
        response={"message_id": "om_1", "response": {"message_id": "reply_1"}},
    )

    attempts = store.list_reply_attempts()
    assert attempts[0]["id"] == attempt_id
    assert attempts[0]["question_id"] == "q-1"
    assert attempts[0]["message_id"] == "om_1"
    assert attempts[0]["channel"] == "feishu-long-connection"
    assert attempts[0]["status"] == "success"
    assert attempts[0]["error"] is None
    assert "reply_1" in attempts[0]["response_json"]


def test_records_runtime_sessions(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()

    session_id = store.record_runtime_session(
        kind="feishu-long-connection",
        process_id=1234,
        status="started",
        metadata={
            "model_provider": "deepseek",
            "search_provider": "browser",
            "feishu_app_id_configured": True,
            "deepseek_api_key_configured": True,
        },
    )

    sessions = store.list_runtime_sessions()
    assert sessions[0]["id"] == session_id
    assert sessions[0]["kind"] == "feishu-long-connection"
    assert sessions[0]["process_id"] == 1234
    assert sessions[0]["status"] == "started"
    assert "deepseek" in sessions[0]["metadata_json"]


def test_records_experience_items(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()

    experience_id = store.add_experience_item(
        title="Verify API claims before replying",
        body="When a question mentions current API behavior, search first and calibrate the answer.",
        source_ids=["q-1", "ev-1"],
    )

    rows = store.list_experience_items()
    assert rows[0]["id"] == experience_id
    assert rows[0]["title"] == "Verify API claims before replying"
    assert "search first" in rows[0]["body"]
    assert rows[0]["source_ids_json"] == '["q-1", "ev-1"]'


def test_records_source_candidates_provider_trace_feedback_signals_and_layered_memory(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    topic = store.upsert_topic("u-1", "c-1", "industrial AI")
    store.set_topic_source_recipe(topic.id, {"general-web": 3.0, "bilibili": 1.0})
    refreshed = store.upsert_topic("u-1", "c-1", "industrial AI")

    candidate = SourceCandidate(
        id="cand-1",
        topic_id=topic.id,
        user_id="u-1",
        title="Industrial AI",
        url="https://example.com/industrial-ai",
        snippet="Agent connects MES work orders.",
        platform="web",
        provider="browser-bing",
        query="industrial AI",
        status="accepted",
        reason="accepted for briefing ranking",
        relevance_score=2.0,
        importance_score=8.0,
        retrieved_at="2026-07-25T00:00:00Z",
        created_at="2026-07-25T00:00:00Z",
    )
    event = ProviderTraceEvent(
        provider="browser-bing",
        query="industrial AI",
        status="success",
        result_count=1,
        tier="tier-3",
        budget_share=0.2,
        checked_at="2026-07-25T00:00:00Z",
    )

    store.record_source_candidate(candidate)
    store.record_provider_trace_event(event, run_id="brief-1", topic_id=topic.id)
    signal_id = store.add_topic_feedback_signal(
        topic.id,
        "u-1",
        "c-1",
        signal_type="style",
        scope="topic",
        body="Make the briefing easier to read.",
        metadata={"feedback_id": "feedback-1"},
    )
    memory_id = store.add_layered_memory_item(
        "preference",
        "briefing_style_feedback",
        "Make the briefing easier to read.",
        source_id=signal_id,
        user_id="u-1",
        chat_id="c-1",
    )

    assert refreshed.source_recipe == {"general-web": 3.0, "bilibili": 1.0}
    assert store.list_source_candidates(topic_id=topic.id)[0].status == "accepted"
    stored_event = store.list_provider_trace_events(topic_id=topic.id)[0]
    assert stored_event.provider == "browser-bing"
    assert stored_event.tier == "tier-3"
    assert stored_event.budget_share == 0.2
    assert store.list_topic_feedback_signals(topic.id)[0].metadata["feedback_id"] == "feedback-1"
    assert store.list_layered_memory_items("preference", "u-1", "c-1")[0].id == memory_id
