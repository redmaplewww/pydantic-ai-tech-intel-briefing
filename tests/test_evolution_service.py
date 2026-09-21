from datetime import UTC, date, datetime, timedelta
import json
import sqlite3
import sys
from io import BytesIO, TextIOWrapper

import pytest

from search_assistant.contracts import (
    BriefingSynthesis,
    BriefingTheme,
    CollectedSource,
    DailyBriefing,
    DomainKnowledgeCandidate,
    DomainKnowledgeEvidence,
)
from search_assistant.evolution.service import DomainKnowledgeCandidateService
from search_assistant.memory.store import MemoryStore
from search_assistant.reports.service import ReportService


def test_briefing_evidence_creates_deduplicated_reviewable_domain_candidates(tmp_path):
    database_path = tmp_path / "assistant.sqlite3"
    store = MemoryStore(database_path)
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    briefing = _briefing(source_count=2)
    independent_briefing = _briefing(source_count=2, briefing_id="briefing-2", url_prefix="independent")

    first_ids = service.capture_briefing(briefing)
    second_ids = service.capture_briefing(independent_briefing)

    assert first_ids == second_ids
    assert len(first_ids) == 1
    candidates = store.list_domain_knowledge_candidates()
    assert len(candidates) == 1
    assert candidates[0]["topic"] == "industrial AI"
    assert candidates[0]["status"] == "candidate"
    assert candidates[0]["confidence"] == "medium"
    # Evidence and source ids from both independent briefings are merged.
    assert len(candidates[0]["evidence"]) == 4
    assert candidates[0]["source_ids"][0] == briefing.id
    assert "briefing-2" in candidates[0]["source_ids"]
    assert service.independent_trajectory_count(candidates[0]) == 2
    graph_links = store.list_domain_candidate_graph_links(first_ids[0])
    assert len(graph_links) == 1
    assert graph_links[0]["graph_id"] == "industrial-ai"
    assert graph_links[0]["entity_id"] in {"mes", "controlled-work-order-orchestration"}
    with sqlite3.connect(database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="candidate graph links are immutable"):
            connection.execute(
                "UPDATE domain_knowledge_candidate_graph_links SET note = 'changed' WHERE candidate_id = ?",
                (first_ids[0],),
            )

    validation = service.validate(first_ids[0])
    service.deprecate(first_ids[0], "superseded by a later evidence review")

    assert validation["validated"] is True
    assert validation["candidate_id"] == first_ids[0]
    assert validation["failures"] == []
    assert validation["knowledge_layer"] == "validated_knowledge"
    assert validation["independent_trajectory_count"] == 2
    assert store.get_domain_knowledge_candidate(first_ids[0])["status"] == "deprecated"
    gates = store.list_gate_records()
    assert {gate["gate_type"] for gate in gates} == {
        "domain_knowledge_candidate_validation",
        "domain_knowledge_candidate_rollback",
    }
    assert all(gate["result"] == "passed" for gate in gates)
    rollback_entries = store.list_project_ledger_entries(entry_type="knowledge_rollback")
    assert rollback_entries[0]["status"] == "deprecated"
    events = store.list_domain_knowledge_candidate_events(first_ids[0])
    assert [event["to_status"] for event in events] == ["candidate", "candidate", "validated", "deprecated"]
    assert events[-1]["reason"] == "superseded by a later evidence review"
    assert any("merged evidence" in event["reason"] for event in events)
    with sqlite3.connect(database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="candidate events are immutable"):
            connection.execute(
                "UPDATE domain_knowledge_candidate_events SET reason = 'changed' WHERE candidate_id = ?",
                (first_ids[0],),
            )


def test_candidate_validation_keeps_single_source_claim_in_candidate_state(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing(source_count=1))[0]

    result = service.validate(candidate_id)

    assert result["validated"] is False
    assert result["failures"] == [
        "fewer_than_two_independent_sources",
        "insufficient_source_authority",
        "low_confidence",
    ]
    assert store.get_domain_knowledge_candidate(candidate_id)["status"] == "candidate"
    gates = store.list_gate_records(gate_type="domain_knowledge_candidate_validation")
    assert len(gates) == 1
    assert gates[0]["result"] == "failed"
    assert gates[0]["metadata"]["evidence_url_count"] == 1
    assert gates[0]["metadata"]["source_domain_count"] == 1
    assert gates[0]["metadata"]["effective_source_count"] == 1.0
    assert gates[0]["metadata"]["independent_trajectory_count"] == 1
    assert gates[0]["metadata"]["knowledge_layer"] == "weak_signal"
    assert "exploratory clue" in gates[0]["metadata"]["intended_use"]
    assert result["knowledge_layer"] == "weak_signal"


def test_multi_source_single_trajectory_reaches_pending_but_not_validated(tmp_path):
    """With the trajectory gate relaxed to one briefing, a well-sourced claim
    passes every gate and distillation moves it to pending_user_confirm; it is
    not marked validated until a user confirms it."""
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing(source_count=2))[0]

    gate = service.record_validation_gate(candidate_id)
    assert gate["validated"] is True
    assert "fewer_than_two_independent_trajectories" not in gate["failures"]
    assert gate["knowledge_layer"] == "validated_knowledge"

    distill = service.distill_candidates()
    candidate = store.get_domain_knowledge_candidate(candidate_id)
    assert candidate["status"] == "pending_user_confirm"
    # validated counts the validated_knowledge LAYER (gate passed); the
    # candidate STATUS stays pending until a user confirms.
    assert distill["validated"] == 1
    assert distill["weak_signal"] == 0


def test_marketing_heavy_candidate_fails_authority_and_domain_gates(tmp_path):
    """Two marketing-funnel URLs are neither two domains nor enough authority."""
    from search_assistant.contracts import DomainKnowledgeCandidate, DomainKnowledgeEvidence

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    now = "2026-08-17T00:00:00Z"
    candidate = DomainKnowledgeCandidate(
        id="knowledge-marketing",
        topic="工业智能体",
        claim="该智能体方案已被广泛采用。",
        applies_when="test",
        evidence=[
            DomainKnowledgeEvidence(title="推广文 A", url="https://mp.weixin.qq.com/a", provider="wechat", retrieved_at=now),
            DomainKnowledgeEvidence(title="推广文 B", url="https://mp.weixin.qq.com/b", provider="wechat", retrieved_at=now),
            DomainKnowledgeEvidence(title="推广文 C", url="https://toutiao.com/c", provider="toutiao", retrieved_at=now),
        ],
        contradictions=[],
        confidence="medium",
        status="candidate",
        source_ids=["briefing-a", "briefing-b"],
        fingerprint="marketing-heavy-fingerprint",
        created_at=now,
        updated_at=now,
    )
    store.add_domain_knowledge_candidate(candidate)
    service = DomainKnowledgeCandidateService(store)

    result = service.record_validation_gate(candidate.id)

    # 3 marketing URLs: 2 domains (weixin/toutiao) but effective count 1.0 < 2.
    assert result["validated"] is False
    assert "insufficient_source_authority" in result["failures"]
    assert result["effective_source_count"] == 1.0
    assert result["source_domain_count"] == 2


def test_authoritative_source_offsets_marketing_sources(tmp_path):
    """One paper + one marketing URL passes the authority gate (3 + 1/3 >= 2)."""
    from search_assistant.contracts import DomainKnowledgeCandidate, DomainKnowledgeEvidence

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    now = "2026-08-17T00:00:00Z"
    candidate = DomainKnowledgeCandidate(
        id="knowledge-authority-mix",
        topic="工业智能体",
        claim="受控工单编排需要人工审批与回滚。",
        applies_when="test",
        evidence=[
            DomainKnowledgeEvidence(title="arXiv 论文", url="https://arxiv.org/abs/2508.0001", provider="arxiv", retrieved_at=now),
            DomainKnowledgeEvidence(title="推广文", url="https://mp.weixin.qq.com/x", provider="wechat", retrieved_at=now),
        ],
        contradictions=[],
        confidence="medium",
        status="candidate",
        source_ids=["briefing-a", "briefing-b"],
        fingerprint="authority-mix-fingerprint",
        created_at=now,
        updated_at=now,
    )
    store.add_domain_knowledge_candidate(candidate)
    service = DomainKnowledgeCandidateService(store)

    result = service.record_validation_gate(candidate.id)

    assert "insufficient_source_authority" not in result["failures"]
    assert "fewer_than_two_independent_sources" not in result["failures"]
    # effective_source_count is rounded to 3 decimals in the gate record.
    assert abs(result["effective_source_count"] - (3.0 + 1.0 / 3.0)) < 0.001
    # Two distinct domains + two trajectories + enough authority -> validated.
    assert result["validated"] is True
    assert result["failures"] == []


def test_rephrased_claim_merges_across_briefings_for_cross_trajectory_gate(tmp_path):
    """A rephrased claim from a second briefing on the same topic merges into the
    same candidate, so the cross-trajectory gate can actually fire."""
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    first = _briefing(source_count=2, briefing_id="briefing-1", url_prefix="run-a")
    second = _briefing(source_count=2, briefing_id="briefing-2", url_prefix="run-b")
    second.synthesis.themes[0].analysis = (
        "Industrial AI systems maintain a human approval gate before writing work "
        "orders into MES production execution."
    )

    first_id = service.capture_briefing(first)[0]
    second_id = service.capture_briefing(second)[0]

    assert first_id == second_id
    candidate = store.get_domain_knowledge_candidate(first_id)
    assert service.independent_trajectory_count(candidate) == 2
    result = service.validate(first_id)
    assert result["validated"] is True
    assert result["failures"] == []


def test_different_topic_claim_does_not_fuzzy_merge(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    first_id = service.capture_briefing(_briefing(source_count=2, briefing_id="briefing-1", url_prefix="a"))[0]
    second_id = service.capture_briefing(
        _briefing(topic="vector store latency", source_count=2, briefing_id="briefing-2", url_prefix="b")
    )[0]

    assert first_id != second_id
    assert len(store.list_domain_knowledge_candidates()) == 2


def test_chinese_rephrased_claim_merges_across_briefings(tmp_path):
    """Rephrased Chinese claims on the same topic merge via the trigram path."""
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    first = _briefing(topic="工业智能体 MES 工单编排 生产协同", source_count=2, briefing_id="briefing-1", url_prefix="za")
    first.synthesis.themes[0].analysis = "材料要求工业智能体低延时、高可靠、可审计，应部署为靠近设备侧的受控组件。"
    second = _briefing(topic="工业智能体 MES 工单编排 生产协同", source_count=2, briefing_id="briefing-2", url_prefix="zb")
    second.synthesis.themes[0].analysis = "今日头条文章明确工业AI智能体需有低延时、高可靠、可审计特性，并支持边缘推理。"

    first_id = service.capture_briefing(first)[0]
    second_id = service.capture_briefing(second)[0]

    assert first_id == second_id
    candidate = store.get_domain_knowledge_candidate(first_id)
    assert service.independent_trajectory_count(candidate) == 2


def test_same_topic_different_claims_do_not_fuzzy_merge(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    first = _briefing(topic="工业智能体 MES 工单编排 生产协同", source_count=2, briefing_id="briefing-1", url_prefix="na")
    first.synthesis.themes[0].analysis = "材料要求工业智能体低延时、高可靠、可审计，应部署为靠近设备侧的受控组件。"
    second = _briefing(topic="工业智能体 MES 工单编排 生产协同", source_count=2, briefing_id="briefing-2", url_prefix="nb")
    second.synthesis.themes[0].analysis = "本批材料明确把 OPC UA / REST API 与语义层统一作为打通 MES 的关键，应先解决数据标准化和语义映射。"

    first_id = service.capture_briefing(first)[0]
    second_id = service.capture_briefing(second)[0]

    assert first_id != second_id
    assert len(store.list_domain_knowledge_candidates()) == 2


def test_stale_evidence_blocks_validation(tmp_path):
    """Evidence older than the staleness window cannot validate."""
    from datetime import timedelta
    from search_assistant.contracts import DomainKnowledgeCandidate, DomainKnowledgeEvidence

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    now = datetime.now(UTC)
    stale = (now - timedelta(days=60)).isoformat()
    candidate = DomainKnowledgeCandidate(
        id="knowledge-stale",
        topic="工业智能体",
        claim="受控工单编排需要人工审批与回滚。",
        applies_when="test",
        evidence=[
            DomainKnowledgeEvidence(title="旧文档 A", url="https://example.com/old-a", provider="example", retrieved_at=stale),
            DomainKnowledgeEvidence(title="旧文档 B", url="https://example.org/old-b", provider="example", retrieved_at=stale),
        ],
        contradictions=[],
        confidence="medium",
        status="candidate",
        source_ids=["briefing-a", "briefing-b"],
        fingerprint="stale-fingerprint",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )
    store.add_domain_knowledge_candidate(candidate)
    service = DomainKnowledgeCandidateService(store)

    result = service.record_validation_gate(candidate.id)

    assert "stale_evidence" in result["failures"]
    assert result["validated"] is False


def test_candidate_list_can_surface_weak_signals_without_promoting_them(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing(source_count=1))[0]

    service.record_validation_gate(candidate_id)
    weak_signals = service.list_candidates(layer="weak_signal")

    assert [candidate["id"] for candidate in weak_signals] == [candidate_id]
    assert weak_signals[0]["status"] == "candidate"
    assert weak_signals[0]["knowledge_layer"] == "weak_signal"
    assert "independent corroborating sources" in weak_signals[0]["knowledge_layer_next_action"]


def test_cli_candidate_list_filters_by_knowledge_layer(tmp_path, monkeypatch):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing(source_count=1))[0]
    service.record_validation_gate(candidate_id)
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["knowledge-candidate-list", "--layer", "weak_signal", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert [candidate["id"] for candidate in output] == [candidate_id]
    assert output[0]["knowledge_layer"] == "weak_signal"


def test_candidate_approval_records_human_gate_and_release_ledger(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing(source_count=2))[0]
    # A second independent briefing makes the claim cross-trajectory validated.
    service.capture_briefing(_briefing(source_count=2, briefing_id="briefing-2", url_prefix="independent"))

    result = service.approve(
        candidate_id,
        reviewer="unit-reviewer",
        reason="safe to use as planning context",
        eval_gate={"passed": True, "reason": "unit eval passed", "report_path": "unit-evaluation-report.json"},
    )

    assert result["approved"] is True
    assert store.get_domain_knowledge_candidate(candidate_id)["status"] == "validated"
    gates = store.list_gate_records()
    gate_types = {gate["gate_type"] for gate in gates}
    assert "domain_knowledge_candidate_eval_release" in gate_types
    assert "domain_knowledge_candidate_validation" in gate_types
    assert "domain_knowledge_candidate_human_review" in gate_types
    release_entries = store.list_project_ledger_entries(entry_type="knowledge_release")
    assert len(release_entries) == 1
    assert release_entries[0]["status"] == "approved"
    assert result["gate_id"] in release_entries[0]["evidence_refs"]
    assert result["eval_gate"]["gate_id"] in release_entries[0]["evidence_refs"]


def test_candidate_approval_requires_eval_gate_before_release(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing(source_count=2))[0]

    result = service.approve(candidate_id, reviewer="unit-reviewer")

    assert result["approved"] is False
    assert result["eval_gate"]["passed"] is False
    assert store.get_domain_knowledge_candidate(candidate_id)["status"] == "candidate"
    gates = store.list_gate_records(gate_type="domain_knowledge_candidate_eval_release")
    assert len(gates) == 1
    assert gates[0]["result"] == "failed"
    assert not store.list_project_ledger_entries(entry_type="knowledge_release")


def test_evolve_command_backfills_existing_candidate_graph_links(tmp_path, capsys):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    now = "2026-08-06T00:00:00Z"
    candidate = DomainKnowledgeCandidate(
        id="knowledge-medical-imaging",
        topic="医学影像大模型",
        claim="DICOM/PACS data boundaries shape how medical imaging foundation models enter clinical workflows.",
        applies_when="Use when researching medical imaging AI deployment and validation.",
        evidence=[
            DomainKnowledgeEvidence(
                title="Medical imaging model deployment",
                url="https://example.com/medical-imaging-model",
                provider="example",
                retrieved_at=now,
            )
        ],
        contradictions=[],
        confidence="medium",
        status="candidate",
        source_ids=["briefing-medical-imaging"],
        fingerprint="medical-imaging-candidate-fingerprint",
        created_at=now,
        updated_at=now,
    )
    store.add_domain_knowledge_candidate(candidate)

    assert cli.main(["evolve", "--data-dir", str(tmp_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["domain_knowledge_candidate_graph_links"] == {"created": 1, "total": 1}
    links = store.list_domain_candidate_graph_links(candidate.id)
    assert len(links) == 1
    assert links[0]["graph_id"] == "medical-imaging-ai"
    assert links[0]["entity_id"] in {"dicom-pacs", "multimodal-medical-imaging-model"}


def test_candidate_graph_linking_ignores_unrelated_low_score_matches(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    now = "2026-08-13T00:00:00Z"
    candidate = DomainKnowledgeCandidate(
        id="knowledge-unrelated-chemistry",
        topic="盐酸的作用",
        claim="盐酸常用于调节酸碱度和参与化学实验，材料没有讨论 Agent、GraphRAG 或工具编排。",
        applies_when="Use when researching chemistry basics.",
        evidence=[
            DomainKnowledgeEvidence(
                title="Hydrochloric acid basics",
                url="https://example.com/hcl/one",
                provider="example",
                retrieved_at=now,
            ),
            DomainKnowledgeEvidence(
                title="Chemistry laboratory safety",
                url="https://example.com/hcl/two",
                provider="example",
                retrieved_at=now,
            ),
        ],
        contradictions=[],
        confidence="medium",
        status="candidate",
        source_ids=["briefing-hcl"],
        fingerprint="unrelated-chemistry-candidate-fingerprint",
        created_at=now,
        updated_at=now,
    )
    store.add_domain_knowledge_candidate(candidate)

    result = service.link_candidates_to_knowledge_graphs([candidate.id])

    assert result["candidates_considered"] == 1
    assert result["created_links"] == 0
    assert result["links"] == []
    assert store.list_domain_candidate_graph_links(candidate.id) == []


def test_learning_report_surfaces_trajectory_and_domain_candidate_status(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    DomainKnowledgeCandidateService(store).capture_briefing(_briefing(source_count=2))

    report = ReportService(store, output_dir=tmp_path / "reports").generate_markdown()

    assert "## Trajectory Evaluation Summary" in report
    assert "Immutable trajectories: 0" in report
    assert "## Domain Knowledge Candidates" in report
    assert "[candidate; confidence medium] industrial AI" in report
    assert "evidence: 2" in report
    assert "graph links: 1" in report


def _briefing(
    topic: str = "industrial AI",
    source_count: int = 2,
    briefing_id: str = "briefing-1",
    url_prefix: str | None = None,
) -> DailyBriefing:
    now = datetime.now(UTC)
    prefix = url_prefix or "industrial-ai"
    sources = [
        CollectedSource(
            id=f"source-{prefix}-{index}",
            topic_id="topic-1",
            user_id="user-1",
            title=f"Industrial AI source {index}",
            # Alternate host so multi-source briefings span at least two domains.
            url=f"https://{'example.org' if index % 2 == 0 else 'example.com'}/{prefix}/{index}",
            snippet="A source-backed workflow with review gates.",
            platform="web",
            provider=f"provider-{index}",
            query="industrial AI workflow",
            relevance_score=0.9,
            importance_score=0.9,
            retrieved_at=(now - timedelta(days=index)).isoformat().replace("+00:00", "Z"),
        )
        for index in range(1, source_count + 1)
    ]
    return DailyBriefing(
        id=briefing_id,
        topic_id="topic-1",
        user_id="user-1",
        chat_id="chat-1",
        topic=topic,
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
