"""Tests for the Path-C candidate pipeline: relaxed trajectory gate,
semantic quality gating, pending-user-confirm promotion, and topic-level
semantic merging."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from search_assistant.contracts import (
    BriefingSynthesis,
    BriefingTheme,
    CollectedSource,
    DailyBriefing,
)
from search_assistant.evolution.semantics import (
    SEMANTIC_KINDS,
    SemanticVerdict,
    classify_semantic_quality,
    is_noise_verdict,
)
from search_assistant.evolution.service import (
    DomainKnowledgeCandidateService,
    _MIN_INDEPENDENT_TRAJECTORIES,
)


def _briefing(briefing_id: str, url_prefix: str, source_count: int = 2) -> DailyBriefing:
    now = datetime.now(UTC)
    sources = [
        CollectedSource(
            id=f"source-{url_prefix}-{index}",
            topic_id="topic-1",
            user_id="user-1",
            title=f"Industrial AI source {index}",
            url=f"https://{'example.org' if index % 2 == 0 else 'example.com'}/{url_prefix}/{index}",
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


# -- semantic classification (rule fallback) ----------------------------------


def test_rule_judge_flags_marketing_rhetoric():
    verdict = classify_semantic_quality("这个产品遥遥领先，颠覆行业，全网首发限时抢购！", "工业智能体")
    assert verdict.kind == "marketing"
    assert verdict.judge == "rules"
    assert is_noise_verdict(verdict)


def test_rule_judge_accepts_technical_mechanism_claim():
    verdict = classify_semantic_quality(
        "OPC UA 语义层统一打通 MES，工单审批后回滚需人工确认，接口延迟低于 200ms。",
        "工业智能体 MES",
    )
    assert verdict.kind == "technical"
    assert not is_noise_verdict(verdict)


def test_rule_judge_flags_vague_short_claim():
    verdict = classify_semantic_quality("总的来说，未来发展值得关注。", "通用")
    assert verdict.kind == "vague"
    assert is_noise_verdict(verdict)


# -- semantic classification (LLM primary) ------------------------------------


def test_llm_verdict_is_primary_when_parsable():
    def fake_llm(instructions, payload):
        assert "语义质检器" in instructions
        return '{"kind": "marketing", "confidence": "high", "reason": "rhetoric"}'

    verdict = classify_semantic_quality("任意文本", "任意主题", llm_runner=fake_llm)
    assert verdict.kind == "marketing"
    assert verdict.judge == "llm"


def test_llm_unparsable_falls_back_to_rules():
    def fake_llm(instructions, payload):
        return "not json at all"

    verdict = classify_semantic_quality("这个产品遥遥领先！", "t", llm_runner=fake_llm)
    assert verdict.judge == "rules"
    assert verdict.kind == "marketing"


def test_semantic_kinds_are_stable():
    assert set(SEMANTIC_KINDS) == {"technical", "marketing", "news", "vague"}
    assert isinstance(SemanticVerdict("technical", "high", "ok", "llm"), SemanticVerdict)


# -- trajectory gate constant -------------------------------------------------


def test_trajectory_gate_is_relaxed_to_one():
    assert _MIN_INDEPENDENT_TRAJECTORIES == 1


# -- pending confirm flow ------------------------------------------------------


def test_distill_promotes_to_pending_not_validated(tmp_path):
    from search_assistant.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing("briefing-1", "run-1"))[0]

    distill = service.distill_candidates()
    # validated counts the validated_knowledge LAYER (the gate passed); the
    # candidate STATUS stays pending_user_confirm until a user confirms.
    assert store.get_domain_knowledge_candidate(candidate_id)["status"] == "pending_user_confirm"
    assert distill["validated"] == 1
    assert distill["weak_signal"] == 0


def test_confirm_candidate_releases_to_validated(tmp_path):
    from search_assistant.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing("briefing-1", "run-1"))[0]
    service.distill_candidates()

    confirmed = service.confirm_candidate(candidate_id, reviewer="unit-user")
    assert confirmed["confirmed"] is True
    assert store.get_domain_knowledge_candidate(candidate_id)["status"] == "validated"
    assert store.list_gate_records(gate_type="domain_knowledge_candidate_human_review")
    assert store.list_project_ledger_entries(entry_type="knowledge_release")


def test_confirm_rejects_non_pending_candidate(tmp_path):
    from search_assistant.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)
    candidate_id = service.capture_briefing(_briefing("briefing-1", "run-1"))[0]

    try:
        service.confirm_candidate(candidate_id)
    except ValueError:
        pass
    else:
        raise AssertionError("confirm_candidate should reject a candidate that is not pending_user_confirm")


# -- topic semantic merging ----------------------------------------------------


class _FakeEmbedSimilar:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0], [0.98, 0.02, 0.0]]


class _FakeEmbedDissimilar:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_topic_semantic_similarity_uses_embedding_when_literal_fails(tmp_path):
    from search_assistant.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store, embedding_provider=_FakeEmbedSimilar())

    assert service._topic_semantically_similar(
        "AI Agent Harness 上下文工程 工具调用可靠性", "上下文工程"
    ) is True
    assert service._topic_semantically_similar("a b c d", "x y z w") is True


def test_topic_semantic_similarity_rejects_dissimilar_vectors(tmp_path):
    from search_assistant.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store, embedding_provider=_FakeEmbedDissimilar())
    assert service._topic_semantically_similar("alpha beta", "gamma delta") is False


def test_topic_semantic_similarity_offline_when_provider_empty(tmp_path):
    from search_assistant.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeCandidateService(store)  # Null provider -> empty vectors
    assert service._topic_semantically_similar("anything", "anything else") is False
