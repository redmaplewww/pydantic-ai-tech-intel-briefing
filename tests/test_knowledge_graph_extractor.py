from __future__ import annotations

import os
import tempfile
from datetime import date, UTC, datetime

from search_assistant.contracts import (
    BriefingSynthesis,
    BriefingTheme,
    CollectedSource,
    DailyBriefing,
)
from search_assistant.knowledge_graph.extractor import (
    extract_entities,
    extract_relations,
    extend_graph_from_briefing,
)
from search_assistant.memory.store import MemoryStore


def _source(url: str, title: str, snippet: str = "") -> CollectedSource:
    return CollectedSource(
        id=f"src-{url}",
        topic_id="t-1",
        user_id="u-1",
        title=title,
        url=url,
        snippet=snippet,
        platform="test",
        provider="test",
        query="memory architectures",
        retrieved_at=datetime.now(UTC).isoformat(),
    )


def _briefing(topic: str = "AI agent memory architectures") -> DailyBriefing:
    now = datetime.now(UTC).isoformat()
    return DailyBriefing(
        id="brief-1",
        topic_id="t-1",
        user_id="u-1",
        chat_id="c-1",
        topic=topic,
        run_date=date(2026, 8, 17),
        search_directions=["memory architectures", "persistent memory systems"],
        keywords=["persistent memory", "vector store"],
        sources=[
            _source(
                "https://example.com/1",
                "Persistent memory systems for AI agents",
                "vector store and retrieval architectures",
            ),
            _source(
                "https://example.com/2",
                "Memory architectures in agent frameworks",
                "persistent memory and tool use",
            ),
        ],
        synthesis=BriefingSynthesis(
            search_content_summary=(
                "AI agents increasingly rely on persistent memory and vector store "
                "retrieval to maintain long-horizon context."
            ),
            short_summary="Persistent memory and vector store are core to agent architectures.",
            detailed_summary="",
            themes=[
                BriefingTheme(
                    name="记忆架构",
                    analysis=(
                        "persistent memory 与 vector store 的组合是 agent 记忆架构的关键路径，"
                        "tool use 需要跨会话上下文。"
                    ),
                    what_is_happening="行业内正在用 vector store 构建持久化记忆层。",
                    source_urls=["https://example.com/1", "https://example.com/2"],
                )
            ],
            key_signal_interpretation="memory + retrieval 成为标配",
            analysis_judgment="persistent memory 将决定 agent 的长期能力。",
            next_search_directions=["retrieval evaluation", "memory compression"],
            landing_suggestions=["评估 vector store 延迟", "建立记忆分层策略"],
        ),
        markdown="",
        created_at=now,
    )


def _store() -> MemoryStore:
    db_path = os.path.join(tempfile.mkdtemp(), "extractor-test.sqlite3")
    store = MemoryStore(db_path)
    store.initialize()
    return store


def test_extract_entities_finds_recurring_terms():
    entities = extract_entities(_briefing())
    names = {entity.name for entity in entities}
    assert "persistent memory" in names
    assert "vector store" in names
    assert all(entity.metadata.get("auto_extracted") for entity in entities)
    assert all(entity.evidence_refs == ["briefing:brief-1"] for entity in entities)


def test_extract_entities_filters_stop_words():
    entities = extract_entities(_briefing())
    names = {entity.name for entity in entities}
    for stop in ("the", "and", "系统", "技术"):
        assert stop not in names


def test_extract_entities_rejects_cjk_fragment_bigrams():
    """Bigram fragments of a longer CJK run (智能/能体 from 智能体) are dropped."""
    briefing = _briefing("智能体 编排")
    briefing.search_directions = ["智能体 任务编排", "智能体 工具调用"]
    briefing.keywords = ["智能体", "工作流"]
    briefing.sources = [
        _source(
            "https://example.com/cjk/1",
            "智能体 上下文管理",
            "智能体 需要受控工具调用与人工审批",
        )
    ]
    briefing.synthesis.themes[0].analysis = "智能体 编排需要受控工具调用与回滚机制。"

    names = {entity.name for entity in extract_entities(briefing)}
    assert "智能体" in names
    assert "能体" not in names
    assert "智能" not in names


def test_extract_entities_keeps_standalone_cjk_bigrams():
    """A 2-char CJK term that is not a fragment of a longer run stays an entity."""
    briefing = _briefing("视觉质检")
    briefing.search_directions = ["机器视觉 质检", "视觉 缺陷检测"]
    briefing.keywords = ["视觉", "缺陷"]
    briefing.sources = [_source("https://example.com/vision/1", "视觉 质检流程", "视觉 定位缺陷")]

    names = {entity.name for entity in extract_entities(briefing)}
    assert "视觉" in names


def test_is_meaningful_entity_name_filters_generic_and_fragments():
    from search_assistant.knowledge_graph.extractor import is_meaningful_entity_name

    assert is_meaningful_entity_name("受控工单编排") is True
    assert is_meaningful_entity_name("MES") is True
    assert is_meaningful_entity_name("ai") is False
    assert is_meaningful_entity_name("模型") is False
    assert is_meaningful_entity_name("能体", {"智能体"}) is False
    assert is_meaningful_entity_name("视觉", {"机器视觉"}) is False
    assert is_meaningful_entity_name("视觉", {"机器视觉质检"}) is False
    assert is_meaningful_entity_name("视觉", {"目标检测"}) is True
    # Overlapping CJK window artifacts are rejected; Latin-CJK pairs survive.
    assert is_meaningful_entity_name("大模 模型") is False
    assert is_meaningful_entity_name("端视 视觉") is False
    assert is_meaningful_entity_name("mes 工单") is True
    assert is_meaningful_entity_name("a2a 协议") is True
    assert is_meaningful_entity_name("persistent memory") is True


def test_query_relevant_skips_fragment_and_generic_entities():
    from search_assistant.contracts import DomainKnowledgeEntity, DomainKnowledgeGraph
    from search_assistant.knowledge_graph.service import DomainKnowledgeGraphService

    store = _store()
    graph = DomainKnowledgeGraph(
        id="auto-quality-test",
        name="自动提取图谱（质量测试）",
        description="test graph",
        overview="test overview",
        source="auto-extracted",
        version="1",
        entities=[
            DomainKnowledgeEntity(
                id="e1", name="智能体", entity_type="concept", aliases=[],
                summary="智能体 编排与工具调用", evidence_refs=["briefing:b1"], metadata={},
            ),
            DomainKnowledgeEntity(
                id="e2", name="能体", entity_type="concept", aliases=[],
                summary="切词碎片", evidence_refs=["briefing:b1"], metadata={},
            ),
            DomainKnowledgeEntity(
                id="e3", name="ai", entity_type="concept", aliases=[],
                summary="泛化词", evidence_refs=["briefing:b1"], metadata={},
            ),
            DomainKnowledgeEntity(
                id="e4", name="受控工单编排", entity_type="workflow", aliases=[],
                summary="受控工单编排 写回 MES", evidence_refs=["briefing:b1"], metadata={},
            ),
        ],
        relations=[],
    )
    store.upsert_domain_knowledge_graph(graph)
    service = DomainKnowledgeGraphService(store, hybrid_retrieval=True)
    hits = service.query_relevant("智能体 受控工单编排", limit=10, min_score=0.0)
    names = {hit.entity_name for hit in hits}
    assert "智能体" in names
    assert "受控工单编排" in names
    assert "能体" not in names
    assert "ai" not in names


def test_extract_relations_uses_co_occurrence():
    entities = extract_entities(_briefing())
    relations = extract_relations(entities, _briefing())
    by_name = {entity.name: entity for entity in entities}
    pairs = {
        (relation.source_entity_id, relation.target_entity_id) for relation in relations
    }
    if "persistent memory" in by_name and "vector store" in by_name:
        pair = (by_name["persistent memory"].id, by_name["vector store"].id)
        assert pair in pairs or (pair[1], pair[0]) in pairs


def test_extend_graph_creates_auto_graph_and_merges():
    store = _store()
    briefing = _briefing()
    result = extend_graph_from_briefing(store, briefing)
    assert result["graph_id"].startswith("auto-")
    assert result["added_entities"] > 0
    graphs = store.list_domain_knowledge_graphs()
    auto_graphs = [g for g in graphs if str(g["id"]).startswith("auto-")]
    assert len(auto_graphs) == 1

    # A second run on the SAME topic merges into the same graph without
    # duplicating entities.
    second_briefing = _briefing("AI agent memory architectures")
    second_briefing.id = "brief-2"
    result2 = extend_graph_from_briefing(store, second_briefing)
    assert result2["graph_id"] == result["graph_id"]
    graph = store.get_domain_knowledge_graph(result["graph_id"])
    assert graph is not None
    # "vector store" already exists from the first run -> not duplicated.
    names = [entity.name for entity in graph.entities]
    assert names.count("vector store") <= 1


def test_extend_graph_isolates_auto_graphs_by_topic():
    """Different topics must land in different auto graphs (no cross-topic merge)."""
    store = _store()
    first = extend_graph_from_briefing(store, _briefing("AI agent memory architectures"))
    second = extend_graph_from_briefing(store, _briefing("vector store latency"))

    assert first["graph_id"] != second["graph_id"]
    auto_graphs = [g for g in store.list_domain_knowledge_graphs() if str(g["id"]).startswith("auto-")]
    assert len(auto_graphs) == 2
    first_graph = store.get_domain_knowledge_graph(first["graph_id"])
    second_graph = store.get_domain_knowledge_graph(second["graph_id"])
    assert first_graph is not None and second_graph is not None
    # Every entity stays in the graph of the topic that produced it: the entity
    # summary records the source briefing topic.
    assert all("AI agent memory architectures" in (entity.summary or "") for entity in first_graph.entities)
    assert all("vector store latency" in (entity.summary or "") for entity in second_graph.entities)


def test_extend_graph_with_empty_briefing_returns_noop():
    store = _store()
    briefing = _briefing()
    briefing.sources = []
    briefing.synthesis.themes = []
    briefing.search_directions = []
    briefing.keywords = []
    # Synthesis fields are the only remaining text source; when they are blank
    # too, there is nothing to extract and the call must be a no-op.
    briefing.synthesis.search_content_summary = ""
    briefing.synthesis.short_summary = ""
    briefing.synthesis.detailed_summary = ""
    briefing.synthesis.key_signal_interpretation = ""
    briefing.synthesis.analysis_judgment = ""
    result = extend_graph_from_briefing(store, briefing)
    assert result["added_entities"] == 0


def test_extend_graph_lowers_occurrence_threshold_for_short_briefings():
    """Fewer than 10 sources drops the occurrence threshold from 3 to 2."""
    store = _store()
    result = extend_graph_from_briefing(store, _briefing())
    assert result["min_occurrences_used"] == 2  # test helper has 2 sources


def test_extend_graph_keeps_occurrence_threshold_for_large_briefings():
    store = _store()
    briefing = _briefing()
    briefing.sources = [_source(f"https://example.com/many/{i}", f"Source {i}", "more content") for i in range(12)]
    result = extend_graph_from_briefing(store, briefing)
    assert result["min_occurrences_used"] == 3


def test_extend_graph_caps_auto_graph_entities():
    from search_assistant.contracts import DomainKnowledgeEntity, DomainKnowledgeGraph

    store = _store()
    entities = [
        DomainKnowledgeEntity(
            id=f"cap-e{i}",
            name=f"term{i:03d}",
            entity_type="concept",
            aliases=[],
            summary=f"term{i:03d}",
            evidence_refs=["briefing:cap"],
            metadata={"auto_extracted": True, "occurrences": 10 - (i % 5)},
        )
        for i in range(55)
    ]
    graph = DomainKnowledgeGraph(
        id="auto-cap-test",
        name="cap test",
        description="cap test graph",
        overview="cap test",
        source="auto-extracted",
        version="1",
        entities=entities,
        relations=[],
    )
    store.upsert_domain_knowledge_graph(graph)

    result = extend_graph_from_briefing(store, _briefing("cap test"))

    assert result["graph_id"] == "auto-cap-test"
    merged = store.get_domain_knowledge_graph("auto-cap-test")
    assert merged is not None
    assert len(merged.entities) <= 60
