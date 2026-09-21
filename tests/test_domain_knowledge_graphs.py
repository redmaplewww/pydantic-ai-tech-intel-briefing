import json
import sys
from io import BytesIO, TextIOWrapper
from pathlib import Path

from search_assistant.knowledge_graph.service import DomainKnowledgeGraphService
from search_assistant.memory.store import MemoryStore


def test_seeds_three_default_domain_knowledge_graphs(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeGraphService(store)

    result = service.seed_default_graphs()
    graphs = service.list_graphs()

    assert result["seeded_graphs"] == 3
    assert [graph["id"] for graph in graphs] == [
        "agent-engineering",
        "industrial-ai",
        "medical-imaging-ai",
    ]
    assert {graph["entity_count"] for graph in graphs} == {7}
    assert all(graph["relation_count"] >= 5 for graph in graphs)


def test_query_returns_entity_with_relationship_context(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeGraphService(store)
    service.seed_default_graphs()

    hits = service.query("MES 工单 写回", domain_id="industrial-ai", limit=3)

    assert hits
    assert hits[0].graph_id == "industrial-ai"
    assert hits[0].entity_id in {"mes", "controlled-work-order-orchestration"}
    combined_relations = "\n".join(hits[0].incoming_relations + hits[0].outgoing_relations)
    assert "工单" in hits[0].summary or "工单" in combined_relations
    assert "MES" in hits[0].entity_name or "MES" in hits[0].summary or "MES" in combined_relations


def test_relevant_query_keeps_strong_matches_and_filters_unrelated_terms(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeGraphService(store)
    service.seed_default_graphs()

    harness_hits = service.query_relevant("harness是什么", limit=3)
    unrelated_hits = service.query_relevant("盐酸的作用", limit=3)

    assert any(hit.graph_id == "agent-engineering" and hit.entity_id == "harness-engineering" for hit in harness_hits)
    assert unrelated_hits == []


def test_export_markdown_preserves_entities_and_graph_links(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeGraphService(store)
    service.seed_default_graphs(["agent-engineering"])

    markdown = service.export_markdown("agent-engineering")

    assert "# AI Agent 工程 知识图谱" in markdown
    assert "GraphRAG" in markdown
    assert "[Harness 工程](#harness-工程)" in markdown
    assert "实体-关系" in markdown


def test_cli_seeds_queries_and_exports_domain_graph(monkeypatch, tmp_path):
    from search_assistant import cli

    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)
    assert cli.main(["knowledge-graph-seed", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    seed_output = json.loads(stream.buffer.getvalue().decode("utf-8"))
    assert seed_output["seeded_graphs"] == 3

    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)
    assert (
        cli.main(
            [
                "knowledge-graph-query",
                "医学影像大模型 DICOM",
                "--domain",
                "medical-imaging-ai",
                "--limit",
                "2",
                "--data-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    stream.flush()
    hits = json.loads(stream.buffer.getvalue().decode("utf-8"))
    assert hits[0]["graph_id"] == "medical-imaging-ai"
    assert hits[0]["entity_id"] in {"multimodal-medical-imaging-model", "dicom-pacs"}

    output_path = tmp_path / "exported" / "industrial-ai.md"
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)
    assert (
        cli.main(
            [
                "knowledge-graph-export",
                "industrial-ai",
                "--output",
                str(output_path),
                "--data-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    stream.flush()
    export_output = json.loads(stream.buffer.getvalue().decode("utf-8"))
    assert Path(export_output["path"]) == output_path
    assert "受控工单编排" in output_path.read_text(encoding="utf-8")
