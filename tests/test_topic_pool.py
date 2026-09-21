"""Tests for the briefing-loop topic pool resolution."""

from __future__ import annotations

from search_assistant.cli import _topic_pool


def test_topic_pool_keeps_primary_and_appends_topics():
    pool = _topic_pool("工业智能体", "GraphRAG,医学影像,工业智能体", None)
    assert pool == ["工业智能体", "GraphRAG", "医学影像"]


def test_topic_pool_reads_topics_file(tmp_path):
    topics_file = tmp_path / "pool.txt"
    topics_file.write_text("工业智能体\nGraphRAG\n\n医学影像\n", encoding="utf-8")
    pool = _topic_pool("工业智能体", None, str(topics_file))
    assert pool == ["工业智能体", "GraphRAG", "医学影像"]


def test_topic_pool_merges_both_sources_and_deduplicates(tmp_path):
    topics_file = tmp_path / "pool.txt"
    topics_file.write_text("GraphRAG\n医学影像\n", encoding="utf-8")
    pool = _topic_pool("工业智能体", "GraphRAG,边缘端", str(topics_file))
    assert pool == ["工业智能体", "GraphRAG", "边缘端", "医学影像"]


def test_topic_pool_primary_only_when_no_extra_sources():
    assert _topic_pool("工业智能体", None, None) == ["工业智能体"]
