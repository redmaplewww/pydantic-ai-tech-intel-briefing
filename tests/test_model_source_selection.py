"""Tests for the primary-evidence quota in model source selection."""

from __future__ import annotations

from search_assistant.briefing.service import DailyBriefingService
from search_assistant.contracts import CollectedSource


def _source(index: int, url: str, title: str, snippet: str = "") -> CollectedSource:
    return CollectedSource(
        id=f"src-{index}",
        topic_id="t-1",
        user_id="u-1",
        title=title,
        url=url,
        snippet=snippet,
        platform="web",
        provider="test",
        query="q",
        relevance_score=1.0 - index / 100,
        importance_score=1.0 - index / 100,
        retrieved_at="2026-08-17T00:00:00Z",
    )


def _service() -> DailyBriefingService:
    # _select_model_sources only reads its arguments; store/search are unused.
    return DailyBriefingService(store=None, search_client=None)


def test_select_model_sources_prefers_primary_and_caps_weak():
    service = _service()
    primary = [_source(i, f"https://arxiv.org/abs/2508.{i:04d}", f"Paper {i}") for i in range(10)]
    neutral = [_source(100 + i, f"https://example.org/news/{i}", f"Neutral {i}") for i in range(5)]
    weak = [_source(200 + i, f"https://mp.weixin.qq.com/s/{i}", f"推广 {i}") for i in range(3)]
    ranked = primary + neutral + weak

    selected = service._select_model_sources(ranked, limit=12)

    assert len(selected) == 12
    assert all(source.url.startswith("https://arxiv.org") for source in selected[:10])
    # Two neutral sources fill the remainder; no weak source is included.
    assert sum(source.url.startswith("https://example.org") for source in selected) == 2
    assert sum(source.url.startswith("https://mp.weixin.qq.com") for source in selected) == 0


def test_select_model_sources_weak_cap_when_primary_limited():
    service = _service()
    primary = [_source(i, f"https://github.com/org/repo{i}", f"Repo {i}") for i in range(2)]
    weak = [_source(200 + i, f"https://mp.weixin.qq.com/s/{i}", f"推广 {i}") for i in range(5)]
    neutral = [_source(100 + i, f"https://example.org/a/{i}", f"News {i}") for i in range(3)]

    selected = service._select_model_sources(primary + neutral + weak, limit=12)

    assert len(selected) == 7  # 2 primary + 3 neutral + 2 weak (cap)
    assert sum(source.url.startswith("https://mp.weixin.qq.com") for source in selected) == 2
    assert selected[0].url.startswith("https://github.com")


def test_select_model_sources_all_weak_still_fills_limit():
    service = _service()
    weak = [_source(200 + i, f"https://mp.weixin.qq.com/s/{i}", f"推广 {i}") for i in range(5)]

    selected = service._select_model_sources(weak, limit=4)

    assert len(selected) == 4
    assert all(source.url.startswith("https://mp.weixin.qq.com") for source in selected)


def test_select_model_sources_handles_empty_and_zero_limit():
    service = _service()
    assert service._select_model_sources([], limit=12) == []
    assert service._select_model_sources([_source(1, "https://example.org/x", "X")], limit=0) == []
