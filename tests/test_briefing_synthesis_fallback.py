"""Tests for the briefing-synthesis fallback fixes (Path D):

1. A dedicated synthesis timeout (longer than the general call timeout) so
   large briefings do not silently degrade to deterministic fallback.
2. Fallback degradation is audited (gate + ledger + synthesis_source marker).
3. Fallback detailed-summary headings keep stable fixed suffixes.
"""

from __future__ import annotations

from datetime import date

from search_assistant.briefing.service import DailyBriefingService
from search_assistant.config import Settings
from search_assistant.contracts import CollectedSource
from search_assistant.memory.store import MemoryStore
from search_assistant.runtime import DeepSeekChatRuntime


# -- fix 1: dedicated synthesis timeout ---------------------------------------


def test_settings_has_briefing_synthesis_timeout_default():
    settings = Settings.from_env({})
    assert settings.briefing_synthesis_timeout_seconds == 120.0


def test_settings_reads_briefing_synthesis_timeout_from_env():
    settings = Settings.from_env({"BRIEFING_SYNTHESIS_TIMEOUT_SECONDS": "150"})
    assert settings.briefing_synthesis_timeout_seconds == 150.0


def test_runtime_synthesis_timeout_is_longer_than_general():
    runtime = DeepSeekChatRuntime(
        api_key="k",
        timeout_seconds=60.0,
        briefing_synthesis_timeout_seconds=120.0,
    )
    assert runtime.briefing_synthesis_timeout_seconds == 120.0
    assert runtime.timeout_seconds == 60.0


def test_runtime_synthesis_timeout_falls_back_to_general_when_unset():
    runtime = DeepSeekChatRuntime(api_key="k", timeout_seconds=60.0)
    assert runtime.briefing_synthesis_timeout_seconds == 60.0


def test_runtime_synthesis_timeout_never_below_general():
    runtime = DeepSeekChatRuntime(
        api_key="k",
        timeout_seconds=90.0,
        briefing_synthesis_timeout_seconds=30.0,
    )
    assert runtime.briefing_synthesis_timeout_seconds == 90.0


# -- fix 2: fallback is audited ------------------------------------------------


def test_fallback_records_gate_and_ledger(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DailyBriefingService(store, _EmptySearchClient(), runtime=_NoneRuntime())

    source = _source("意图识别", "意图识别和槽位抽取是 NLU 关键部分，影响 Agent 交互质量。")
    service.run("意图识别", "u-1", "c-1", run_date=date(2026, 8, 9))

    gates = store.list_gate_records(gate_type="briefing_synthesis_fallback")
    assert gates, "fallback should be recorded as a gate"
    assert gates[0]["result"] == "failed"
    ledgers = store.list_project_ledger_entries(entry_type="briefing_synthesis_fallback")
    assert ledgers, "fallback should be recorded in the project ledger"
    assert ledgers[0]["status"] == "degraded"


def test_synthesis_source_marker_reflects_fallback(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DailyBriefingService(store, _EmptySearchClient(), runtime=_NoneRuntime())
    service.run("意图识别", "u-1", "c-1", run_date=date(2026, 8, 9))
    assert getattr(service, "last_synthesis_fallback", False) is True


# -- fix 3: deterministic fallback headings -----------------------------------


def _source(title: str, snippet: str) -> CollectedSource:
    return CollectedSource(
        id="src-1",
        topic_id="topic-1",
        user_id="u-1",
        title=title,
        url="https://example.com/src",
        snippet=snippet,
        platform="web",
        provider="browser-bing",
        query="q",
        relevance_score=1.0,
        importance_score=8.0,
        retrieved_at="2026-08-09T00:00:00+00:00",
    )


def test_display_generic_evidence_theme_name_uses_fixed_suffix():
    # The fallback keeps stable, predictable headings instead of a
    # hand-maintained evidence-word list.
    name = DailyBriefingService._display_generic_evidence_theme_name(
        "AI 编译器", "技术底座、数据与开源生态"
    )
    assert name == "AI 编译器的技术底座、数据与开源生态"


def test_display_generic_evidence_theme_name_does_not_repeat_topic():
    name = DailyBriefingService._display_generic_evidence_theme_name(
        "意图识别", "综合产业动态与待核验证据"
    )
    assert name == "意图识别的待核验证据线索"
    assert "意图识别的意图识别" not in name


class _EmptySearchClient:
    def search(self, query, limit=5):
        return []


class _NoneRuntime:
    def synthesize_briefing(self, topic, sources, context):
        return None
