import os
from pathlib import Path

import pytest

from search_assistant.knowledge_graph.embedding import (
    NullEmbeddingProvider,
    OpenAIEmbeddingProvider,
    build_embedding_provider,
    cosine_similarity,
)
from search_assistant.knowledge_graph.service import DomainKnowledgeGraphService
from search_assistant.knowledge_graph.seeds import default_domain_graphs
from search_assistant.memory.store import MemoryStore


class _FakeEmbeddingProvider:
    """Deterministic embedding provider for tests.

    Each text is turned into a unit vector where dimensions encode simple
    cross-lingual semantic signals.  This is not a real embedding model but is
    enough to verify that the service uses semantic similarity to bridge English
    queries and Chinese entities.
    """

    # Multiple surface forms share the same dimension to simulate a cross-lingual
    # embedding space where "agent loop" and "工具调用循环" map nearby.
    _VOCAB: dict[str | tuple[str, ...], int] = {
        ("agent", "智能体", "tool", "工具"): 0,
        ("loop", "循环"): 1,
        ("memory", "记忆"): 2,
        ("context", "上下文"): 3,
        ("harness", "工程"): 4,
        "llm": 5,
        "function": 6,
        "calling": 7,
        "persistent": 8,
    }

    def __init__(self, dim: int = 32):
        self.model = "fake"
        self._dim = dim
        self._token_to_dim: dict[str, int] = {}
        for key, idx in self._VOCAB.items():
            if isinstance(key, tuple):
                for token in key:
                    self._token_to_dim[token] = idx
            else:
                self._token_to_dim[key] = idx

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            text_lower = text.lower()
            tokens = {idx for token, idx in self._token_to_dim.items() if token in text_lower}
            vec = [0.0] * self._dim
            for idx in tokens:
                if idx < self._dim:
                    vec[idx] = 1.0
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            results.append(vec)
        return results


def test_cosine_similarity_handles_empty_and_zero_vectors():
    assert cosine_similarity([], [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], []) == 0.0
    assert cosine_similarity([1.0, 2.0], [3.0, 4.0, 5.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_similarity_for_perpendicular_and_identical_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-9)
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0, abs=1e-9)


def test_null_provider_returns_empty_vectors():
    provider = NullEmbeddingProvider()
    assert provider.embed(["hello", "world"]) == [[], []]


def test_build_embedding_provider_returns_null_when_disabled():
    provider = build_embedding_provider(enabled=False)
    assert isinstance(provider, NullEmbeddingProvider)


def test_build_embedding_provider_returns_null_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = build_embedding_provider()
    assert isinstance(provider, NullEmbeddingProvider)


def test_build_embedding_provider_returns_openai_with_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    provider = build_embedding_provider(model="text-embedding-3-small")
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.model == "text-embedding-3-small"


def test_build_embedding_provider_falls_back_to_deepseek_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    provider = build_embedding_provider()
    assert isinstance(provider, OpenAIEmbeddingProvider)


def test_build_embedding_provider_prefers_openai_over_deepseek(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    provider = build_embedding_provider()
    assert isinstance(provider, OpenAIEmbeddingProvider)


class _FakeSettings:
    """Minimal Settings-like object; os.environ is left empty by the test."""

    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


def test_embedding_provider_from_settings_uses_resolved_deepseek_credentials(monkeypatch):
    """A DeepSeek key that lives only in Settings (like .env.local) must be used."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = _FakeSettings(
        embedding_enabled=True,
        openai_api_key=None,
        openai_base_url=None,
        deepseek_api_key="sk-from-settings",
        deepseek_base_url="https://api.deepseek.com/v1",
    )
    from search_assistant.knowledge_graph.embedding import embedding_provider_from_settings

    provider = embedding_provider_from_settings(settings)
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider._client.api_key == "sk-from-settings"


def test_embedding_provider_from_settings_returns_null_without_any_credentials(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = _FakeSettings(embedding_enabled=True)
    from search_assistant.knowledge_graph.embedding import embedding_provider_from_settings

    assert isinstance(embedding_provider_from_settings(settings), NullEmbeddingProvider)


def test_service_uses_semantic_match_for_english_query_and_chinese_entity(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeGraphService(store, embedding_provider=_FakeEmbeddingProvider())
    service.seed_default_graphs(["agent-engineering"])

    # "agent loop" has no literal overlap with the Chinese entity "工具调用循环"
    # (Tool Calling Loop), but the fake cross-lingual embedding maps them nearby.
    hits = service.query_relevant("agent loop", limit=3, min_score=5.0)

    assert any(hit.entity_id == "tool-loop" for hit in hits), [
        (hit.entity_id, hit.score, hit.matched_aliases) for hit in hits
    ]


def test_semantic_match_does_not_pollute_unrelated_queries(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeGraphService(store, embedding_provider=_FakeEmbeddingProvider())
    service.seed_default_graphs()

    hits = service.query_relevant("盐酸的作用", limit=3, min_score=5.0)
    assert hits == []


def test_service_falls_back_to_literal_matching_without_provider(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DomainKnowledgeGraphService(store, embedding_provider=NullEmbeddingProvider())
    service.seed_default_graphs()

    # Literal Chinese query should still work.
    hits = service.query_relevant("harness是什么", limit=3, min_score=5.0)
    assert any(hit.entity_id == "harness-engineering" for hit in hits)
