from __future__ import annotations

import hashlib
import os
from typing import Protocol


def _embedding_cache_key(text: str) -> str:
    """Return a stable hash for a piece of text used as an embedding cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors without extra dependencies."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b) ** 0.5


class EmbeddingProvider(Protocol):
    """Protocol for turning a batch of texts into dense vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class NullEmbeddingProvider:
    """Provider that returns empty vectors, effectively disabling semantic matching."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


class OpenAIEmbeddingProvider:
    """OpenAI-compatible embedding provider (also works with compatible endpoints).

    Embedding failures (e.g. an endpoint without an embedding model, such as the
    DeepSeek chat-only API) are detected on the first call, after which the
    provider returns empty vectors so callers can fall back to literal matching
    without repeatedly paying the failure cost on every query.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "text-embedding-3-small",
        timeout: float = 30.0,
        failure_threshold: int = 1,
    ):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self._failure_threshold = max(1, failure_threshold)
        self._consecutive_failures = 0
        self._disabled = False

    def _disable(self) -> None:
        self._disabled = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._disabled:
            return [[] for _ in texts]
        # OpenAI embedding API supports up to 2048 texts per request; keep a safe batch size.
        batch_size = 64
        results: list[list[float]] = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self._client.embeddings.create(model=self.model, input=batch)
                results.extend(item.embedding for item in response.data)
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._disable()
            return [[] for _ in texts]
        self._consecutive_failures = 0
        return results


def build_embedding_provider(
    enabled: bool | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> EmbeddingProvider:
    """Build an embedding provider from explicit arguments or environment variables.

    Args:
        enabled: Whether semantic matching is enabled. Defaults to True unless
            ``SEARCH_ASSISTANT_EMBEDDING_ENABLED`` is set to ``false``/``0``/``no``.
        api_key: OpenAI-compatible API key. Falls back to ``OPENAI_API_KEY``,
            then ``DEEPSEEK_API_KEY``.
        base_url: OpenAI-compatible base URL. Falls back to ``OPENAI_BASE_URL``,
            then ``DEEPSEEK_BASE_URL``.
        model: Embedding model name. Falls back to ``OPENAI_EMBEDDING_MODEL``
            then ``text-embedding-3-small``.
    """
    if enabled is None:
        raw = os.getenv("SEARCH_ASSISTANT_EMBEDDING_ENABLED", "true").strip().lower()
        enabled = raw not in {"false", "0", "no", "off"}

    if not enabled:
        return NullEmbeddingProvider()

    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return NullEmbeddingProvider()

    return OpenAIEmbeddingProvider(
        api_key=key,
        base_url=base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL"),
        model=model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )


def embedding_provider_from_settings(settings: object) -> EmbeddingProvider:
    """Build an embedding provider from a ``Settings``-like object.

    Prefer this over :func:`build_embedding_provider` inside services that
    already hold a ``Settings`` instance: it reads the resolved values (which
    include the ``DEEPSEEK_*`` fallbacks applied by ``Settings.from_env``)
    instead of relying on raw ``os.environ``, which does not contain
    ``.env.local`` entries.  The explicit ``OPENAI_*`` settings win; otherwise
    the resolved DeepSeek credentials from ``Settings`` are used (``.env.local``
    values are loaded into ``Settings``, not into ``os.environ``), and only as a
    last resort does the function fall back to the process environment.
    """
    enabled = bool(getattr(settings, "embedding_enabled", True))
    if not enabled:
        return NullEmbeddingProvider()
    api_key = (
        getattr(settings, "openai_api_key", None)
        or getattr(settings, "deepseek_api_key", None)
        or os.getenv("DEEPSEEK_API_KEY")
    )
    if not api_key:
        return NullEmbeddingProvider()
    base_url = (
        getattr(settings, "openai_base_url", None)
        or getattr(settings, "deepseek_base_url", None)
        or os.getenv("DEEPSEEK_BASE_URL")
    )
    model = getattr(settings, "openai_embedding_model", None) or "text-embedding-3-small"
    return OpenAIEmbeddingProvider(api_key=api_key, base_url=base_url, model=model)
