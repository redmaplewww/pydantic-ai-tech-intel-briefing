from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from search_assistant.contracts import BriefingSynthesis, BriefingUnderstanding, CollectedSource


AgentRunner = Callable[[str, str, str, str, str, float, int, float], str]


class ModelRuntimeError(RuntimeError):
    """Raised when the configured model runtime cannot return a usable response."""


class TextModelRuntime(Protocol):
    def run_text_task(
        self,
        instructions: str,
        payload: dict[str, object],
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> str:
        ...


class AgentRuntime(TextModelRuntime, Protocol):
    def plan_search_queries(self, question: str, context: dict[str, object]) -> list[str]:
        ...

    def generate_answer(self, question: str, context: dict[str, object]) -> str:
        ...

    def calibrate(self, draft: str, context: dict[str, object]) -> dict[str, object]:
        ...

    def review_answer(self, answer: str, context: dict[str, object]) -> dict[str, object]:
        ...


class BriefingRuntime(TextModelRuntime, Protocol):
    def understand_briefing(self, topic: str, context: dict[str, object]) -> BriefingUnderstanding:
        ...

    def plan_briefing_queries(self, topic: str, context: dict[str, object]) -> list[str]:
        ...

    def synthesize_briefing(
        self,
        topic: str,
        sources: list[CollectedSource],
        context: dict[str, object],
    ) -> BriefingSynthesis | None:
        ...
