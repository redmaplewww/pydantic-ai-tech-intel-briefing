from __future__ import annotations

from search_assistant.contracts import (
    BriefingSynthesis,
    BriefingUnderstanding,
    CollectedSource,
    default_briefing_understanding,
)


class FakeAgentRuntime:
    def __init__(self, answer_text: str = "This is a local deterministic answer."):
        self.answer_text = answer_text
        self.answer_calls = 0
        self.calibration_calls = 0
        self.search_plan_calls = 0
        self.review_calls = 0
        self.text_task_calls = 0

    def run_text_task(
        self,
        instructions: str,
        payload: dict[str, object],
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> str:
        self.text_task_calls += 1
        return self.answer_text

    def _run_agent_with_max_tokens(
        self,
        instructions: str,
        payload: dict[str, object],
        temperature: float,
        max_tokens: int,
    ) -> str:
        return self.run_text_task(
            instructions,
            payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def plan_search_queries(self, question: str, context: dict[str, object]) -> list[str]:
        self.search_plan_calls += 1
        return []

    def plan_briefing_queries(self, topic: str, context: dict[str, object]) -> list[str]:
        return []

    def understand_briefing(self, topic: str, context: dict[str, object]) -> BriefingUnderstanding:
        return default_briefing_understanding(topic)

    def generate_answer(self, question: str, context: dict[str, object]) -> str:
        self.answer_calls += 1
        return self.answer_text

    def calibrate(self, draft: str, context: dict[str, object]) -> dict[str, object]:
        self.calibration_calls += 1
        unverified = context.get("unverified_claims") or []
        critique = "No blocking issues found."
        if unverified:
            critique = "Draft contains claims that need verification."
        return {
            "ran": True,
            "critique": critique,
            "revision": draft,
        }

    def review_answer(self, answer: str, context: dict[str, object]) -> dict[str, object]:
        self.review_calls += 1
        return {
            "ran": True,
            "approved": True,
            "issues": [],
            "revision": answer,
        }

    def synthesize_briefing(
        self,
        topic: str,
        sources: list[CollectedSource],
        context: dict[str, object],
    ) -> BriefingSynthesis | None:
        return None
