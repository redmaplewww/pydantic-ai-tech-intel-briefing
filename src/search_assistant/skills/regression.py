"""Replay hook: regression-check a versioned skill change with eval-replay.

Listens for a skill version change (staging -> active) and re-runs the
evaluation questions through the same Delta + McNemar machinery used by
``eval-replay``.  The judge verdicts, per-dimension ranking, and McNemar exact
test are reused verbatim from ``search_assistant.cli`` (lazy imports keep the
module graph acyclic); only the orchestration is new.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from search_assistant.evolution.verification import RuleQualityJudge
from search_assistant.skills.library import SkillRecord

#: Runner that produces a fresh answer payload for one evaluation question.
#: Production wraps the CLI replay loop; tests inject a fake.
AnswerRunner = Callable[[str], dict[str, Any]]

#: Judge verdicts per dimension for one answer.
VerdictRunner = Callable[[Any, str, str, list[str], list[str]], dict[str, str]]


@dataclass(frozen=True)
class RegressionResult:
    """Outcome of a skill version regression run."""

    skill: SkillRecord
    judge: str
    replayed: int
    improved: int
    regressed: int
    stable: int
    improved_questions: int
    regressed_questions: int
    stable_questions: int
    mcnemar_p_value: float
    significant: bool
    per_item: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill.to_dict(),
            "judge": self.judge,
            "replayed": self.replayed,
            "improved": self.improved,
            "regressed": self.regressed,
            "stable": self.stable,
            "improved_questions": self.improved_questions,
            "regressed_questions": self.regressed_questions,
            "stable_questions": self.stable_questions,
            "mcnemar_p_value": round(self.mcnemar_p_value, 4),
            "significant": self.significant,
            "per_item": self.per_item,
        }


class SkillRegressionHook:
    """Re-run evaluation questions after a skill version change and compare
    old vs new answers with the shared Delta + McNemar logic."""

    def __init__(
        self,
        judge: Any | None = None,
        answer_runner: AnswerRunner | None = None,
        verdict_runner: VerdictRunner | None = None,
        default_unverified: list[str] | None = None,
    ) -> None:
        self.judge = judge if judge is not None else RuleQualityJudge()
        self.answer_runner = answer_runner
        self.verdict_runner = verdict_runner
        self.default_unverified = list(default_unverified or [])

    def run(self, skill: SkillRecord, previous_items: list[dict[str, Any]], max_items: int | None = None) -> RegressionResult:
        """Compare previous evaluation items against fresh answers.

        ``previous_items`` mirrors the per-item records of an
        ``evaluation-report.json`` (each with ``question``, ``answer_excerpt``,
        ``source_urls``, ``unverified_claim_samples``).  For every question a
        fresh answer is produced via ``answer_runner``; if no runner is wired,
        the previous item is re-judged against itself (idempotency check).
        """
        selected = previous_items[:max_items] if max_items is not None else previous_items
        judge_name = str(getattr(self.judge, "name", "rule"))
        verdict_runner = self.verdict_runner or _default_verdict_runner

        per_item: list[dict[str, Any]] = []
        improved = regressed = stable = 0
        improved_questions = regressed_questions = stable_questions = 0
        for item in selected:
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            previous_answer = _answer_body(str(item.get("answer_excerpt") or ""))
            previous_urls = _urls(item.get("source_urls"))
            previous_unverified = _claims(item.get("unverified_claim_samples"))

            current_answer = previous_answer
            current_urls = previous_urls
            current_unverified = previous_unverified
            if self.answer_runner is not None:
                fresh = self.answer_runner(question)
                current_answer = _answer_body(str(fresh.get("answer_excerpt") or fresh.get("final_answer") or ""))
                current_urls = _urls(fresh.get("source_urls"))
                current_unverified = _claims(fresh.get("unverified_claim_samples") or fresh.get("unverified_claims"))

            previous_verdicts = verdict_runner(self.judge, question, previous_answer, previous_urls, previous_unverified)
            current_verdicts = verdict_runner(self.judge, question, current_answer, current_urls, current_unverified)
            dim_improved = [
                dim for dim in current_verdicts
                if _verdict_rank(current_verdicts[dim]) > _verdict_rank(previous_verdicts.get(dim, "uncertain"))
            ]
            dim_regressed = [
                dim for dim in current_verdicts
                if _verdict_rank(current_verdicts[dim]) < _verdict_rank(previous_verdicts.get(dim, "uncertain"))
            ]
            dim_stable = [dim for dim in current_verdicts if dim not in dim_improved and dim not in dim_regressed]
            improved += len(dim_improved)
            regressed += len(dim_regressed)
            stable += len(dim_stable)
            if dim_improved and not dim_regressed:
                improved_questions += 1
            elif dim_regressed and not dim_improved:
                regressed_questions += 1
            else:
                stable_questions += 1
            per_item.append(
                {
                    "question": question,
                    "improved": dim_improved,
                    "regressed": dim_regressed,
                    "stable": dim_stable,
                    "previous_verdicts": previous_verdicts,
                    "current_verdicts": current_verdicts,
                }
            )

        p_value = _mcnemar_p_value(improved_questions, regressed_questions)
        return RegressionResult(
            skill=skill,
            judge=judge_name,
            replayed=len(per_item),
            improved=improved,
            regressed=regressed,
            stable=stable,
            improved_questions=improved_questions,
            regressed_questions=regressed_questions,
            stable_questions=stable_questions,
            mcnemar_p_value=p_value,
            significant=p_value < 0.05,
            per_item=per_item,
        )


def _default_verdict_runner(judge: Any, question: str, answer: str, source_urls: list[str], unverified: list[str]) -> dict[str, str]:
    """Mirror cli._judge_verdicts: run the judge and reduce to per-dimension verdicts."""
    from search_assistant.cli import _judge_verdicts

    return _judge_verdicts(judge, question, answer, source_urls, unverified)


def _verdict_rank(verdict: str) -> int:
    from search_assistant.cli import _verdict_rank

    return _verdict_rank(verdict)


def _mcnemar_p_value(improved: int, regressed: int) -> float:
    from search_assistant.cli import _mcnemar_p_value

    return _mcnemar_p_value(improved, regressed)


def _answer_body(text: str) -> str:
    from search_assistant.cli import _answer_body

    return _answer_body(text)


def _urls(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(url) for url in value if str(url).startswith(("http://", "https://"))]


def _claims(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(claim) for claim in value if str(claim).strip()]
