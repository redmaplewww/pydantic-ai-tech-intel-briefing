"""Learning trigger: turn a fixed Bad Case into a versioned skill draft.

When a Bad Case is attributed to the prompt/skill layer and fixed, the fix is
written as a skill draft using a built-in template.  The draft lands in
staging only -- it never becomes active until the regression gate passes.

The LLM generator is injected as ``llm_runner(instructions, payload) -> str``
so tests can mock it and production can reuse the existing runtime agent call
(see ``cli._build_quality_judge`` for the same runner convention).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from search_assistant.skills.library import (
    SkillRecord,
    VersionedSkillLibrary,
    parse_frontmatter,
    render_skill_document,
    slugify,
)

LLMRunner = Callable[[str, dict[str, Any]], str]

#: Built-in instructions template for distilling a Bad Case fix into a skill.
DRAFT_INSTRUCTIONS = """你是 Agent 评测系统的技能沉淀器。把下面的 Bad Case 修复经验写成一份可复用的技能。

要求：
1. 输出 SKILL.md 完整内容，包含 frontmatter（--- 包裹）与正文。
2. frontmatter 必须含 name（小写连字符 slug）、version（整数）、description、related（相关技能 slug 列表，可空）。
3. 正文按 ## Search Discipline 与 ## Answer Discipline 两节组织，给出可操作的具体规则。
4. 规则必须来自 Bad Case 的实际教训，禁止编造不存在的证据；技能只是程序性指导，不是事实来源。
5. 只输出 SKILL.md 本身，不要输出解释或代码块围栏。

Bad Case 信息:
{context}
"""


@dataclass(frozen=True)
class DraftResult:
    """Outcome of a learning-trigger run."""

    skill: SkillRecord
    generated_text: str
    bad_case_id: str
    fixed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill.to_dict(),
            "bad_case_id": self.bad_case_id,
            "fixed": self.fixed,
            "staging_path": str(self.skill.path),
        }


class SkillLearningTrigger:
    """Distill a Bad Case fix into a staging skill draft via an injected LLM."""

    def __init__(self, library: VersionedSkillLibrary, llm_runner: LLMRunner) -> None:
        self.library = library
        self.llm_runner = llm_runner

    def learn(self, bad_case: dict[str, Any], fix_note: str = "") -> DraftResult:
        """Generate a staging draft from a fixed Bad Case.

        ``bad_case`` should carry at least ``id`` (or ``question_id``) and
        optionally ``question``, ``quality_flags``, ``root_causes``,
        ``answer_excerpt``, and ``tier``.  ``fix_note`` describes the fix that
        was applied (attribution layer prompt/skill).
        """
        bad_case_id = str(bad_case.get("id") or bad_case.get("question_id") or "unknown-bad-case")
        context = _render_bad_case_context(bad_case, fix_note)
        instructions = DRAFT_INSTRUCTIONS.format(context=context)
        payload = {"bad_case_id": bad_case_id, "fix_note": fix_note, "bad_case": bad_case}
        generated = self.llm_runner(instructions, payload)
        skill = self._materialize_draft(bad_case, generated, bad_case_id)
        return DraftResult(skill=skill, generated_text=generated, bad_case_id=bad_case_id, fixed=True)

    def _materialize_draft(self, bad_case: dict[str, Any], generated: str, bad_case_id: str) -> SkillRecord:
        meta, body = parse_frontmatter(generated)
        name = str(meta.get("name") or bad_case.get("skill_name") or _skill_name_from_bad_case(bad_case))
        description = str(meta.get("description") or "Draft distilled from a fixed evaluation Bad Case.")
        related = meta.get("related", [])
        if not isinstance(related, list):
            related = [str(item) for item in str(related).split(",") if str(item).strip()]
        version = 1
        try:
            version = max(1, int(meta.get("version", 1)))
        except (TypeError, ValueError):
            version = 1
        body = body.strip()
        if not body:
            body = _fallback_body(bad_case, bad_case_id)
        record = self.library.create_draft(
            name=name,
            body=body,
            description=description,
            related=[str(item) for item in related],
            version=version,
            source_bad_case=bad_case_id,
        )
        return record


def _render_bad_case_context(bad_case: dict[str, Any], fix_note: str) -> str:
    lines = [
        f"- id: {bad_case.get('id') or bad_case.get('question_id') or 'unknown'}",
        f"- question: {bad_case.get('question') or '(未提供)'}",
        f"- tier: {bad_case.get('tier') or '(未提供)'}",
        f"- quality_flags: {json.dumps(bad_case.get('quality_flags') or [], ensure_ascii=False)}",
        f"- root_causes: {json.dumps(bad_case.get('root_causes') or [], ensure_ascii=False)}",
        f"- attribution_layer: {bad_case.get('attribution_layer') or 'prompt/skill'}",
        f"- fix_note: {fix_note or '(未提供)'}",
    ]
    excerpt = str(bad_case.get("answer_excerpt") or "")
    if excerpt:
        lines.append(f"- answer_excerpt: {excerpt[:600]}")
    return "\n".join(lines)


def _skill_name_from_bad_case(bad_case: dict[str, Any]) -> str:
    question = str(bad_case.get("question") or "skill")
    for flag in ("review_rejected", "blocked_answer"):
        if flag in (bad_case.get("quality_flags") or []):
            return "Review Rejection Recovery Practice"
    return re.sub(r"[^\w\u4e00-\u9fff]+", " ", question).strip()[:40] or "Bad Case Recovery Practice"


def _fallback_body(bad_case: dict[str, Any], bad_case_id: str) -> str:
    return (
        "## Search Discipline\n\n"
        "- Search primary sources first; keep queries tied to concrete entities, dates, and constraints.\n\n"
        "## Answer Discipline\n\n"
        f"- Answer similar future questions with explicit assumptions and uncertainty (distilled from bad case {bad_case_id}).\n"
        "- Do not treat this skill as factual evidence; use it only as reviewed procedure guidance.\n"
    )
