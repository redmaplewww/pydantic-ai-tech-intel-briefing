"""Audit trail for the learning -> regression -> decision loop (Path B).

Every full cycle writes a JSON ledger and a human-readable REPORT.md under
``<audit_dir>/<cycle_id>/``.  Both are traceable back to the triggering
Bad Case id, the skill slug/version, the regression statistics, and the
release decision with its gate/ledger ids.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_cycle_id() -> str:
    return f"learn_{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkillLearningAudit:
    def __init__(self, audit_dir: str | Path = "skills/audits") -> None:
        self.audit_dir = Path(audit_dir)

    def write(
        self,
        cycle: dict[str, Any],
        cycle_id: str | None = None,
    ) -> dict[str, str]:
        """Persist one learning cycle: JSON ledger + REPORT.md.

        ``cycle`` should contain ``bad_case``, ``draft``, ``regression``,
        ``release`` and optional ``notes``.  Returns the written paths.
        """
        cycle_id = cycle_id or str(cycle.get("cycle_id") or new_cycle_id())
        target_dir = self.audit_dir / cycle_id
        target_dir.mkdir(parents=True, exist_ok=True)
        record = dict(cycle)
        record["cycle_id"] = cycle_id
        record.setdefault("created_at", _now_iso())

        json_path = target_dir / "ledger.json"
        json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        report_path = target_dir / "REPORT.md"
        report_path.write_text(_render_report_markdown(record), encoding="utf-8")
        return {"cycle_id": cycle_id, "json_path": str(json_path), "report_path": str(report_path)}


def _render_report_markdown(cycle: dict[str, Any]) -> str:
    bad_case = cycle.get("bad_case") or {}
    draft = cycle.get("draft") or {}
    skill = draft.get("skill") or {}
    regression = cycle.get("regression") or {}
    release = cycle.get("release") or {}
    decision = release.get("decision") or {}

    lines = [
        "# 学习闭环与评测闭环审计报告",
        "",
        f"- cycle_id: {cycle.get('cycle_id', 'unknown')}",
        f"- created_at: {cycle.get('created_at', 'unknown')}",
        "",
        "## 触发的 Bad Case",
        "",
        f"- id: {bad_case.get('id') or bad_case.get('question_id') or 'unknown'}",
        f"- question: {bad_case.get('question') or '(未提供)'}",
        f"- tier: {bad_case.get('tier') or '(未提供)'}",
        f"- quality_flags: {', '.join(str(flag) for flag in (bad_case.get('quality_flags') or [])) or '(无)'}",
        f"- root_causes: {', '.join(str(cause) for cause in (bad_case.get('root_causes') or [])) or '(无)'}",
        f"- attribution_layer: {bad_case.get('attribution_layer') or 'prompt/skill'}",
        "",
        "## 技能草稿（staging）",
        "",
        f"- name: {skill.get('name', 'unknown')}",
        f"- slug: {skill.get('slug', 'unknown')}",
        f"- version: {skill.get('version', 'unknown')}",
        f"- path: {skill.get('path', 'unknown')}",
        "",
        "## 回归（eval-replay 复用 Delta + McNemar）",
        "",
        f"- judge: {regression.get('judge', 'unknown')}",
        f"- replayed: {regression.get('replayed', 0)}",
        f"- improved: {regression.get('improved', 0)} 维 / {regression.get('improved_questions', 0)} 题",
        f"- regressed: {regression.get('regressed', 0)} 维 / {regression.get('regressed_questions', 0)} 题",
        f"- stable: {regression.get('stable', 0)} 维 / {regression.get('stable_questions', 0)} 题",
        f"- McNemar p 值: {regression.get('mcnemar_p_value', 'unknown')}",
        f"- 显著: {'是' if regression.get('significant') else '否'}",
        "",
        "## 发布判定",
        "",
        f"- action: {decision.get('action', 'unknown')}",
        f"- approved: {decision.get('approved')}",
        f"- reason: {decision.get('reason', '')}",
        f"- gate_result: {decision.get('gate_result', 'unknown')}",
        f"- gate_id: {release.get('gate_id', 'unknown')}",
        f"- ledger_id: {release.get('ledger_id', 'unknown')}",
        "",
        "## 追踪",
        "",
        f"- 触发 Bad Case: {bad_case.get('id') or bad_case.get('question_id') or 'unknown'}",
        f"- 技能文件: {skill.get('path', 'unknown')}",
    ]
    notes = cycle.get("notes")
    if notes:
        lines.append("")
        lines.append("## 备注")
        lines.append("")
        lines.append(str(notes))
    return "\n".join(lines) + "\n"
