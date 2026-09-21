"""Release gate: merge or reject a versioned skill based on regression stats.

A skill change is merged only when the McNemar test is significant
(p < 0.05) AND there are no regressed questions.  Otherwise the draft is
rejected and archived.  Every decision writes a gate record and a project
ledger entry so the whole learning->regression->decision chain is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from search_assistant.memory.store import MemoryStore
from search_assistant.skills.library import VersionedSkillLibrary
from search_assistant.skills.regression import RegressionResult

ALPHA = 0.05


@dataclass(frozen=True)
class ReleaseDecision:
    """The decision computed from a regression result."""

    approved: bool
    p_value: float
    significant: bool
    no_regression: bool
    reason: str
    gate_result: str
    action: str  # "merge" | "reject"

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "p_value": round(self.p_value, 4),
            "significant": self.significant,
            "no_regression": self.no_regression,
            "reason": self.reason,
            "gate_result": self.gate_result,
            "action": self.action,
        }


def decide_regression(result: RegressionResult, alpha: float = ALPHA) -> ReleaseDecision:
    """Apply the merge rule: p < alpha AND zero regressed questions."""
    p_value = float(result.mcnemar_p_value)
    significant = p_value < alpha
    no_regression = result.regressed_questions == 0 and result.regressed == 0
    approved = significant and no_regression
    if approved:
        reason = (
            f"McNemar p={p_value:.4f} < {alpha} and no regressed dimensions/questions; "
            f"{result.improved_questions} improved / {result.regressed_questions} regressed / "
            f"{result.stable_questions} stable question(s)."
        )
        gate_result = "passed"
        action = "merge"
    else:
        reasons = []
        if not significant:
            reasons.append(f"p={p_value:.4f} >= {alpha} (not statistically significant)")
        if not no_regression:
            reasons.append(
                f"{result.regressed_questions} regressed question(s), {result.regressed} regressed dimension(s)"
            )
        reason = "Rejected: " + "; ".join(reasons) + "."
        gate_result = "failed"
        action = "reject"
    return ReleaseDecision(
        approved=approved,
        p_value=p_value,
        significant=significant,
        no_regression=no_regression,
        reason=reason,
        gate_result=gate_result,
        action=action,
    )


class SkillReleaseGate:
    """Apply a release decision to the skill library and record the audit trail."""

    def __init__(self, store: MemoryStore, library: VersionedSkillLibrary) -> None:
        self.store = store
        self.library = library

    def release(
        self,
        name_or_slug: str,
        result: RegressionResult,
        bad_case_id: str = "",
        alpha: float = ALPHA,
    ) -> dict[str, Any]:
        """Run the full merge/reject flow and persist gate + ledger records."""
        decision = decide_regression(result, alpha=alpha)
        draft = self.library.find_staging(name_or_slug)
        if draft is None:
            raise FileNotFoundError(f"No staging skill draft found for {name_or_slug}")
        skill: Any
        if decision.approved:
            skill = self.library.promote(draft.slug)
        else:
            skill = self.library.reject(draft.slug, reason=decision.reason)

        evidence_refs = [
            f"skill:{skill.slug}",
            f"bad_case:{bad_case_id}" if bad_case_id else "",
            str(skill.path),
        ]
        evidence_refs = [ref for ref in evidence_refs if ref]

        gate_id = self.store.add_gate_record(
            gate_type="skill_release",
            subject_type="skill",
            subject_id=skill.slug,
            result=decision.gate_result,
            reason=decision.reason,
            evidence_refs=evidence_refs,
            metadata={
                "skill_version": skill.version,
                "p_value": round(decision.p_value, 4),
                "significant": decision.significant,
                "no_regression": decision.no_regression,
                "improved_questions": result.improved_questions,
                "regressed_questions": result.regressed_questions,
                "action": decision.action,
                "bad_case_id": bad_case_id,
            },
        )
        ledger_id = self.store.add_project_ledger_entry(
            entry_type="skill_learning",
            subject=skill.slug,
            status="completed" if decision.approved else "rejected",
            summary=(
                f"Skill {skill.slug} v{skill.version} merged after significant regression "
                f"(p={decision.p_value:.4f})."
                if decision.approved
                else f"Skill draft {skill.slug} rejected: {decision.reason}"
            ),
            evidence_refs=[gate_id] + evidence_refs,
            risk=(
                "Skill changes alter future answer procedures; regression significance "
                "does not replace human review of skill content."
            ),
            rollback="Archive the skill or restore the previous version from git history.",
            metadata={
                "bad_case_id": bad_case_id,
                "p_value": round(decision.p_value, 4),
                "action": decision.action,
                "skill_version": skill.version,
            },
        )
        return {
            "decision": decision.to_dict(),
            "skill": skill.to_dict(),
            "gate_id": gate_id,
            "ledger_id": ledger_id,
        }
