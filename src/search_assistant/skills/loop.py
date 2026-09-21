"""End-to-end learning loop orchestration (Path B).

Binds the learning trigger, the regression hook, the release gate and the
audit writer into a single ``run_cycle`` entry point that mirrors the CLI
workflow: bad case -> staging draft -> eval-replay regression -> release
decision -> JSON + REPORT.md audit trail.
"""

from __future__ import annotations

from typing import Any, Callable

from search_assistant.memory.store import MemoryStore
from search_assistant.skills.audit import SkillLearningAudit, new_cycle_id
from search_assistant.skills.learning import SkillLearningTrigger
from search_assistant.skills.library import VersionedSkillLibrary
from search_assistant.skills.regression import RegressionResult, SkillRegressionHook
from search_assistant.skills.release import SkillReleaseGate

#: Builds a fresh answer payload for one evaluation question.
AnswerRunner = Callable[[str], dict[str, Any]]


class SkillLearningLoop:
    """Orchestrates one full learning->regression->decision->audit cycle."""

    def __init__(
        self,
        store: MemoryStore,
        library: VersionedSkillLibrary | None = None,
        audit: SkillLearningAudit | None = None,
        trigger: SkillLearningTrigger | None = None,
        regression_hook: SkillRegressionHook | None = None,
        release_gate: SkillReleaseGate | None = None,
    ) -> None:
        self.store = store
        self.library = library if library is not None else VersionedSkillLibrary()
        self.audit = audit if audit is not None else SkillLearningAudit()
        self.trigger = trigger  # may be injected; else set via learn()
        self.regression_hook = regression_hook if regression_hook is not None else SkillRegressionHook()
        self.release_gate = release_gate if release_gate is not None else SkillReleaseGate(store, self.library)

    # -- single steps ---------------------------------------------------------

    def learn(self, bad_case: dict[str, Any], fix_note: str = "") -> dict[str, Any]:
        if self.trigger is None:
            raise RuntimeError("SkillLearningTrigger not configured for learn()")
        draft_result = self.trigger.learn(bad_case, fix_note=fix_note)
        return {
            "draft": draft_result.to_dict(),
            "bad_case": bad_case,
        }

    def regress(
        self,
        name_or_slug: str,
        previous_items: list[dict[str, Any]],
        max_items: int | None = None,
    ) -> RegressionResult:
        draft = self.library.find_staging(name_or_slug)
        if draft is None:
            raise FileNotFoundError(f"No staging skill draft found for {name_or_slug}")
        return self.regression_hook.run(draft, previous_items, max_items=max_items)

    def decide(self, name_or_slug: str, result: RegressionResult, bad_case_id: str = "") -> dict[str, Any]:
        return self.release_gate.release(name_or_slug, result, bad_case_id=bad_case_id)

    # -- full cycle -----------------------------------------------------------

    def run_cycle(
        self,
        bad_case: dict[str, Any],
        previous_items: list[dict[str, Any]],
        fix_note: str = "",
        max_items: int | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """One complete learning->regression->decision->audit cycle.

        Returns the cycle record including audit paths; also persists the
        audit JSON + REPORT.md on disk.
        """
        bad_case_id = str(bad_case.get("id") or bad_case.get("question_id") or "unknown-bad-case")
        learned = self.learn(bad_case, fix_note=fix_note)
        draft_slug = learned["draft"]["skill"]["slug"]
        regression = self.regress(draft_slug, previous_items, max_items=max_items)
        release = self.decide(draft_slug, regression, bad_case_id=bad_case_id)

        cycle = {
            "cycle_id": new_cycle_id(),
            "bad_case": bad_case,
            "draft": learned["draft"],
            "regression": regression.to_dict(),
            "release": release,
            "notes": notes,
        }
        paths = self.audit.write(cycle)
        cycle["audit"] = paths
        return cycle
