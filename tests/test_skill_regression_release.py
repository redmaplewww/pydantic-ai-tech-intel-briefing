"""Tests for the regression hook, release gate, and audit trail (Path B).

Covers McNemar threshold boundaries with exact p-values, the staging ->
merge / staging -> reject flows, and the JSON + REPORT.md audit traceability
to the triggering bad case.
"""

from __future__ import annotations

from search_assistant.memory.store import MemoryStore
from search_assistant.skills.audit import SkillLearningAudit
from search_assistant.skills.library import VersionedSkillLibrary
from search_assistant.skills.regression import RegressionResult, SkillRegressionHook
from search_assistant.skills.release import SkillReleaseGate, decide_regression


def _result(
    improved: int = 0,
    regressed: int = 0,
    stable: int = 0,
    skill=None,
    judge: str = "rule",
) -> RegressionResult:
    skill = skill if skill is not None else _dummy_skill()
    return RegressionResult(
        skill=skill,
        judge=judge,
        replayed=improved + regressed + stable,
        improved=improved,
        regressed=regressed,
        stable=stable,
        improved_questions=improved,
        regressed_questions=regressed,
        stable_questions=stable,
        mcnemar_p_value=_mcnemar(improved, regressed),
        significant=_mcnemar(improved, regressed) < 0.05,
    )


def _dummy_skill():
    from search_assistant.skills.library import SkillRecord

    return SkillRecord(
        name="dummy",
        slug="dummy",
        version=1,
        description="",
        related=[],
        path=__import__("pathlib").Path("skills/dummy/SKILL.md"),
        state="staging",
    )


def _mcnemar(improved: int, regressed: int) -> float:
    n = improved + regressed
    if n == 0:
        return 1.0
    import math

    k = min(improved, regressed)
    return min(1.0, 2.0 * sum(math.comb(n, i) * (0.5**n) for i in range(k + 1)))


# -- McNemar threshold boundaries --------------------------------------------


def test_decide_merge_requires_significance_and_no_regression():
    # 5 improved / 0 regressed -> p = 2 * 0.5^5 = 0.0625 >= 0.05 -> reject.
    decision = decide_regression(_result(improved=5, regressed=0))
    assert decision.approved is False
    assert decision.action == "reject"
    assert decision.gate_result == "failed"
    assert abs(decision.p_value - 0.0625) < 1e-9
    assert decision.significant is False

    # 6 improved / 0 regressed -> p = 2 * 0.5^6 = 0.03125 < 0.05 -> merge.
    decision = decide_regression(_result(improved=6, regressed=0))
    assert decision.approved is True
    assert decision.action == "merge"
    assert decision.gate_result == "passed"
    assert abs(decision.p_value - 0.03125) < 1e-9
    assert decision.significant is True
    assert decision.no_regression is True


def test_decide_merge_rejects_significant_with_regression():
    # 10 improved / 1 regressed -> p = 2 * (C(11,0)+C(11,1)) * 0.5^11 = 0.0117 < 0.05,
    # significant, but the regressed question must still block the merge.
    decision = decide_regression(_result(improved=10, regressed=1))
    assert decision.significant is True
    assert decision.no_regression is False
    assert decision.approved is False
    assert decision.action == "reject"


def test_decide_merge_rejects_no_discordant_pairs():
    # No discordant pairs -> p = 1.0 -> reject (no evidence of improvement).
    decision = decide_regression(_result(improved=0, regressed=0, stable=8))
    assert decision.approved is False
    assert decision.p_value == 1.0


def test_decide_merge_treats_dimension_and_question_regression_as_blocking():
    decision = decide_regression(_result(improved=6, regressed=0, stable=0).__class__(
        skill=_dummy_skill(),
        judge="rule",
        replayed=7,
        improved=6,
        regressed=1,
        stable=0,
        improved_questions=6,
        regressed_questions=0,
        stable_questions=1,
        mcnemar_p_value=0.03125,
        significant=True,
    ))
    assert decision.approved is False
    assert decision.reason  # reason mentions regressed dimension


# -- regression hook ----------------------------------------------------------


def _verdicts(verdict: str):
    return {
        "fact_grounding": verdict,
        "citation_fidelity": verdict,
        "commitment_action_consistency": verdict,
        "expression_quality": verdict,
    }


def test_regression_hook_computes_delta_and_mcnemar(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")
    draft = library.create_draft("Recovery Practice", "body")
    previous_items = [
        {"question": f"q{i}", "answer_excerpt": f"old answer {i}", "source_urls": [], "unverified_claim_samples": []}
        for i in range(6)
    ]

    # The fake answer runner produces better answers for all 6 questions, and a
    # verdict runner that turns better answers into "pass" vs previous "fail".
    def answer_runner(question: str):
        return {"answer_excerpt": f"new answer for {question}", "source_urls": [], "unverified_claims": []}

    def verdict_runner(judge, question, answer, source_urls, unverified):
        if answer.startswith("new"):
            return _verdicts("pass")
        return _verdicts("fail")

    hook = SkillRegressionHook(answer_runner=answer_runner, verdict_runner=verdict_runner)
    result = hook.run(draft, previous_items)
    assert result.replayed == 6
    assert result.improved_questions == 6
    assert result.regressed_questions == 0
    assert abs(result.mcnemar_p_value - 0.03125) < 1e-9
    assert result.significant is True


def test_regression_hook_no_runner_is_idempotent(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")
    draft = library.create_draft("Recovery Practice", "body")
    previous_items = [
        {"question": "q1", "answer_excerpt": "same", "source_urls": [], "unverified_claim_samples": []}
    ]
    hook = SkillRegressionHook()
    result = hook.run(draft, previous_items)
    assert result.replayed == 1
    assert result.improved_questions == 0
    assert result.regressed_questions == 0
    assert result.mcnemar_p_value == 1.0


# -- release gate: staging -> merge / reject ----------------------------------


def test_release_gate_merges_on_significant_no_regression(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    library = VersionedSkillLibrary(tmp_path / "skills")
    library.create_draft("Recovery Practice", "draft body")
    gate = SkillReleaseGate(store, library)

    result = gate.release("recovery-practice", _result(improved=6, regressed=0), bad_case_id="bc-1")
    assert result["decision"]["approved"] is True
    assert result["decision"]["action"] == "merge"
    assert result["skill"]["state"] == "active"
    assert library.find_staging("recovery-practice") is None
    assert library.find_active("recovery-practice") is not None

    gates = store.list_gate_records(gate_type="skill_release")
    assert gates and gates[0]["result"] == "passed"
    assert "bc-1" in (gates[0]["metadata"] or {}).get("bad_case_id", "")
    ledger = store.list_project_ledger_entries(entry_type="skill_learning")
    assert ledger and ledger[0]["status"] == "completed"


def test_release_gate_rejects_and_archives_on_insignificant(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    library = VersionedSkillLibrary(tmp_path / "skills")
    library.create_draft("Weak Practice", "draft body")
    gate = SkillReleaseGate(store, library)

    result = gate.release("weak-practice", _result(improved=5, regressed=0), bad_case_id="bc-2")
    assert result["decision"]["approved"] is False
    assert result["decision"]["action"] == "reject"
    assert result["skill"]["state"] == "archive"
    assert library.find_staging("weak-practice") is None
    assert library.find_active("weak-practice") is None
    assert len(library.list_archive()) == 1

    gates = store.list_gate_records(gate_type="skill_release")
    assert gates and gates[0]["result"] == "failed"
    ledger = store.list_project_ledger_entries(entry_type="skill_learning")
    assert ledger and ledger[0]["status"] == "rejected"


def test_release_gate_missing_draft_raises(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    gate = SkillReleaseGate(store, VersionedSkillLibrary(tmp_path / "skills"))
    try:
        gate.release("missing", _result(improved=6, regressed=0))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


# -- audit trail --------------------------------------------------------------


def test_audit_writes_json_and_report_traceable_to_bad_case(tmp_path):
    audit = SkillLearningAudit(tmp_path / "audits")
    paths = audit.write(
        {
            "bad_case": {"id": "bc-audit-1", "question": "q", "quality_flags": ["review_rejected"]},
            "draft": {"skill": _dummy_skill().to_dict()},
            "regression": _result(improved=6, regressed=0).to_dict(),
            "release": {"decision": {"approved": True}, "gate_id": "gate-1", "ledger_id": "ledger-1"},
        },
        cycle_id="learn_test_cycle",
    )
    assert paths["json_path"].endswith("ledger.json")
    assert paths["report_path"].endswith("REPORT.md")
    assert paths["cycle_id"] == "learn_test_cycle"

    import json
    from pathlib import Path

    ledger = json.loads(Path(paths["json_path"]).read_text(encoding="utf-8"))
    assert ledger["cycle_id"] == "learn_test_cycle"
    assert ledger["bad_case"]["id"] == "bc-audit-1"
    assert ledger["regression"]["significant"] is True

    report = Path(paths["report_path"]).read_text(encoding="utf-8")
    assert "学习闭环与评测闭环审计报告" in report
    assert "bc-audit-1" in report
    assert "gate-1" in report
    assert "ledger-1" in report
