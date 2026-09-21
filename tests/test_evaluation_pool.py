"""Tests for the layered evaluation pool, judge-injected eval, and replay deltas."""

from __future__ import annotations

from search_assistant.cli import _mcnemar_p_value, _verdict_rank
from search_assistant.evaluation.service import (
    EVALUATION_QUESTION_POOL,
    EVALUATION_TIERS,
    resolve_evaluation_questions,
)


def test_pool_has_all_tiers_and_notes():
    tiers = {entry["tier"] for entry in EVALUATION_QUESTION_POOL}
    assert tiers == set(EVALUATION_TIERS)
    assert all(entry["question"] and entry["note"] for entry in EVALUATION_QUESTION_POOL)


def test_resolve_full_pool_keeps_order():
    questions, tier_map = resolve_evaluation_questions()
    assert len(questions) == len(EVALUATION_QUESTION_POOL)
    assert tier_map[EVALUATION_QUESTION_POOL[0]["question"]] == "simple"


def test_resolve_mixed_samples_two_per_tier():
    questions, tier_map = resolve_evaluation_questions(tier="mixed")
    assert len(questions) == 2 * len(EVALUATION_TIERS)
    counts = {tier: sum(1 for q in questions if tier_map.get(q) == tier) for tier in EVALUATION_TIERS}
    assert counts == {tier: 2 for tier in EVALUATION_TIERS}


def test_resolve_single_tier_and_max_questions():
    questions, tier_map = resolve_evaluation_questions(tier="hard")
    assert questions and all(tier_map[q] == "hard" for q in questions)
    short, short_map = resolve_evaluation_questions(tier="mixed", max_questions=3)
    assert len(short) == 3
    assert len(short_map) == 3


def test_resolve_explicit_questions_wins():
    questions, tier_map = resolve_evaluation_questions(questions=["只测这一道", "  "])
    assert questions == ["只测这一道"]
    assert tier_map == {}


def test_verdict_rank_orders_pass_over_uncertain_over_fail():
    assert _verdict_rank("pass") > _verdict_rank("uncertain") > _verdict_rank("fail")
    assert _verdict_rank("unknown") == _verdict_rank("uncertain")


def test_mcnemar_p_value_exact_cases():
    # No discordant pairs -> p = 1.0 (not significant).
    assert _mcnemar_p_value(0, 0) == 1.0
    # Strongly one-sided: 5 improved vs 0 regressed -> p = 2 * 0.5^5 = 0.0625.
    assert abs(_mcnemar_p_value(5, 0) - 0.0625) < 1e-9
    # Balanced discordance is never significant.
    assert _mcnemar_p_value(3, 3) == 1.0
    # 6 vs 0 -> p = 2 * 0.5^6 = 0.03125 < 0.05.
    assert _mcnemar_p_value(6, 0) < 0.05
    # Clamped to 1.0 for tiny samples.
    assert _mcnemar_p_value(1, 1) == 1.0


def test_structured_verification_injects_rubric_fail_flags():
    from search_assistant.evaluation.service import _structured_trajectory_verification

    class _FailingJudge:
        name = "fake-failing"

        def judge(self, payload, sources):
            return {
                "judge": self.name,
                "dimensions": {
                    "fact_grounding": {"verdict": "fail", "evidence": "no support", "confidence": "high"},
                    "expression_quality": {"verdict": "pass", "evidence": "ok", "confidence": "medium"},
                },
            }

    item = {
        "question": "测试问题",
        "classification": "research",
        "answer_excerpt": "一个完整的测试答案。",
        "search_record_structured": {"executed": True},
        "source_count": 1,
        "source_urls": ["https://example.com/a"],
        "unverified_claims": 0,
        "unverified_claim_samples": [],
        "calibration_ran": True,
        "review_ran": True,
        "review_approved": True,
        "quality_flags": [],
        "confidence": "high",
        "uncertainty_assessment": "not_applicable",
    }

    verification = _structured_trajectory_verification(item, _FailingJudge())

    quality = verification["quality_verification"]
    assert quality["judge"] == "fake-failing"
    assert quality["rubric"]["dimensions"]["fact_grounding"]["verdict"] == "fail"
    assert "fact_grounding_failed" in quality["flags"]
    assert quality["passed"] is False
