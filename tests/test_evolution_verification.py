"""Tests for the three-layer trajectory verifier and quality judges."""

from __future__ import annotations

from search_assistant.evolution.verification import (
    DIMENSIONS,
    LLMRubricJudge,
    RuleQualityJudge,
    TrajectoryVerifier,
    calibrate_quality_judge,
)


def _payload(
    *,
    final_answer: str = "A technical answer with a concrete mechanism and evidence basis. It cites the collected source directly.",
    question: str = "How does distributed inference work?",
    classification: str = "hard",
    search_executed: bool = True,
    source_urls: list[str] | None = None,
    review_approved: bool = True,
    calibration_ran: bool = True,
    unverified_claims: list[str] | None = None,
    review_ran: bool = True,
) -> dict:
    source_urls = source_urls or ["https://example.com/source-1"]
    return {
        "question": question,
        "classification": classification,
        "search_record": {
            "executed": search_executed,
            "engines": ["bing"],
            "queries": ["distributed inference"],
            "sources": [{"title": "S1", "url": url, "snippet": "x"} for url in source_urls],
        },
        "final_answer": final_answer,
        "calibration": {"ran": calibration_ran, "revision": final_answer},
        "review": {"ran": review_ran, "approved": review_approved, "issues": [], "revision": final_answer},
        "unverified_claims": unverified_claims or [],
        "active_skills": [],
        "answer_strategy": {},
        "runtime_metadata": {},
        "execution_flags": {},
    }


def _trajectory(payload: dict, trajectory_id: str = "traj-1") -> dict:
    return {"id": trajectory_id, "payload": payload, "user_id": "u1", "chat_id": "c1"}


def test_rule_judge_passes_clean_answer():
    judge = RuleQualityJudge()
    result = judge.judge(_payload(), [{"url": "https://example.com/source-1"}])

    assert result["judge"] == "rule"
    assert result["dimensions"]["fact_grounding"]["verdict"] == "pass"
    assert result["dimensions"]["citation_fidelity"]["verdict"] == "pass"
    assert result["dimensions"]["commitment_action_consistency"]["verdict"] == "pass"
    assert result["dimensions"]["expression_quality"]["verdict"] == "pass"


def test_rule_judge_fact_grounding_fails_on_undisclosed_unverified_claims():
    judge = RuleQualityJudge()
    result = judge.judge(
        _payload(unverified_claims=["该模型可以运行在 8 卡集群上"]),
        [{"url": "https://example.com/source-1"}],
    )

    assert result["dimensions"]["fact_grounding"]["verdict"] == "fail"


def test_rule_judge_fact_grounding_abstains_without_sources():
    judge = RuleQualityJudge()
    result = judge.judge(_payload(), [])

    assert result["dimensions"]["fact_grounding"]["verdict"] == "uncertain"
    assert result["dimensions"]["fact_grounding"]["confidence"] == "low"


def test_rule_judge_citation_fidelity_flags_outside_urls():
    judge = RuleQualityJudge()
    answer = "See https://example.com/collected for details and https://external.example/x for more."
    result = judge.judge(
        _payload(final_answer=answer),
        [{"url": "https://example.com/collected"}],
    )

    assert result["dimensions"]["citation_fidelity"]["verdict"] == "fail"
    assert "external.example" in result["dimensions"]["citation_fidelity"]["evidence"]


def test_rule_judge_commitment_fails_on_blocked_marker():
    judge = RuleQualityJudge()
    result = judge.judge(
        _payload(final_answer="我已阻断本轮生成，因为证据不足。", review_approved=False),
        [{"url": "https://example.com/source-1"}],
    )

    assert result["dimensions"]["commitment_action_consistency"]["verdict"] == "fail"


def test_rule_judge_expression_fails_on_terse_answer():
    judge = RuleQualityJudge()
    result = judge.judge(_payload(final_answer="可以。"), [{"url": "https://example.com/source-1"}])

    assert result["dimensions"]["expression_quality"]["verdict"] == "fail"


def test_verifier_result_and_process_layers():
    verifier = TrajectoryVerifier()
    verification = verifier.verify(_trajectory(_payload()))

    assert verification["trajectory_id"] == "traj-1"
    assert verification["result_verification"]["passed"] is True
    assert verification["process_verification"]["passed"] is True
    assert verification["process_verification"]["search_required"] is True
    assert verification["process_verification"]["search_executed"] is True
    assert verification["quality_verification"]["passed"] is True


def test_verifier_flags_missing_search():
    verifier = TrajectoryVerifier()
    verification = verifier.verify(
        _trajectory(_payload(search_executed=False, source_urls=[]))
    )

    assert verification["result_verification"]["passed"] is True
    assert "required_search_not_executed" in verification["process_verification"]["flags"]
    assert verification["process_verification"]["passed"] is False


def test_verifier_flags_citation_outside_sources():
    verifier = TrajectoryVerifier()
    answer = "Evidence at https://example.com/source-1 and https://elsewhere.example/x."
    verification = verifier.verify(_trajectory(_payload(final_answer=answer)))

    assert "citation_outside_collected_sources" in verification["process_verification"]["flags"]
    assert "citation_fidelity_failed" in verification["quality_verification"]["flags"]


def test_verifier_task_blocked_when_review_rejects():
    verifier = TrajectoryVerifier()
    verification = verifier.verify(
        _trajectory(_payload(review_approved=False, final_answer="最终审查未通过：证据不足。"))
    )

    assert "task_blocked" in verification["result_verification"]["flags"]
    assert "blocked_answer" in verification["process_verification"]["flags"]
    assert verification["quality_verification"]["passed"] is False


def test_llm_judge_parses_structured_output():
    def runner(instructions: str, payload: dict) -> str:
        assert "fact_grounding" in instructions
        assert payload["final_answer"]
        return (
            '{"dimensions": {"fact_grounding": {"verdict": "pass", "evidence": "supported by source-1", '
            '"confidence": "high"}, "citation_fidelity": {"verdict": "pass", "evidence": "urls in sources", '
            '"confidence": "high"}, "commitment_action_consistency": {"verdict": "uncertain", "evidence": "review missing", '
            '"confidence": "low"}, "expression_quality": {"verdict": "pass", "evidence": "structured", '
            '"confidence": "medium"}}}'
        )

    judge = LLMRubricJudge(judge_runner=runner)
    result = judge.judge(_payload(), [{"url": "https://example.com/source-1"}])

    assert result["judge"] == "llm"
    assert result["dimensions"]["fact_grounding"]["verdict"] == "pass"
    assert result["dimensions"]["commitment_action_consistency"]["verdict"] == "uncertain"


def test_llm_judge_abstains_on_garbage_output():
    def runner(instructions: str, payload: dict) -> str:
        return "not json at all"

    judge = LLMRubricJudge(judge_runner=runner)
    result = judge.judge(_payload(), [{"url": "https://example.com/source-1"}])

    assert all(
        result["dimensions"][dim]["verdict"] == "uncertain" for dim in DIMENSIONS
    )
    assert "abstained_reason" in result


def test_llm_judge_abstains_without_runner():
    judge = LLMRubricJudge(judge_runner=None)
    result = judge.judge(_payload(), [])

    assert all(result["dimensions"][dim]["verdict"] == "uncertain" for dim in DIMENSIONS)


def test_calibrate_reports_agreement_and_fail_recall():
    judge = RuleQualityJudge()

    def good(label: dict) -> tuple[dict, list[dict], dict[str, str]]:
        return _payload(), [{"url": "https://example.com/source-1"}], label

    def bad(label: dict) -> tuple[dict, list[dict], dict[str, str]]:
        payload = _payload(
            final_answer=(
                "我已阻断本轮生成，因为证据不足。无法确认该模型可以运行在目标集群上，"
                "缺少模型参数与部署指标。请提供官方模型卡或基准评测后再判断。"
            ),
            review_approved=False,
            unverified_claims=["该模型可以运行"],
        )
        return payload, [{"url": "https://example.com/source-1"}], label

    labeled = [
        good({"fact_grounding": "pass", "citation_fidelity": "pass", "commitment_action_consistency": "pass", "expression_quality": "pass"}),
        bad({"fact_grounding": "fail", "citation_fidelity": "pass", "commitment_action_consistency": "fail", "expression_quality": "pass"}),
    ]
    report = calibrate_quality_judge(judge, labeled)

    assert report["fact_grounding"]["total"] == 2
    assert report["fact_grounding"]["fail_recall"] == 1.0
    assert report["commitment_action_consistency"]["fail_precision"] == 1.0
    assert report["expression_quality"]["exact_agreement"] == 1.0


def test_calibrate_exposes_judge_that_never_fails():
    class AlwaysPassJudge:
        name = "always-pass"

        def judge(self, payload: dict, sources: list[dict]) -> dict:
            return {
                "judge": self.name,
                "dimensions": {
                    dim: {"verdict": "pass", "evidence": "none", "confidence": "low"} for dim in DIMENSIONS
                },
            }

    labeled = [
        (
            _payload(final_answer="我已阻断本轮生成，因为证据不足。", review_approved=False),
            [{"url": "https://example.com/source-1"}],
            {"fact_grounding": "pass", "citation_fidelity": "pass", "commitment_action_consistency": "fail", "expression_quality": "pass"},
        )
    ]
    report = calibrate_quality_judge(AlwaysPassJudge(), labeled)

    assert report["commitment_action_consistency"]["fail_recall"] == 0.0
