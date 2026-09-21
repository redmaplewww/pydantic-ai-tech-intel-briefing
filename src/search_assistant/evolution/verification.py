"""Three-layer trajectory verification for the self-evolution loop.

This mirrors the three-layer verifier structure in "AI Agents in Depth"
(Chapter 8, Figure 8-2):

- **Result verifier** reads environment truth from the immutable trajectory
  payload (search executed, sources returned, answer present).  Code, not LLM.
- **Process verifier** checks business rules: required search, calibration and
  review discipline, citation provenance, and uncertainty disclosure.  Code,
  not LLM.
- **Quality verifier** judges language and strategy with a Rubric.  The
  default judge is deterministic and offline (``RuleQualityJudge``); an LLM
  rubric judge (``LLMRubricJudge``) is available and should be calibrated
  against expert labels before its output is trusted (see
  :func:`calibrate_quality_judge`).

Every verdict carries evidence (answer excerpt, source URL, review state) and
may abstain with ``uncertain`` when evidence is insufficient instead of
guessing — the same policy the book applies to LLM validators.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Protocol

RULE_JUDGE_NAME = "rule"
LLM_JUDGE_NAME = "llm"

DIMENSIONS: tuple[str, ...] = (
    "fact_grounding",
    "citation_fidelity",
    "commitment_action_consistency",
    "expression_quality",
)
VETO_DIMENSIONS: tuple[str, ...] = ("fact_grounding", "citation_fidelity")

_URL_PATTERN = re.compile(r"https?://[^\s)\]}>\"'，。；;、]+")


class QualityJudge(Protocol):
    """Protocol for the quality layer judge.

    ``judge`` receives the trajectory payload and the collected source dicts
    and returns ``{"judge": name, "dimensions": {dim: {"verdict": "pass" |
    "fail" | "uncertain", "evidence": str, "confidence": "high"|"medium"|"low"}}}``.
    """

    name: str

    def judge(self, payload: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
        ...


def _cited_urls(text: str) -> list[str]:
    """Return the http(s) URLs mentioned in a piece of text, in order."""
    return list(dict.fromkeys(match.group(0).rstrip(".,;") for match in _URL_PATTERN.finditer(text)))


def _has_uncertainty_disclosure(answer_body: str) -> bool:
    """Detect explicit uncertainty/evidence-gap disclosure in an answer."""
    cues = (
        "无法确认", "不能确认", "未确认", "未公开", "没有公开", "未检索到", "没有检索到",
        "未提供", "无法给出", "不能一概而论", "缺少", "缺失", "证据不足", "证据弱",
        "搜索结果相关性不足", "搜索未返回", "低置信", "条件估算", "条件性估算", "假设",
        "取决于", "unknown", "cannot confirm", "not confirmed", "no confirmation",
        "not public", "not disclosed", "missing", "insufficient evidence", "weak evidence",
        "no live search evidence", "conditional estimate", "assumption", "depends on",
        "low-confidence", "did not provide", "did not give", "did not list", "not fetched",
        "not retrieved", "not visible", "did not find", "lack actual deployment evidence",
        "no verification", "没有找到", "并未提供", "未给出", "未列出", "搜索中未提供",
    )
    return any(cue in answer_body for cue in cues)


def _unverified_claims_are_evidence_gaps(unverified_samples: list[Any], answer_body: str) -> bool:
    """Every unverified claim sample must read as a disclosed evidence gap."""
    if not unverified_samples:
        return True
    disqualifying_cues = (
        "can run", "can fit", "could fit", "can deploy", "could deploy", "deployable",
        "definitely", "will run", "full precision", "满血", "可以运行", "能够运行",
        "有可能在内存", "可能装下", "可以装下", "能装下", "权重肯定", "肯定放得下",
        "肯定装得下", "肯定能装下",
    )
    evidence_gap_cues = (
        "未从搜索结果中确认", "无法确认", "不能确认", "未确认", "未公开", "没有公开",
        "未检索到", "没有检索到", "缺少", "缺失", "证据不足", "证据弱", "搜索结果相关性不足",
        "搜索未返回", "not found", "not confirmed", "does not confirm", "unconfirmed",
        "cannot confirm", "unknown", "not presented", "no announced", "no timeline",
        "research demo", "research system", "research-to-early-product", "no benchmark data",
        "no benchmark", "no product or public api", "no public api", "not mentioned",
        "did not find", "did not provide", "lack actual deployment evidence", "no verification",
        "no evidence", "未提供", "并未提供", "未给出", "未列出", "搜索中未提供", "没有找到",
    )
    for raw_sample in unverified_samples:
        sample = str(raw_sample).lower()
        if any(cue in sample for cue in disqualifying_cues):
            return False
        if not any(cue in sample for cue in evidence_gap_cues):
            return False
    return True


class RuleQualityJudge:
    """Deterministic, offline quality judge.

    Verdicts come from the trajectory payload itself: unverified-claim
    disclosure, citation provenance, review/blocked markers, and answer
    substance.  ``uncertain`` means "not enough evidence to judge", which the
    quality layer reports but does not count as a failure.
    """

    name = RULE_JUDGE_NAME

    def judge(self, payload: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
        final_answer = str(payload.get("final_answer", "") or "")
        unverified_claims = payload.get("unverified_claims") or []
        if not isinstance(unverified_claims, list):
            unverified_claims = []
        review = payload.get("review") or {}
        if not isinstance(review, dict):
            review = {}
        source_urls = [
            str(item.get("url", ""))
            for item in sources
            if isinstance(item, dict) and str(item.get("url", "")).startswith(("http://", "https://"))
        ]
        dimensions = {
            "fact_grounding": self._fact_grounding(final_answer, unverified_claims, source_urls),
            "citation_fidelity": self._citation_fidelity(final_answer, source_urls),
            "commitment_action_consistency": self._commitment_action_consistency(final_answer, review),
            "expression_quality": self._expression_quality(final_answer),
        }
        return {"judge": self.name, "dimensions": dimensions}

    @staticmethod
    def _fact_grounding(answer: str, unverified_claims: list[Any], source_urls: list[str]) -> dict[str, str]:
        if not source_urls:
            return {
                "verdict": "uncertain",
                "evidence": "no collected sources are available to ground claims",
                "confidence": "low",
            }
        if unverified_claims:
            if _unverified_claims_are_evidence_gaps(unverified_claims, answer):
                return {
                    "verdict": "pass",
                    "evidence": "unverified claims are disclosed as evidence gaps",
                    "confidence": "medium",
                }
            return {
                "verdict": "fail",
                "evidence": "unverified claims without explicit evidence-gap disclosure",
                "confidence": "high",
            }
        return {
            "verdict": "pass",
            "evidence": "claims are reported as verified against collected sources",
            "confidence": "medium",
        }

    @staticmethod
    def _citation_fidelity(answer: str, source_urls: list[str]) -> dict[str, str]:
        cited = _cited_urls(answer)
        allowed = set(source_urls)
        if not cited:
            return {
                "verdict": "pass",
                "evidence": "answer cites no URLs, so nothing violates provenance",
                "confidence": "low",
            }
        outside = [url for url in cited if url not in allowed]
        if outside:
            return {
                "verdict": "fail",
                "evidence": "cited URLs outside the collected source set: " + "; ".join(outside[:3]),
                "confidence": "high",
            }
        return {
            "verdict": "pass",
            "evidence": "all cited URLs are within the collected source set",
            "confidence": "high",
        }

    @staticmethod
    def _commitment_action_consistency(answer: str, review: dict[str, Any]) -> dict[str, str]:
        if "我已阻断本轮生成" in answer or "最终审查未通过" in answer:
            return {
                "verdict": "fail",
                "evidence": "answer carries a blocked/aborted generation marker",
                "confidence": "high",
            }
        approved = review.get("approved")
        if approved is True:
            return {
                "verdict": "pass",
                "evidence": "final review approved the answer",
                "confidence": "high",
            }
        if approved is False:
            return {
                "verdict": "fail",
                "evidence": "final review did not approve the answer",
                "confidence": "high",
            }
        return {
            "verdict": "uncertain",
            "evidence": "review state is missing from the trajectory",
            "confidence": "low",
        }

    @staticmethod
    def _expression_quality(answer: str) -> dict[str, str]:
        stripped = answer.strip()
        if len(stripped) < 40:
            return {
                "verdict": "fail",
                "evidence": "answer is too terse to carry evidence-based substance",
                "confidence": "high",
            }
        if (
            re.search(r"^#{1,4}\s+\S", stripped, flags=re.MULTILINE)
            or len(re.findall(r"[。！？!?；;\n]", stripped)) >= 2
            or len(re.findall(r"[.!?](?:\s|$)", stripped)) >= 2
        ):
            return {
                "verdict": "pass",
                "evidence": "answer has structure and multiple sentences",
                "confidence": "medium",
            }
        return {
            "verdict": "uncertain",
            "evidence": "answer is single-block prose without structure cues",
            "confidence": "low",
        }


_RUBRIC_INSTRUCTIONS = (
    "You are the quality verifier in a three-layer trajectory evaluation pipeline. "
    "Judge ONLY the supplied final_answer against the supplied collected_sources and review state. "
    "Do not use your own knowledge as evidence; a claim that is not supported by the supplied "
    "sources is a fact_grounding failure. Do not reward length: a concise evidence-grounded answer "
    "is better than a long generic one. For every dimension return verdict pass, fail, or uncertain. "
    "Return uncertain when the supplied material is insufficient to judge (missing sources, missing "
    "review state, ambiguous text) - abstaining is always allowed. Return ONLY one JSON object with "
    "key dimensions mapping each of the four dimensions fact_grounding, citation_fidelity, "
    "commitment_action_consistency, expression_quality to an object with keys verdict, evidence "
    "(a short pointer to where in the answer/sources the decision comes from), and confidence "
    "(high, medium, or low)."
)


class LLMRubricJudge:
    """Rubric quality judge backed by an LLM call.

    ``judge_runner`` is a callable ``(instructions, payload) -> text`` that
    returns the model's raw output.  Any failure (timeout, invalid JSON,
    missing dimensions) degrades to a full ``uncertain`` abstention instead of
    a fabricated verdict.
    """

    name = LLM_JUDGE_NAME

    def __init__(
        self,
        judge_runner: Callable[[str, dict[str, Any]], str] | None = None,
        name: str = LLM_JUDGE_NAME,
    ):
        self.judge_runner = judge_runner
        self.name = name

    def judge(self, payload: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
        if self.judge_runner is None:
            return self._abstain("llm judge runner is not configured")
        prompt_payload = {
            "question": str(payload.get("question", "") or ""),
            "final_answer": str(payload.get("final_answer", "") or "")[:6000],
            "collected_sources": [
                {
                    "title": str(item.get("title", "") or ""),
                    "url": str(item.get("url", "") or ""),
                    "snippet": str(item.get("snippet", "") or "")[:400],
                }
                for item in sources
                if isinstance(item, dict)
            ][:12],
            "unverified_claims": [str(item) for item in (payload.get("unverified_claims") or [])][:5],
            "review_approved": bool((payload.get("review") or {}).get("approved")),
        }
        try:
            content = self.judge_runner(_RUBRIC_INSTRUCTIONS, prompt_payload)
            parsed = _parse_judge_json(content)
        except Exception:
            return self._abstain("judge call failed or returned unparsable output")
        dimensions: dict[str, dict[str, str]] = {}
        for dim in DIMENSIONS:
            raw = (parsed.get("dimensions") or {}).get(dim) or {}
            if not isinstance(raw, dict):
                raw = {}
            verdict = str(raw.get("verdict", "uncertain")).strip().lower()
            if verdict not in {"pass", "fail", "uncertain"}:
                verdict = "uncertain"
            confidence = str(raw.get("confidence", "low")).strip().lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"
            dimensions[dim] = {
                "verdict": verdict,
                "evidence": str(raw.get("evidence", "") or "")[:500],
                "confidence": confidence,
            }
        return {"judge": self.name, "dimensions": dimensions}

    def _abstain(self, reason: str) -> dict[str, Any]:
        return {
            "judge": self.name,
            "abstained_reason": reason,
            "dimensions": {
                dim: {"verdict": "uncertain", "evidence": reason, "confidence": "low"} for dim in DIMENSIONS
            },
        }


def _parse_judge_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("judge output is not a JSON object")
    return parsed


class TrajectoryVerifier:
    """Three-layer verifier over an immutable trajectory log row."""

    def __init__(self, judge: QualityJudge | None = None):
        self.judge = judge if judge is not None else RuleQualityJudge()

    def verify(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        payload = trajectory.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        question = str(payload.get("question", "") or "")
        final_answer = str(payload.get("final_answer", "") or "")
        classification = str(payload.get("classification", "") or "")
        search_record = payload.get("search_record")
        if not isinstance(search_record, dict):
            search_record = {}
        sources = search_record.get("sources") or []
        if not isinstance(sources, list):
            sources = []
        source_urls = [
            str(item.get("url", ""))
            for item in sources
            if isinstance(item, dict) and str(item.get("url", "")).startswith(("http://", "https://"))
        ]
        review = payload.get("review")
        if not isinstance(review, dict):
            review = {}
        calibration = payload.get("calibration")
        if not isinstance(calibration, dict):
            calibration = {}
        unverified_claims = payload.get("unverified_claims") or []
        if not isinstance(unverified_claims, list):
            unverified_claims = []

        result_verification = self._verify_result(final_answer, search_record, review)
        process_verification = self._verify_process(
            final_answer,
            classification,
            search_record,
            calibration,
            review,
            unverified_claims,
            source_urls,
        )
        quality_verification = self._verify_quality(payload, sources, source_urls)
        return {
            "trajectory_id": str(trajectory.get("id", "")),
            "question": question,
            "result_verification": result_verification,
            "process_verification": process_verification,
            "quality_verification": quality_verification,
        }

    @staticmethod
    def _verify_result(
        final_answer: str,
        search_record: dict[str, Any],
        review: dict[str, Any],
    ) -> dict[str, Any]:
        flags: list[str] = []
        answer_present = len(final_answer.strip()) >= 20
        if not answer_present:
            flags.append("empty_answer")
        review_blocked = bool(review.get("ran")) and not bool(review.get("approved", False))
        if review_blocked or "我已阻断本轮生成" in final_answer:
            flags.append("task_blocked")
        return {
            "passed": not flags,
            "flags": flags,
            "answer_present": answer_present,
        }

    @staticmethod
    def _verify_process(
        final_answer: str,
        classification: str,
        search_record: dict[str, Any],
        calibration: dict[str, Any],
        review: dict[str, Any],
        unverified_claims: list[Any],
        source_urls: list[str],
    ) -> dict[str, Any]:
        flags: list[str] = []
        search_required = classification in {"research", "hard", "high_stakes"}
        search_executed = bool(search_record.get("executed"))
        if search_required and not search_executed:
            flags.append("required_search_not_executed")
        if search_executed and not source_urls:
            flags.append("search_returned_no_sources")
        calibration_required = search_required or bool(unverified_claims)
        calibration_ran = bool(calibration.get("ran"))
        if calibration_required and not calibration_ran:
            flags.append("missing_calibration")
        review_ran = bool(review.get("ran"))
        if not review_ran:
            flags.append("review_not_run")
        if bool(review.get("ran")) and not bool(review.get("approved", False)):
            flags.append("blocked_answer")
        if "我已阻断本轮生成" in final_answer or "最终审查未通过" in final_answer:
            flags.append("blocked_answer")
        cited = _cited_urls(final_answer)
        outside = sorted(url for url in cited if url not in set(source_urls))
        if outside:
            flags.append("citation_outside_collected_sources")
        if unverified_claims and not _has_uncertainty_disclosure(final_answer):
            flags.append("unverified_claims_not_disclosed")
        return {
            "passed": not flags,
            "flags": flags,
            "search_required": search_required,
            "search_executed": search_executed,
            "calibration_required": calibration_required,
            "calibration_ran": calibration_ran,
            "review_ran": review_ran,
            "citation_count": len(cited),
            "cited_outside_count": len(outside),
        }

    def _verify_quality(
        self,
        payload: dict[str, Any],
        sources: list[dict[str, Any]],
        source_urls: list[str],
    ) -> dict[str, Any]:
        judged = self.judge.judge(payload, sources)
        dimensions = judged.get("dimensions")
        if not isinstance(dimensions, dict):
            dimensions = {}
        flags: list[str] = []
        for dim in VETO_DIMENSIONS:
            verdict = str((dimensions.get(dim) or {}).get("verdict", ""))
            if verdict == "fail":
                flags.append(f"{dim}_failed")
        for dim in DIMENSIONS:
            if dim in VETO_DIMENSIONS:
                continue
            verdict = str((dimensions.get(dim) or {}).get("verdict", ""))
            if verdict == "fail":
                flags.append(f"{dim}_failed")
        uncertain_dimensions = [
            dim for dim in DIMENSIONS if str((dimensions.get(dim) or {}).get("verdict", "")) == "uncertain"
        ]
        return {
            "passed": not flags,
            "flags": flags,
            "judge": str(judged.get("judge", getattr(self.judge, "name", "unknown"))),
            "dimensions": dimensions,
            "uncertain_dimensions": uncertain_dimensions,
            "source_url_count": len(source_urls),
        }


def calibrate_quality_judge(
    judge: QualityJudge,
    labeled: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]],
) -> dict[str, Any]:
    """Compare judge verdicts against expert labels per dimension.

    ``labeled`` entries are ``(payload, sources, expert)`` where ``expert``
    maps each dimension to ``"pass"``/``"fail"``/``"uncertain"``.  The report
    gives exact agreement plus fail precision/recall per dimension, so a judge
    that never fails anything is exposed by zero recall instead of a clean
    agreement score.
    """
    pairs: dict[str, list[tuple[str, str]]] = {dim: [] for dim in DIMENSIONS}
    for payload, sources, expert in labeled:
        judged = judge.judge(payload, sources)
        dimensions = judged.get("dimensions")
        if not isinstance(dimensions, dict):
            continue
        for dim in DIMENSIONS:
            verdict = str((dimensions.get(dim) or {}).get("verdict", "uncertain"))
            expected = str(expert.get(dim, "uncertain"))
            pairs[dim].append((expected, verdict))
    report: dict[str, Any] = {}
    for dim, dim_pairs in pairs.items():
        total = len(dim_pairs)
        if not total:
            report[dim] = {"total": 0, "exact_agreement": None, "fail_precision": None, "fail_recall": None}
            continue
        exact = sum(1 for expected, verdict in dim_pairs if expected == verdict)
        expected_fails = sum(1 for expected, _ in dim_pairs if expected == "fail")
        detected_fails = sum(1 for _, verdict in dim_pairs if verdict == "fail")
        true_positives = sum(1 for expected, verdict in dim_pairs if expected == "fail" and verdict == "fail")
        report[dim] = {
            "total": total,
            "exact_agreement": round(exact / total, 3),
            "fail_precision": round(true_positives / detected_fails, 3) if detected_fails else None,
            "fail_recall": round(true_positives / expected_fails, 3) if expected_fails else None,
        }
    return report


# A small expert-labeled calibration set for quality judges.  The labels are
# chosen so a well-behaved judge (rule or LLM) should agree on most of them;
# run `evolution-judge-calibrate` to measure agreement before trusting an LLM
# judge's output in the offline loop.
DEFAULT_CALIBRATION_EXAMPLES: tuple[tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]], ...] = (
    (
        {
            "question": "分布式大模型是否可以理解为多个相对独立的节点分别负责一部分推理工作？",
            "classification": "hard",
            "final_answer": (
                "可以，分布式推理把模型拆分到多个节点协同工作，节点之间通过互联通信。"
                "但缺少模型参数证据时只能给出条件性估算。参见 https://example.com/distributed。"
            ),
            "review": {"ran": True, "approved": True, "issues": [], "revision": "same"},
            "unverified_claims": ["缺少官方模型参数，部署规模只能条件性估算"],
        },
        [{"title": "分布式推理", "url": "https://example.com/distributed", "snippet": "节点协同"}],
        {
            "fact_grounding": "pass",
            "citation_fidelity": "pass",
            "commitment_action_consistency": "pass",
            "expression_quality": "pass",
        },
    ),
    (
        {
            "question": "2026 年世界模型进展如何？",
            "classification": "research",
            "final_answer": "我已阻断本轮生成，因为搜索证据不足。",
            "review": {"ran": True, "approved": False, "issues": ["evidence gap"], "revision": "same"},
            "unverified_claims": ["世界模型公开基准缺失，无法横向比较"],
        },
        [{"title": "世界模型", "url": "https://example.com/world-model", "snippet": "research demo"}],
        {
            "fact_grounding": "pass",
            "citation_fidelity": "pass",
            "commitment_action_consistency": "fail",
            "expression_quality": "fail",
        },
    ),
    (
        {
            "question": "什么是 CXL？",
            "classification": "research",
            "final_answer": "CXL 与 AI 服务器、内存池化的关系：CXL 是缓存一致性互连协议。它允许内存池化与设备共享。",
            "review": {"ran": True, "approved": True, "issues": [], "revision": "same"},
            "unverified_claims": [],
        },
        [],
        {
            "fact_grounding": "uncertain",
            "citation_fidelity": "pass",
            "commitment_action_consistency": "pass",
            "expression_quality": "pass",
        },
    ),
)
