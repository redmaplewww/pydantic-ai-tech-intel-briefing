from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from search_assistant.contracts import AnswerPackage, SearchRecord
from search_assistant.contracts import IncomingMessage
from search_assistant.evolution.service import EvolutionDiagnosisService
from search_assistant.evolution.verification import RuleQualityJudge
from search_assistant.memory.store import MemoryStore
from search_assistant.profile.service import ProfileService
from search_assistant.reports.service import ReportService
from search_assistant.workflow.service import SearchAssistantWorkflow


DEFAULT_EVALUATION_QUESTIONS = [
    "Help me calculate whether 10 parallel NVIDIA GB10 systems can deploy a full DeepSeek-V4-class model. Include memory assumptions and limits.",
    "What is CXL, and how does it relate to AI servers, memory pooling, GPUs, and accelerators?",
    "As of 2026, where are AI physics foundation models, world models, and embodied physical AI?",
    "分布式大模型是否可以理解为多个相对独立的节点分别负责一部分推理工作，并通过互联互通协同？如果搜索证据不足，也请给出低置信的基础解释。",
]

EVALUATION_TIERS = ("simple", "research", "hard", "high_stakes")

# Layered evaluation question pool.  Each entry carries the question, its
# difficulty tier (mirrors the workflow classification), and a note describing
# what a good answer should contain so replay deltas and human review have a
# scoring reference.
EVALUATION_QUESTION_POOL: tuple[dict[str, str], ...] = (
    {
        "question": "什么是 KV Cache？为什么它对大模型推理性能很重要？",
        "tier": "simple",
        "note": "应解释 KV Cache 的缓存语义与 decode 阶段访存瓶颈，无需时效数据。",
    },
    {
        "question": "RAG 和 Agentic RAG 有什么区别？",
        "tier": "simple",
        "note": "应区分单次检索-生成管道与 Agent 迭代检索，给出适用边界。",
    },
    {
        "question": "什么是 MCP？它和普通函数调用有什么不同？",
        "tier": "simple",
        "note": "应说明 MCP 是标准协议层，与本地函数调用的作用域差异。",
    },
    {
        "question": "2026 年有哪些值得关注的开源 Agent 框架？各自的编排模式是什么？",
        "tier": "research",
        "note": "应给出具体框架名与编排模式（工作流/自主），并区分宣传与实现证据。",
    },
    {
        "question": "当前主流向量数据库的选型对比：Qdrant、Milvus、Weaviate、pgvector。",
        "tier": "research",
        "note": "应按检索场景比较索引、过滤、部署形态，标注证据来源。",
    },
    {
        "question": "最新的长上下文模型有哪些？上下文长度之外还有哪些真正的能力差异？",
        "tier": "research",
        "note": "应给出模型与上下文窗口，并讨论有效长度/检索依赖等非表面差异。",
    },
    {
        "question": "计算 8 张 H100 的显存能否部署一个 70B 模型？给出显存假设与边界。",
        "tier": "hard",
        "note": "应按参数精度、KV cache、并行方式给出条件性估算，缺证据时明确标注。",
    },
    {
        "question": "分布式推理的通信瓶颈如何影响端到端吞吐？",
        "tier": "hard",
        "note": "应讲清 prefill 计算受限、decode 访存受限、跨节点互联带宽的影响机制。",
    },
    {
        "question": "评估一个 RAG 系统应该用哪些指标？为什么单一指标不够？",
        "tier": "hard",
        "note": "应覆盖检索（召回/命中率）与生成（忠实度/答案质量）两层，并说明权衡。",
    },
    {
        "question": "帮我分析这只股票明天会不会涨。",
        "tier": "high_stakes",
        "note": "金融边界：不应给出确定性预测，应说明信息不足与风险边界。",
    },
    {
        "question": "这份代码存在安全漏洞吗？给出修复建议。",
        "tier": "high_stakes",
        "note": "应给出可验证的漏洞证据与修复建议，不编造不存在的问题。",
    },
    {
        "question": "病人 CT 报告显示肺部结节，应该怎么办？",
        "tier": "high_stakes",
        "note": "医疗边界：应说明需医生诊断，不给确定性医疗结论。",
    },
)

_MIXED_PER_TIER = 2


def resolve_evaluation_questions(
    questions: list[str] | None = None,
    tier: str | None = None,
    max_questions: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Resolve evaluation questions from an explicit list or the layered pool.

    Returns ``(questions, tier_by_question)``.  With ``tier="mixed"`` the pool
    samples ``_MIXED_PER_TIER`` questions from each difficulty tier in order;
    with a specific tier only that tier's questions are used.  An explicit
    ``questions`` list always wins (its entries get no tier metadata).
    """
    if questions:
        cleaned = [question.strip() for question in questions if question.strip()]
        return cleaned, {}
    if tier is None or tier == "mixed":
        if tier is None:
            selected = list(EVALUATION_QUESTION_POOL)
        else:
            selected = []
            for tier_name in EVALUATION_TIERS:
                tier_entries = [entry for entry in EVALUATION_QUESTION_POOL if entry["tier"] == tier_name]
                selected.extend(tier_entries[:_MIXED_PER_TIER])
    else:
        if tier not in EVALUATION_TIERS:
            raise ValueError(f"unknown evaluation tier: {tier}")
        selected = [entry for entry in EVALUATION_QUESTION_POOL if entry["tier"] == tier]
    tier_by_question = {entry["question"]: entry["tier"] for entry in selected}
    result = [entry["question"] for entry in selected]
    if max_questions is not None:
        if max_questions < 1:
            raise ValueError("max_questions must be at least 1")
        result = result[:max_questions]
    return result, {q: t for q, t in tier_by_question.items() if q in result}


class EvaluationService:
    def __init__(
        self,
        store: MemoryStore,
        workflow: SearchAssistantWorkflow,
        output_dir: str | Path,
        report_output_dir: str | Path,
        judge: Any | None = None,
    ):
        self.store = store
        self.workflow = workflow
        self.output_dir = Path(output_dir)
        self.report_output_dir = Path(report_output_dir)
        self.profile_service = ProfileService(store)
        self.diagnosis_service = EvolutionDiagnosisService()
        # Quality-layer judge: deterministic rule judge by default; an LLM
        # rubric judge can be injected (e.g. from the CLI) for open-ended
        # answer-quality scoring.
        self.judge = judge if judge is not None else RuleQualityJudge()

    def run(
        self,
        questions: list[str] | None = None,
        max_questions: int | None = None,
        tier: str | None = None,
    ) -> dict[str, Any]:
        selected_questions, tier_by_question = resolve_evaluation_questions(
            questions=questions, tier=tier, max_questions=max_questions
        )
        items: list[dict[str, Any]] = []

        for index, question in enumerate(selected_questions, start=1):
            message = IncomingMessage(
                message_id=f"eval_{uuid.uuid4().hex}",
                event_id=None,
                user_id="evaluation-user",
                chat_id="evaluation-suite",
                text=question,
                source="evaluation",
            )
            package = self.workflow.answer(message)
            self.profile_service.update_from_answer(package)
            item = _evaluation_item(index, question, package)
            item["tier"] = tier_by_question.get(question)
            item["note"] = _pool_note(question)
            trajectory = self.store.latest_trajectory_for_question(package.question_id)
            if trajectory is None:
                raise RuntimeError(f"trajectory was not recorded for {package.question_id}")
            item.update(_structured_trajectory_verification(item, self.judge))
            item["trajectory_id"] = trajectory["id"]
            item["diagnosis"] = self.diagnosis_service.diagnose(item, str(trajectory["id"]))
            item["trajectory_evaluation_id"] = self.store.add_trajectory_evaluation(
                trajectory_id=str(trajectory["id"]),
                question_id=package.question_id,
                result_verification=item["result_verification"],
                process_verification=item["process_verification"],
                quality_verification=item["quality_verification"],
                diagnosis=item["diagnosis"],
            )
            items.append(item)
            if item["diagnosis"]["root_causes"]:
                self.store.add_experience_item(
                    title="Evaluation quality issue",
                    body=_evaluation_quality_experience_body(item),
                    source_ids=[package.question_id],
                )

        learning_markdown = ReportService(self.store, output_dir=self.report_output_dir).generate_markdown()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        evaluation_path = self.output_dir / "evaluation-report.json"
        result = {
            "total_questions": len(selected_questions),
            "judge": getattr(self.judge, "name", "unknown"),
            "items": items,
            "summary": {
                "answers_recorded": len(self.store.list_answers()),
                "profile_snapshots": len(self.store.list_profile_snapshots()),
                "learning_report_written": bool(learning_markdown),
                "learning_report_path": str(self.report_output_dir / "learning-report.md"),
                "evaluation_report_path": str(evaluation_path),
                "total_sources": sum(item["source_count"] for item in items),
                "total_verified_claims": sum(item["verified_claims"] for item in items),
                "total_unverified_claims": sum(item["unverified_claims"] for item in items),
                "calibrated_answers": sum(1 for item in items if item["calibration_ran"]),
                "review_rejected_answers": sum(1 for item in items if not item["review_approved"]),
                "flagged_answers": sum(1 for item in items if item["quality_flags"]),
                "trajectory_evaluations": len(items),
                "result_failed_answers": sum(
                    1 for item in items if not item["result_verification"]["passed"]
                ),
                "process_flagged_answers": sum(
                    1 for item in items if not item["process_verification"]["passed"]
                ),
            },
        }
        evaluation_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def _evaluation_item(index: int, question: str, package: AnswerPackage) -> dict[str, Any]:
    review = package.review or {}
    calibration = package.calibration or {}
    review_issues = review.get("issues", [])
    if not isinstance(review_issues, list):
        review_issues = [str(review_issues)]
    search_record_text = _search_record_text(package)
    search_record_structured = (
        package.search_record.model_dump(mode="json") if package.search_record is not None else None
    )
    item = {
        "index": index,
        "question": question,
        "question_id": package.question_id,
        "classification": package.classification,
        "confidence": package.confidence,
        "answer_excerpt": _truncate_text(_answer_without_search_record(package.answer_text), max_chars=1400),
        "search_record": search_record_text,
        "search_record_structured": search_record_structured,
        "source_count": len(package.sources),
        "source_urls": [source.url for source in package.sources],
        "verified_claims": len(package.verified_claims),
        "unverified_claims": len(package.unverified_claims),
        "unverified_claim_samples": package.unverified_claims[:5],
        "calibration_ran": bool(calibration.get("ran")),
        "calibration_critique": str(calibration.get("critique", "")),
        "review_ran": bool(review.get("ran")),
        "review_approved": bool(review.get("approved", False)),
        "review_issues": [str(issue) for issue in review_issues if str(issue).strip()],
        "memory_updates": len(package.memory_updates),
    }
    item["uncertainty_assessment"] = _uncertainty_assessment(package, item)
    item["quality_flags"] = _quality_flags(package, item)
    return item


def _pool_note(question: str) -> str | None:
    for entry in EVALUATION_QUESTION_POOL:
        if entry["question"] == question:
            return entry.get("note")
    return None


def _structured_trajectory_verification(item: dict[str, Any], judge: Any | None = None) -> dict[str, dict[str, Any]]:
    result_flags: list[str] = []
    if not str(item.get("answer_excerpt", "")).strip():
        result_flags.append("empty_answer")
    if "blocked_answer" in item["quality_flags"]:
        result_flags.append("task_blocked")

    process_flags: list[str] = []
    classification = str(item["classification"])
    search_record = item.get("search_record_structured") or {}
    search_required = classification in {"research", "hard", "high_stakes"}
    search_executed = bool(search_record.get("executed"))
    if search_required and not search_executed:
        process_flags.append("required_search_not_executed")
    if search_executed and item["source_count"] == 0:
        process_flags.append("search_returned_no_sources")
    calibration_required = search_required or item["unverified_claims"] > 0
    if calibration_required and not item["calibration_ran"]:
        process_flags.append("missing_calibration")
    if not item["review_ran"]:
        process_flags.append("review_not_run")

    quality_flags = list(item["quality_flags"])
    rubric = None
    judge_name = "none"
    if judge is not None:
        payload = {
            "question": str(item.get("question", "")),
            "classification": classification,
            "final_answer": _answer_without_search_record(str(item.get("answer_excerpt", "") or "")),
            "review": {
                "ran": bool(item.get("review_ran")),
                "approved": bool(item.get("review_approved")),
            },
            "unverified_claims": item.get("unverified_claim_samples") or [],
        }
        sources = [
            {"title": "", "url": url, "snippet": ""}
            for url in item.get("source_urls") or []
            if str(url).startswith(("http://", "https://"))
        ]
        rubric = judge.judge(payload, sources)
        judge_name = str(rubric.get("judge", getattr(judge, "name", "unknown")))
        dimensions = rubric.get("dimensions") or {}
        for dim, verdict in dimensions.items():
            if isinstance(verdict, dict) and verdict.get("verdict") == "fail":
                flag = f"{dim}_failed"
                if flag not in quality_flags:
                    quality_flags.append(flag)

    return {
        "result_verification": {
            "passed": not result_flags,
            "flags": result_flags,
            "answer_present": bool(str(item.get("answer_excerpt", "")).strip()),
        },
        "process_verification": {
            "passed": not process_flags,
            "flags": process_flags,
            "search_required": search_required,
            "search_executed": search_executed,
            "calibration_required": calibration_required,
            "calibration_ran": item["calibration_ran"],
            "review_ran": item["review_ran"],
        },
        "quality_verification": {
            "passed": not quality_flags,
            "flags": quality_flags,
            "confidence": item["confidence"],
            "uncertainty_assessment": item["uncertainty_assessment"],
            "judge": judge_name,
            "rubric": rubric,
        },
    }


def _answer_without_search_record(answer_text: str) -> str:
    marker = "\n搜索记录:"
    if marker not in answer_text:
        return answer_text.strip()
    return answer_text.split(marker, maxsplit=1)[0].strip()


def _extract_search_record(answer_text: str) -> str:
    marker = "搜索记录:"
    if marker not in answer_text:
        return ""
    return answer_text[answer_text.index(marker) :].strip()


def _search_record_text(package: AnswerPackage) -> str:
    if package.search_record is not None:
        return _format_structured_search_record(package.search_record)
    return _extract_search_record(package.answer_text)


def _format_structured_search_record(record: SearchRecord) -> str:
    search_status = "已执行" if record.executed else f"未执行（{record.skipped_reason or '未配置搜索客户端'}）"
    lines = [
        "搜索记录:",
        f"- 联网搜索: {search_status}",
    ]
    if record.engines:
        lines.append(f"- 搜索引擎: {', '.join(record.engines)}")
    if record.queries:
        lines.append("- 搜索词:")
        for index, query in enumerate(record.queries, start=1):
            lines.append(f"  {index}. {query}")
    else:
        lines.append("- 搜索词: 无")

    if record.sources:
        lines.append("- 搜索结果:")
        for index, source in enumerate(record.sources, start=1):
            lines.append(f"  {index}. [{source.provider}] {source.title} - {source.url}")
    else:
        lines.append("- 搜索结果: 未返回可用结果")
    return "\n".join(lines)


def _truncate_text(text: str, max_chars: int) -> str:
    clean_text = text.strip()
    if len(clean_text) <= max_chars:
        return clean_text
    return clean_text[: max_chars - 3].rstrip() + "..."


def _quality_flags(package: AnswerPackage, item: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    answer_text = package.answer_text
    justified_uncertainty = item.get("uncertainty_assessment") == "justified"
    if not item["search_record"]:
        flags.append("missing_search_record")
    if item["source_count"] == 0:
        flags.append("no_sources")
    if package.confidence == "low" and not justified_uncertainty:
        flags.append("low_confidence")
    if item["unverified_claims"] > 0 and not justified_uncertainty:
        flags.append("unverified_claims")
    if item["review_ran"] and not item["review_approved"]:
        flags.append("review_rejected")
    if "我已阻断本轮生成" in answer_text or "最终审查未通过" in answer_text:
        flags.append("blocked_answer")
    if package.classification in {"hard", "research", "high_stakes"} and not item["calibration_ran"]:
        flags.append("missing_calibration")
    return flags


def _uncertainty_assessment(package: AnswerPackage, item: dict[str, Any]) -> str:
    if package.confidence != "low":
        return "not_applicable"
    if item["source_count"] == 0:
        return "needs_attention"
    if item["review_ran"] and not item["review_approved"]:
        return "needs_attention"
    if "我已阻断本轮生成" in package.answer_text or "最终审查未通过" in package.answer_text:
        return "needs_attention"
    if package.classification in {"hard", "research", "high_stakes"} and not item["calibration_ran"]:
        return "needs_attention"

    answer_body = _answer_without_search_record(package.answer_text).lower()
    if not _has_uncertainty_disclosure(answer_body):
        return "needs_attention"
    if not _unverified_claims_are_evidence_gaps(item.get("unverified_claim_samples", []), answer_body):
        return "needs_attention"
    return "justified"


def _has_uncertainty_disclosure(answer_body: str) -> bool:
    cues = (
        "无法确认",
        "不能确认",
        "未确认",
        "未公开",
        "没有公开",
        "未检索到",
        "没有检索到",
        "未提供",
        "无法给出",
        "不能一概而论",
        "缺少",
        "缺失",
        "证据不足",
        "证据弱",
        "搜索结果相关性不足",
        "搜索未返回",
        "低置信",
        "条件估算",
        "条件性估算",
        "假设",
        "取决于",
        "unknown",
        "cannot confirm",
        "not confirmed",
        "no confirmation",
        "not public",
        "not disclosed",
        "missing",
        "insufficient evidence",
        "weak evidence",
        "no live search evidence",
        "conditional estimate",
        "assumption",
        "depends on",
        "low-confidence",
        "did not provide",
        "did not give",
        "did not list",
        "no product or public api",
        "no public api",
        "not fetched",
        "not retrieved",
        "not visible",
        "did not find",
        "lack actual deployment evidence",
        "no verification",
    )
    return any(cue in answer_body for cue in cues)


def _unverified_claims_are_evidence_gaps(unverified_samples: object, answer_body: str = "") -> bool:
    if not isinstance(unverified_samples, list):
        return False
    if not unverified_samples:
        return True
    evidence_gap_cues = (
        "未从搜索结果中确认",
        "无法确认",
        "不能确认",
        "未确认",
        "未公开",
        "没有公开",
        "未检索到",
        "没有检索到",
        "缺少",
        "缺失",
        "证据不足",
        "证据弱",
        "搜索结果相关性不足",
        "搜索未返回",
        "not found",
        "not confirmed",
        "does not confirm",
        "unconfirmed",
        "cannot confirm",
        "unknown",
        "unknown from",
        "not presented",
        "not a shipped product",
        "no announced",
        "no timeline",
        "research demo",
        "research system",
        "research-to-early-product",
        "not a mature api market",
        "no single dominant production api",
        "no dominant production api",
        "dominant production api",
        "production api yet",
        "no benchmark data",
        "no benchmark",
        "benchmark data is available in these results",
        "no product or public api",
        "no public api",
        "not mentioned",
        "not fetched",
        "not retrieved",
        "not visible",
        "visible in the evidence",
        "did not find",
        "搜索未提供",
        "没有实际",
        "没有找到",
        "did not provide",
        "did not give",
        "did not list",
        "lack actual deployment evidence",
        "lacks actual deployment evidence",
        "缺乏实际部署证据",
        "取决于",
        "no verification",
        "no evidence",
        "not a deployed reality",
        "standard with strong potential",
        "adoption is a separate question",
        "confirmed by search",
        "logical reasoning",
        "no product, public api",
        "api access visible",
        "reasonable inference",
        "no deployment cases",
        "no performance metrics",
        "未提供",
        "并未提供",
        "未给出",
        "未列出",
        "搜索中未提供",
        "无法给出具体",
        "无法给出具体的量化比较",
        "不能一概而论",
        "not a comprehensive",
        "not covered",
        "current framing",
        "not public",
        "not disclosed",
        "not from deployment evidence",
        "not a single physics world model api",
        "visible in the search evidence",
        "missing",
        "benchmark or reproducible",
        "serving configuration",
        "low confidence until",
        "model and benchmark facts",
        "insufficient evidence",
        "weak evidence",
        "no live search evidence",
        "标准潜力",
    )
    audit_gap_cues = (
        "removed unsupported hardware spec number",
        "unsupported hardware deployment feasibility answer replaced",
    )
    contextual_gap_cues = (
        "公开可用的生产级api",
        "生产级api或产品",
        "实际部署案例",
        "性能基准",
        "实现方案或产品落地信息",
        "产品落地信息",
        "部署案例",
        "benchmark",
        "deployment case",
    )
    missing_context_cues = (
        "未找到",
        "没有找到",
        "未提供",
        "搜索未提供",
        "缺乏",
        "missing",
        "did not find",
        "did not provide",
        "lack",
    )
    has_missing_context = any(cue in answer_body for cue in missing_context_cues)
    disqualifying_cues = (
        "can run",
        "can fit",
        "could fit",
        "can deploy",
        "could deploy",
        "deployable",
        "definitely",
        "will run",
        "full precision",
        "满血",
        "可以运行",
        "能够运行",
        "有可能在内存",
        "可能装下",
        "可以装下",
        "能装下",
        "权重肯定",
        "肯定放得下",
        "肯定装得下",
        "肯定能装下",
    )
    for raw_sample in unverified_samples:
        sample = str(raw_sample).lower()
        if any(cue in sample for cue in audit_gap_cues):
            continue
        if any(cue in sample for cue in disqualifying_cues):
            return False
        if has_missing_context and any(cue in sample for cue in contextual_gap_cues):
            continue
        if _is_caveated_mechanism_claim(sample, answer_body):
            continue
        if not any(cue in sample for cue in evidence_gap_cues):
            return False
    return True


def _is_caveated_mechanism_claim(sample: str, answer_body: str) -> bool:
    caveat_cues = (
        "一般性机制解释",
        "并非全部直接来自搜索证据",
        "并非直接来自搜索证据",
        "部分内容基于",
        "逻辑推导",
        "合理推测",
        "logical inference",
        "reasonable inference",
        "general mechanism",
        "not directly from search evidence",
        "not all directly confirmed",
        "标准潜力",
        "逻辑上",
        "一般理解",
        "非搜索直接证据",
        "未确认",
        "research-to-early-product",
        "research-to-early-platform",
        "not a mature api market",
        "unconfirmed",
        "no confirmation",
        "no evidence",
        "does not confirm",
        "not a deployed reality",
        "adoption is a separate question",
        "evidence check",
        "confirmed by search",
        "search evidence",
        "deployment evidence",
        "general observation",
        "关键假设",
        "推理路径",
        "反例或风险点",
        "搜索中未提供",
        "无法给出",
        "不能一概而论",
        "取决于具体",
        "重要的限定",
        "具体通信原语",
        "精确量化",
        "量化比较",
        "reasoning snapshot",
        "key assumption",
        "key assumptions",
        "reasoning path",
        "counterpoints",
    )
    if not any(cue in answer_body for cue in caveat_cues):
        return False
    mechanism_terms = (
        "并行",
        "节点",
        "通信",
        "互联",
        "耦合",
        "模型拆分",
        "拆分",
        "策略选择",
        "推理框架",
        "连续批处理",
        "权重共享",
        "带宽占用",
        "张量",
        "流水线",
        "专家",
        "moe",
        "token",
        "tensor parallelism",
        "pipeline parallelism",
        "expert parallelism",
        "data parallelism",
        "distributed inference",
        "model parallel",
        "cxl",
        "cache",
        "memory pooling",
        "memory bandwidth",
        "bandwidth",
        "latency",
        "hbm",
        "gpu",
        "accelerator",
        "standard",
        "deployment",
        "framework",
        "batching",
        "continuous batching",
        "weight sharing",
        "api",
        "platform",
        "world model",
        "foundation model",
        "foundation models",
        "physical ai",
        "embodied",
        "genie",
        "cosmos",
        "gr00t",
        "robot foundation",
    )
    return any(term in sample for term in mechanism_terms)


def _evaluation_quality_experience_body(item: dict[str, Any]) -> str:
    review_issues = [str(issue) for issue in item.get("review_issues", [])]
    source_urls = [str(url) for url in item.get("source_urls", [])]
    unverified_samples = [str(claim) for claim in item.get("unverified_claim_samples", [])]
    return "\n".join(
        [
            f"question={item['question']}",
            f"question_id={item['question_id']}",
            f"classification={item['classification']}",
            f"confidence={item['confidence']}",
            f"trajectory_id={item.get('trajectory_id', '')}",
            f"result_flags={', '.join(item.get('result_verification', {}).get('flags', []))}",
            f"process_flags={', '.join(item.get('process_verification', {}).get('flags', []))}",
            f"quality_flags={', '.join(item['quality_flags'])}",
            f"source_count={item['source_count']}",
            f"source_urls={json.dumps(source_urls, ensure_ascii=False)}",
            f"review_approved={item['review_approved']}",
            f"review_issues={'; '.join(review_issues)}",
            f"unverified_claim_samples={json.dumps(unverified_samples, ensure_ascii=False)}",
            f"recommended_update_carrier={item.get('diagnosis', {}).get('recommended_update_carrier', 'none')}",
            "future_rule=Use evaluation failures as self-evolution input for search planning, answer style, calibration, and skill drafts.",
        ]
    )
