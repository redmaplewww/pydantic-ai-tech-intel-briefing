from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from search_assistant.contracts import Classification


GroundingCategory = Literal["foundational", "evidence_required", "high_stakes"]
QuestionIntent = Literal[
    "concept_explanation",
    "current_fact_lookup",
    "exact_spec_lookup",
    "deployment_assessment",
    "technical_analysis",
    "high_stakes_advice",
    "general_answer",
]
TemporalScope = Literal["stable", "current_or_versioned"]
ClaimType = Literal[
    "concept",
    "current_fact",
    "precise_number",
    "vendor_version_fact",
    "deployment_feasibility",
    "calculation",
    "high_stakes",
]
RiskLevel = Literal["normal", "high"]


@dataclass(frozen=True)
class QuestionProfile:
    question: str
    intent: QuestionIntent
    temporal_scope: TemporalScope
    requested_claim_types: tuple[ClaimType, ...]
    risk_level: RiskLevel
    entities: tuple[str, ...] = ()
    rationale: str = ""

    def has_claim_type(self, claim_type: ClaimType) -> bool:
        return claim_type in self.requested_claim_types

    def to_context(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "temporal_scope": self.temporal_scope,
            "requested_claim_types": list(self.requested_claim_types),
            "risk_level": self.risk_level,
            "entities": list(self.entities),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class GroundingDecision:
    category: GroundingCategory
    allow_stable_model_knowledge: bool
    require_retrieval_sources: bool
    allow_precise_numbers: bool
    allow_vendor_version_claims: bool
    precise_numbers_require_sources: bool = True
    vendor_version_claims_require_sources: bool = True
    high_risk_requires_sources: bool = False
    required_evidence_families: tuple[str, ...] = ()
    specific_entities: tuple[str, ...] = ()
    rationale: str = ""

    def to_context(self) -> dict[str, object]:
        return {
            "category": self.category,
            "allow_stable_model_knowledge": self.allow_stable_model_knowledge,
            "require_retrieval_sources": self.require_retrieval_sources,
            "allow_precise_numbers": self.allow_precise_numbers,
            "allow_vendor_version_claims": self.allow_vendor_version_claims,
            "precise_numbers_require_sources": self.precise_numbers_require_sources,
            "vendor_version_claims_require_sources": self.vendor_version_claims_require_sources,
            "high_risk_requires_sources": self.high_risk_requires_sources,
            "required_evidence_families": list(self.required_evidence_families),
            "specific_entities": list(self.specific_entities),
            "rationale": self.rationale,
        }


def build_question_profile(question: str) -> QuestionProfile:
    specific_entities = tuple(_unique(_specific_entities(question)))
    has_high_stakes = _has_high_stakes_signal(question)
    has_current_fact = _has_current_fact_signal(question)
    has_exact_value = _has_exact_value_signal(question)
    has_vendor_or_version = bool(specific_entities) or _has_vendor_or_version_signal(question)
    has_deployment_reasoning = _has_deployment_or_calculation_signal(question)
    has_calculation = _has_calculation_signal(question)
    foundational = _is_foundational_question(question)
    technical_analysis = _has_technical_analysis_signal(question)

    claim_types: list[ClaimType] = []
    if foundational:
        claim_types.append("concept")
    if has_current_fact:
        claim_types.append("current_fact")
    if has_exact_value:
        claim_types.append("precise_number")
    if has_vendor_or_version:
        claim_types.append("vendor_version_fact")
    if has_deployment_reasoning:
        claim_types.append("deployment_feasibility")
    if has_calculation or has_deployment_reasoning:
        claim_types.append("calculation")
    if has_high_stakes:
        claim_types.append("high_stakes")

    if has_high_stakes:
        intent: QuestionIntent = "high_stakes_advice"
    elif has_deployment_reasoning:
        intent = "deployment_assessment"
    elif technical_analysis:
        intent = "technical_analysis"
    elif has_current_fact:
        intent = "current_fact_lookup"
    elif has_exact_value:
        intent = "exact_spec_lookup"
    elif foundational:
        intent = "concept_explanation"
    else:
        intent = "general_answer"

    temporal_scope: TemporalScope = (
        "current_or_versioned" if has_current_fact or has_vendor_or_version or has_exact_value else "stable"
    )
    rationale_parts: list[str] = []
    if foundational:
        rationale_parts.append("conceptual explanation signal")
    if has_current_fact:
        rationale_parts.append("current/versioned fact signal")
    if has_exact_value:
        rationale_parts.append("precise value/specification signal")
    if has_vendor_or_version:
        rationale_parts.append("specific entity or vendor/version signal")
    if has_deployment_reasoning:
        rationale_parts.append("deployment feasibility signal")
    if has_high_stakes:
        rationale_parts.append("high-stakes signal")

    return QuestionProfile(
        question=question,
        intent=intent,
        temporal_scope=temporal_scope,
        requested_claim_types=tuple(_unique_claim_types(claim_types)),
        risk_level="high" if has_high_stakes else "normal",
        entities=specific_entities,
        rationale=", ".join(rationale_parts) or "no special grounding signal",
    )


def classify_question_profile(profile: QuestionProfile) -> Classification:
    if profile.risk_level == "high" or profile.has_claim_type("high_stakes"):
        return "high_stakes"
    if (
        profile.has_claim_type("deployment_feasibility")
        or profile.has_claim_type("calculation")
        or profile.intent == "technical_analysis"
    ):
        return "hard"
    if profile.has_claim_type("precise_number") and profile.has_claim_type("vendor_version_fact"):
        return "hard"
    if profile.entities:
        return "hard"
    if profile.intent in {"current_fact_lookup", "exact_spec_lookup", "concept_explanation"}:
        return "research"
    if profile.has_claim_type("current_fact") or profile.has_claim_type("precise_number"):
        return "research"
    return "simple"


def decide_grounding_policy(profile: QuestionProfile, classification: Classification) -> GroundingDecision:
    specific_entities = profile.entities
    if classification == "high_stakes" or profile.risk_level == "high":
        return GroundingDecision(
            category="high_stakes",
            allow_stable_model_knowledge=False,
            require_retrieval_sources=True,
            allow_precise_numbers=True,
            allow_vendor_version_claims=True,
            precise_numbers_require_sources=True,
            vendor_version_claims_require_sources=True,
            high_risk_requires_sources=True,
            required_evidence_families=("authoritative_sources",),
            specific_entities=specific_entities,
            rationale="high-stakes question profile",
        )

    has_current_fact = profile.has_claim_type("current_fact")
    has_exact_value = profile.has_claim_type("precise_number")
    has_vendor_or_version = profile.has_claim_type("vendor_version_fact")
    has_deployment_reasoning = profile.has_claim_type("deployment_feasibility")
    foundational = profile.intent == "concept_explanation" or profile.has_claim_type("concept")

    stable_without_required_evidence = not (
        has_current_fact or has_exact_value or has_vendor_or_version or has_deployment_reasoning
    )
    if stable_without_required_evidence and (foundational or profile.intent == "technical_analysis"):
        return GroundingDecision(
            category="foundational",
            allow_stable_model_knowledge=True,
            require_retrieval_sources=False,
            allow_precise_numbers=False,
            allow_vendor_version_claims=False,
            precise_numbers_require_sources=True,
            vendor_version_claims_require_sources=True,
            high_risk_requires_sources=False,
            specific_entities=specific_entities,
            rationale="stable concept, understanding-check, or non-factual technical-analysis profile",
        )

    evidence_families: list[str] = []
    if has_deployment_reasoning:
        evidence_families.extend(("deployment_feasibility", "model_parameters_or_benchmarks"))
    if has_current_fact:
        evidence_families.append("current_fact")
    if has_exact_value:
        evidence_families.append("exact_values")
    if has_vendor_or_version:
        evidence_families.append("vendor_or_version_claims")

    return GroundingDecision(
        category="evidence_required",
        allow_stable_model_knowledge=False,
        require_retrieval_sources=True,
        allow_precise_numbers=True,
        allow_vendor_version_claims=True,
        precise_numbers_require_sources=True,
        vendor_version_claims_require_sources=True,
        high_risk_requires_sources=False,
        required_evidence_families=tuple(_unique(evidence_families or ["retrieval_sources"])),
        specific_entities=specific_entities,
        rationale="current, exact, vendor/version-specific, or deployment profile",
    )


def _has_high_stakes_signal(question: str) -> bool:
    lowered = question.lower()
    return any(
        word in lowered or word in question
        for word in ("medical", "legal", "financial", "investment", "safety", "医疗", "法律", "金融", "投资", "安全")
    )


def _is_foundational_question(question: str) -> bool:
    lowered = question.lower()
    markers = (
        "can i understand",
        "can be understood",
        "is it fair to say",
        "is it correct to think",
        "how should i understand",
        "what is",
        "explain",
        "concept",
        "principle",
        "mechanism",
        "是否可以理解为",
        "可以理解为",
        "能否理解为",
        "能不能理解为",
        "是不是可以理解",
        "如何理解",
        "怎么理解",
        "是什么",
        "什么是",
        "解释",
        "解释一下",
        "介绍一下",
        "基础",
        "概念",
        "原理",
        "机制",
    )
    return any(marker in lowered or marker in question for marker in markers)


def _has_current_fact_signal(question: str) -> bool:
    lowered = question.lower()
    return any(
        marker in lowered or marker in question
        for marker in (
            "latest",
            "current",
            "newest",
            "today",
            "as of",
            "recent",
            "release date",
            "schedule",
            "price",
            "ranking",
            "benchmark",
            "version",
            "2026",
            "最新",
            "当前",
            "今天",
            "截至",
            "目前",
            "现在",
            "发布",
            "发布日期",
            "版本",
            "排名",
            "价格",
            "基准",
            "实测",
        )
    )


def _has_exact_value_signal(question: str) -> bool:
    lowered = question.lower()
    if re.search(
        r"\b\d+(?:\.\d+)?\s*(?:gb|tb|mb|gb/s|tb/s|gt/s|tok/s|tokens/s|token/s|ms|%)\b",
        lowered,
    ):
        return True
    return any(
        marker in lowered or marker in question
        for marker in (
            "exact",
            "specific number",
            "specification",
            "specs",
            "parameter",
            "parameters",
            "how many",
            "capacity",
            "bandwidth",
            "latency",
            "throughput",
            "精确",
            "准确",
            "具体数字",
            "规格",
            "参数",
            "多少",
            "容量",
            "显存",
            "内存",
            "带宽",
            "延迟",
            "吞吐",
        )
    )


def _has_vendor_or_version_signal(question: str) -> bool:
    return bool(
        re.search(
            r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]+[-\s]?v\d+[A-Za-z0-9-]*(?![A-Za-z0-9])",
            question,
            flags=re.IGNORECASE,
        )
        or re.search(r"(?<![A-Za-z0-9])[A-Z]{2,}\s+[A-Z]?\d{2,}[A-Za-z0-9-]*(?![A-Za-z0-9])", question)
        or re.search(r"(?<![A-Za-z0-9])[A-Za-z]+-\d+[A-Za-z0-9-]*(?![A-Za-z0-9])", question)
    )


def _has_deployment_or_calculation_signal(question: str) -> bool:
    lowered = question.lower()
    return any(
        marker in lowered or marker in question
        for marker in (
            "deploy",
            "deployment",
            "can deploy",
            "could deploy",
            "run on",
            "runs on",
            "fit in memory",
            "fit the model",
            "cluster",
            "feasibility",
            "部署",
            "运行",
            "装下",
            "集群",
            "可行",
        )
    )


def _has_calculation_signal(question: str) -> bool:
    lowered = question.lower()
    return any(
        marker in lowered or marker in question
        for marker in (
            "calculate",
            "calculation",
            "estimate",
            "assumption",
            "assumptions",
            "计算",
            "估算",
            "假设",
        )
    )


def _has_technical_analysis_signal(question: str) -> bool:
    lowered = question.lower()
    return any(
        marker in lowered or marker in question
        for marker in (
            "compare",
            "migration",
            "risks",
            "architecture",
            "debug",
            "why",
            "bottleneck",
            "inference",
            "throughput",
            "tok/s",
            "tokens per second",
            "hbm",
            "kv cache",
            "memory bandwidth",
            "bandwidth",
            "learning direction",
            "next step",
            "比较",
            "风险",
            "架构",
            "调试",
            "为什么",
            "瓶颈",
            "带宽",
            "吞吐",
            "推理",
            "学习方向",
            "学习路线",
            "下一步",
        )
    )


def _specific_entities(question: str) -> list[str]:
    entities: list[str] = []
    patterns = (
        (r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]+[-\s]?v\d+[A-Za-z0-9-]*(?![A-Za-z0-9])", re.IGNORECASE),
        (r"(?<![A-Za-z0-9])[A-Z]{2,}\s+[A-Z]?\d{2,}[A-Za-z0-9-]*(?![A-Za-z0-9])", 0),
        (r"(?<![A-Za-z0-9])[A-Z]{1,8}\d{2,}[A-Za-z0-9-]*(?![A-Za-z0-9])", 0),
        (r"(?<![A-Za-z0-9])[A-Za-z]+-\d+[A-Za-z0-9-]*(?![A-Za-z0-9])", re.IGNORECASE),
    )
    for pattern, flags in patterns:
        for match in re.finditer(pattern, question, flags=flags):
            entities.append(re.sub(r"\s+", " ", match.group(0)).strip())
    return entities


def _unique(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _unique_claim_types(values: list[ClaimType]) -> list[ClaimType]:
    unique: list[ClaimType] = []
    seen: set[ClaimType] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
