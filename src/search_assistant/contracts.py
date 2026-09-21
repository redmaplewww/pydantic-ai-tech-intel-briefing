from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


Classification = Literal["simple", "research", "hard", "high_stakes"]
Confidence = Literal["low", "medium", "high"]
SourceTechnicalValue = Literal["high", "medium", "low"]
SourceQualityType = Literal["primary", "secondary", "community", "marketing", "aggregator", "unknown"]
SourceMarketingLevel = Literal["none", "mixed", "dominant"]
SourceEvidenceDensity = Literal["high", "medium", "low"]
SourceAuthority = Literal["primary", "secondary", "weak"]
SourceKeepRecommendation = Literal["keep", "borderline", "reject"]
ProviderEventStatus = Literal[
    "called",
    "success",
    "empty",
    "error",
    "timeout",
    "skipped",
    "filtered",
    "fallback_used",
]
SourceCandidateStatus = Literal[
    "pending",
    "accepted",
    "duplicate",
    "feedback_seed",
    "rejected_invalid_url",
    "rejected_empty_content",
    "rejected_search_page_dump",
    "rejected_login_page",
    "rejected_generic_reference",
    "rejected_cad_medical",
    "rejected_cad_missing_anchor",
    "rejected_missing_industrial_anchor",
    "rejected_low_relevance",
]
MemoryLayer = Literal["event", "preference", "domain_knowledge", "run_experience"]


class IncomingMessage(BaseModel):
    message_id: str
    event_id: str | None = None
    user_id: str
    chat_id: str
    text: str
    source: str = "cli"
    raw_event: dict[str, Any] = Field(default_factory=dict)

    @property
    def dedupe_key(self) -> str:
        return self.event_id or self.message_id


class VerifiedClaim(BaseModel):
    claim: str
    verdict: Literal["verified", "unverified", "contradicted"]
    source: str | None = None
    checked_at: str
    notes: str | None = None


class SourceEvidence(BaseModel):
    title: str
    url: str
    snippet: str = ""
    provider: str
    checked_at: str


class ProviderTraceEvent(BaseModel):
    """A compact, local-only record of one provider attempt.

    This does not store credentials or response bodies.  It exists so a user can
    tell whether a run really called a provider, skipped it, timed out, or fell
    back to another route.
    """

    provider: str
    query: str
    status: ProviderEventStatus
    result_count: int = 0
    reason: str | None = None
    error: str | None = None
    elapsed_ms: float | None = None
    tier: str | None = None
    budget_share: float | None = None
    checked_at: str


class SearchRecord(BaseModel):
    executed: bool
    queries: list[str] = Field(default_factory=list)
    sources: list[SourceEvidence] = Field(default_factory=list)
    engines: list[str] = Field(default_factory=list)
    skipped_reason: str | None = None


class SourceCandidate(BaseModel):
    """A normalized source candidate before and after briefing filters.

    Accepted candidates become ``CollectedSource`` rows.  Rejected candidates are
    kept as short audit records so overly strict filters can be diagnosed without
    leaking private data or forcing noisy items into the final report.
    """

    id: str
    topic_id: str
    user_id: str
    title: str = ""
    url: str = ""
    snippet: str = ""
    platform: str = ""
    provider: str = ""
    query: str = ""
    status: SourceCandidateStatus = "pending"
    reason: str = ""
    relevance_score: float = 0.0
    importance_score: float = 0.0
    retrieved_at: str
    created_at: str


class SourceQualityVerdict(BaseModel):
    technical_value: SourceTechnicalValue = "medium"
    source_type: SourceQualityType = "unknown"
    marketing_level: SourceMarketingLevel = "mixed"
    evidence_density: SourceEvidenceDensity = "medium"
    authority: SourceAuthority = "secondary"
    keep_recommendation: SourceKeepRecommendation = "borderline"
    rationale: str = ""


_SOURCE_TECHNICAL_VALUES = set(SourceTechnicalValue.__args__)
_SOURCE_QUALITY_TYPES = set(SourceQualityType.__args__)
_SOURCE_MARKETING_LEVELS = set(SourceMarketingLevel.__args__)
_SOURCE_EVIDENCE_DENSITIES = set(SourceEvidenceDensity.__args__)
_SOURCE_AUTHORITIES = set(SourceAuthority.__args__)
_SOURCE_KEEP_RECOMMENDATIONS = set(SourceKeepRecommendation.__args__)


def default_source_quality_verdict() -> SourceQualityVerdict:
    return SourceQualityVerdict(
        technical_value="medium",
        source_type="unknown",
        marketing_level="mixed",
        evidence_density="medium",
        authority="secondary",
        keep_recommendation="borderline",
        rationale="fallback borderline source-quality verdict",
    )


def normalize_source_quality_verdict(value: object) -> SourceQualityVerdict:
    fallback = default_source_quality_verdict()
    if isinstance(value, SourceQualityVerdict):
        return value
    if not isinstance(value, dict):
        return fallback
    data = dict(value)
    technical_value = str(data.get("technical_value") or fallback.technical_value)
    source_type = str(data.get("source_type") or fallback.source_type)
    marketing_level = str(data.get("marketing_level") or fallback.marketing_level)
    evidence_density = str(data.get("evidence_density") or fallback.evidence_density)
    authority = str(data.get("authority") or fallback.authority)
    keep_recommendation = str(data.get("keep_recommendation") or fallback.keep_recommendation)
    data["technical_value"] = technical_value if technical_value in _SOURCE_TECHNICAL_VALUES else fallback.technical_value
    data["source_type"] = source_type if source_type in _SOURCE_QUALITY_TYPES else fallback.source_type
    data["marketing_level"] = marketing_level if marketing_level in _SOURCE_MARKETING_LEVELS else fallback.marketing_level
    data["evidence_density"] = evidence_density if evidence_density in _SOURCE_EVIDENCE_DENSITIES else fallback.evidence_density
    data["authority"] = authority if authority in _SOURCE_AUTHORITIES else fallback.authority
    data["keep_recommendation"] = (
        keep_recommendation if keep_recommendation in _SOURCE_KEEP_RECOMMENDATIONS else fallback.keep_recommendation
    )
    data["rationale"] = " ".join(str(data.get("rationale") or fallback.rationale).split())[:240]
    try:
        return SourceQualityVerdict.model_validate(data)
    except Exception:
        return fallback


class LayeredMemoryItem(BaseModel):
    id: str
    layer: MemoryLayer
    kind: str
    content: str
    source_id: str
    confidence: Confidence = "medium"
    status: Literal["candidate", "active", "superseded", "rejected"] = "active"
    user_id: str = "system"
    chat_id: str = "system"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class TopicFeedbackSignal(BaseModel):
    id: str
    topic_id: str
    user_id: str
    chat_id: str
    signal_type: Literal["style", "evidence", "provider", "fact_correction", "case", "general"]
    scope: Literal["run", "source", "section", "topic", "provider"]
    body: str
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class CalibrationResult(BaseModel):
    ran: bool
    critique: str
    revision: str


class MemoryUpdate(BaseModel):
    kind: str
    content: str
    source_id: str | None = None


class AnswerPackage(BaseModel):
    question_id: str
    answer_text: str
    classification: Classification
    confidence: Confidence
    verified_claims: list[VerifiedClaim] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    sources: list[SourceEvidence] = Field(default_factory=list)
    calibration: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    memory_updates: list[MemoryUpdate | dict[str, Any]] = Field(default_factory=list)
    search_record: SearchRecord | None = None
    trajectory_context: dict[str, Any] = Field(default_factory=dict, exclude=True)


class TopicSubscription(BaseModel):
    id: str
    user_id: str
    chat_id: str
    topic: str
    enabled: bool = True
    source_recipe: dict[str, float] = Field(default_factory=dict)
    created_at: str
    updated_at: str


BriefingIntent = Literal[
    "concept_explanation",
    "technical_tracking",
    "industry_trend",
    "engineering_landing",
    "comparison_decision",
]
BriefingTemporalFocus = Literal["evergreen", "current", "historical", "near_future", "unspecified"]


class DomainProfile(BaseModel):
    domain: str = "technology"
    key_concepts: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)
    ambiguous_terms: list[str] = Field(default_factory=list)
    excluded_meanings: list[str] = Field(default_factory=list)
    evidence_anchors: list[str] = Field(default_factory=list)
    preferred_source_types: list[str] = Field(default_factory=list)


class BriefingUnderstanding(BaseModel):
    primary_intent: BriefingIntent = "technical_tracking"
    secondary_intents: list[BriefingIntent] = Field(default_factory=list)
    temporal_focus: BriefingTemporalFocus = "current"
    research_focus: list[str] = Field(default_factory=list)
    evidence_preferences: list[str] = Field(default_factory=list)
    comparison_dimensions: list[str] = Field(default_factory=list)
    domain_profile: DomainProfile = Field(default_factory=DomainProfile)


_BRIEFING_INTENT_VALUES = set(BriefingIntent.__args__)
_BRIEFING_TEMPORAL_FOCUS_VALUES = set(BriefingTemporalFocus.__args__)


def _clean_text_items(value: object, *, limit: int = 8) -> list[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    cleaned: list[str] = []
    for item in raw_items:
        text = " ".join(str(item or "").split())
        if text and text not in cleaned:
            cleaned.append(text[:160])
        if len(cleaned) >= limit:
            break
    return cleaned


def default_briefing_understanding(topic: str) -> BriefingUnderstanding:
    clean_topic = " ".join(topic.split()).strip() or "technology topic"
    return BriefingUnderstanding(
        primary_intent="technical_tracking",
        secondary_intents=[],
        temporal_focus="current",
        research_focus=[clean_topic],
        evidence_preferences=[
            "official documentation",
            "technical architecture",
            "papers and benchmarks",
            "open source repositories",
            "deployment evidence",
        ],
        comparison_dimensions=[],
        domain_profile=DomainProfile(
            domain=clean_topic,
            key_concepts=[clean_topic],
            subtopics=[],
            ambiguous_terms=[],
            excluded_meanings=[],
            evidence_anchors=[],
            preferred_source_types=[
                "official documentation",
                "research papers",
                "benchmarks",
                "repositories",
                "case studies",
            ],
        ),
    )


def normalize_briefing_understanding(topic: str, value: object) -> BriefingUnderstanding:
    fallback = default_briefing_understanding(topic)
    if isinstance(value, BriefingUnderstanding):
        parsed = value
    elif isinstance(value, dict):
        data = dict(value)
        primary = str(data.get("primary_intent") or fallback.primary_intent)
        if primary not in _BRIEFING_INTENT_VALUES:
            primary = fallback.primary_intent
        temporal = str(data.get("temporal_focus") or fallback.temporal_focus)
        if temporal not in _BRIEFING_TEMPORAL_FOCUS_VALUES:
            temporal = fallback.temporal_focus
        secondary = [
            str(item)
            for item in (data.get("secondary_intents") or [])
            if str(item) in _BRIEFING_INTENT_VALUES and str(item) != primary
        ][:4]
        domain_profile = data.get("domain_profile")
        if not isinstance(domain_profile, dict):
            domain_profile = {}
        domain = " ".join(str(domain_profile.get("domain") or fallback.domain_profile.domain).split())
        data["primary_intent"] = primary
        data["secondary_intents"] = secondary
        data["temporal_focus"] = temporal
        data["research_focus"] = _clean_text_items(data.get("research_focus"), limit=8)
        data["evidence_preferences"] = _clean_text_items(data.get("evidence_preferences"), limit=8)
        data["comparison_dimensions"] = _clean_text_items(data.get("comparison_dimensions"), limit=8)
        data["domain_profile"] = {
            "domain": domain or fallback.domain_profile.domain,
            "key_concepts": _clean_text_items(domain_profile.get("key_concepts"), limit=10),
            "subtopics": _clean_text_items(domain_profile.get("subtopics"), limit=10),
            "ambiguous_terms": _clean_text_items(domain_profile.get("ambiguous_terms"), limit=8),
            "excluded_meanings": _clean_text_items(domain_profile.get("excluded_meanings"), limit=8),
            "evidence_anchors": _clean_text_items(domain_profile.get("evidence_anchors"), limit=10),
            "preferred_source_types": _clean_text_items(domain_profile.get("preferred_source_types"), limit=8),
        }
        try:
            parsed = BriefingUnderstanding.model_validate(data)
        except Exception:
            return fallback
    else:
        return fallback

    updates: dict[str, object] = {}
    if not parsed.research_focus:
        updates["research_focus"] = fallback.research_focus
    if not parsed.evidence_preferences:
        updates["evidence_preferences"] = fallback.evidence_preferences
    profile = parsed.domain_profile
    profile_updates: dict[str, object] = {}
    if not profile.domain.strip():
        profile_updates["domain"] = fallback.domain_profile.domain
    if not profile.key_concepts:
        profile_updates["key_concepts"] = fallback.domain_profile.key_concepts
    if not profile.preferred_source_types:
        profile_updates["preferred_source_types"] = fallback.domain_profile.preferred_source_types
    if profile_updates:
        updates["domain_profile"] = profile.model_copy(update=profile_updates)
    if updates:
        return parsed.model_copy(update=updates)
    return parsed


class CollectedSource(BaseModel):
    id: str
    topic_id: str
    user_id: str
    title: str
    url: str
    snippet: str = ""
    platform: str
    provider: str
    query: str
    relevance_score: float = 0.0
    importance_score: float = 0.0
    retrieved_at: str


class BriefingTheme(BaseModel):
    """A traceable evidence anchor for a briefing synthesis.

    The detailed report is intentionally free-form.  These fields retain the
    source-to-analysis relationship and keep legacy deterministic fallbacks
    compatible without forcing the model to use a fixed theme checklist.
    """

    name: str = Field(description="中文分析块名称，不是平台或文章标题。")
    analysis: str = Field(
        default="",
        description="该分析块的关键技术判断；模型输出时以具体实现、约束或工程取舍为中心。",
    )
    what_is_happening: str = Field(
        default="",
        description="说明业内正在构建、试点或部署的系统能力，概括多条来源而非罗列标题。"
    )
    core_technology: str = Field(
        default="",
        description="说明核心模型、算法、软件架构、系统接口、工程方法和关键约束。"
    )
    data_and_workflow: str = Field(
        default="",
        description="说明输入数据、处理/决策过程、结果写回的业务或工程系统，以及人工审核或接管点。"
    )
    why_it_matters: str = Field(default="", description="解释它影响的成本、效率、可靠性、能力边界或产业结构。")
    maturity: str = Field(default="", description="区分概念、试点、局部规模化或可复制能力，并说明剩余验证条件。")
    source_urls: list[str] = Field(default_factory=list, description="支撑该主题的原始来源 URL；有来源时必须来自输入来源。")


class BriefingSynthesis(BaseModel):
    search_content_summary: str = Field(description="说明本轮来源共同讨论的技术主线和领域变化。")
    short_summary: str = Field(description="用于速览，必须说明具体实现路径、关键技术和变化含义。")
    detailed_summary: str = Field(
        default="",
        description="详细总结正文，使用中文 Markdown 自由组织证据驱动的分析块；不得套用固定技术地图模板。",
    )
    themes: list[BriefingTheme] = Field(min_length=1, description="支撑详细总结的证据锚点，按实际证据组织，至少给出一个。")
    key_signal_interpretation: str = Field(description="解释跨来源重复出现的架构、数据闭环、指标或约束。")
    analysis_judgment: str = Field(description="可检查的跨来源归纳，不输出隐藏思维链。")
    next_search_directions: list[str] = Field(min_length=2, description="下一轮应验证的原始资料、指标或部署变化。")
    landing_suggestions: list[str] = Field(min_length=2, description="有边界、指标和人工接管点的实施建议。")


class DailyBriefing(BaseModel):
    id: str
    topic_id: str
    user_id: str
    chat_id: str
    topic: str
    run_date: date
    search_directions: list[str]
    keywords: list[str]
    sources: list[CollectedSource]
    source_candidates: list[SourceCandidate] = Field(default_factory=list)
    provider_events: list[ProviderTraceEvent] = Field(default_factory=list)
    synthesis: BriefingSynthesis
    markdown: str
    created_at: str


DomainKnowledgeEntityType = Literal[
    "concept",
    "system",
    "data",
    "workflow",
    "metric",
    "risk",
    "control",
    "source",
]


class DomainKnowledgeEntity(BaseModel):
    """A reviewed node in a domain-specific GraphRAG-style knowledge map."""

    id: str = Field(description="Stable graph-local entity identifier.")
    name: str
    entity_type: DomainKnowledgeEntityType = "concept"
    aliases: list[str] = Field(default_factory=list)
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DomainKnowledgeRelation(BaseModel):
    """A typed edge between two reviewed domain entities."""

    id: str = Field(description="Stable graph-local relation identifier.")
    source_entity_id: str
    relation_type: str
    target_entity_id: str
    description: str
    evidence_refs: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, ge=0.0)


class DomainKnowledgeGraph(BaseModel):
    """A small, auditable domain graph that keeps triples and prose together."""

    id: str
    name: str
    description: str
    overview: str
    source: str = "reviewed-seed"
    version: str = "1"
    entities: list[DomainKnowledgeEntity] = Field(min_length=1)
    relations: list[DomainKnowledgeRelation] = Field(default_factory=list)


class DomainKnowledgeSearchHit(BaseModel):
    graph_id: str
    graph_name: str
    entity_id: str
    entity_name: str
    entity_type: DomainKnowledgeEntityType
    score: float
    summary: str
    matched_aliases: list[str] = Field(default_factory=list)
    outgoing_relations: list[str] = Field(default_factory=list)
    incoming_relations: list[str] = Field(default_factory=list)


CandidateStatus = Literal["candidate", "validated", "deprecated"]


class DomainKnowledgeEvidence(BaseModel):
    title: str
    url: str
    provider: str
    retrieved_at: str


class DomainKnowledgeCandidate(BaseModel):
    id: str
    topic: str
    claim: str
    applies_when: str
    evidence: list[DomainKnowledgeEvidence] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    confidence: Confidence
    status: CandidateStatus = "candidate"
    source_ids: list[str] = Field(default_factory=list)
    fingerprint: str
    created_at: str
    updated_at: str
