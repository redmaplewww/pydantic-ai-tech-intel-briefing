"""Semantic quality classification for knowledge candidates (Path C).

Distinguishes technical, evidence-based claims from marketing/self-media
rhetoric and vague filler so the validation gate can reject noise even when
the trajectory-count gate is relaxed to one briefing.

An injected LLM runner is the primary judge (same convention as
``cli._build_quality_judge`` and ``SkillLearningTrigger``); a deterministic
keyword fallback keeps the gate offline-capable and testable without a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

LLMRunner = Callable[[str, dict[str, Any]], str]

SEMANTIC_KINDS = ("technical", "marketing", "news", "vague")

#: Instructions for the LLM semantic judge.
SEMANTIC_JUDGE_INSTRUCTIONS = """你是知识候选的语义质检器。判断下面这条候选论断的内容类型。

只输出 JSON：{{"kind": "technical" | "marketing" | "news" | "vague", "reason": "<一句话理由>"}}

判定标准：
- technical：可验证的技术论断——包含机制、数字、协议、接口、实现路径、工程约束、研究结论等，有具体对象。
- marketing：营销/自媒体话术——宣传性形容词（领先、极致、颠覆、遥遥领先、爆款、全网首发、马上抢等），无机制细节。
- news：新闻/事件描述——谁宣布了什么、市场发生了什么，不构成可验证技术论断。
- vague：模糊空话——无具体对象、无机制、无数字，句子是泛泛而谈。

候选信息:
{topic}
{claim}
{evidence_titles}
"""

# Marketing/self-media rhetoric cues (Chinese + English).
_MARKETING_CUES = (
    "遥遥领先", "极致", "颠覆", "爆款", "全网首发", "马上抢", "限量", "秒杀",
    "不容错过", "重磅", "惊人", "震撼", "革命性突破", "行业第一", "首创", "唯一",
    "领先", "王牌", "爆火", "火出圈", "疯狂", "一夜暴富", "红利期", "风口",
    "best-in-class", "game-changing", "revolutionary", "unmissable", "hurry",
    "limited-time", "exclusive deal",
)

# Technical-mechanism cues: a claim with any of these and no marketing cue is
# treated as technical.
_TECHNICAL_CUES = (
    "协议", "接口", "api", "延迟", "吞吐", "带宽", "精度", "召回", "基准",
    "benchmark", "部署", "实现", "架构", "调度", "算子", "并行", "流水线",
    "缓存", "kv", "量化", "剪枝", "蒸馏", "微调", "推理", "显存", "内存",
    "带宽", "tensor", "cuda", "triton", "moe", "attention", "检索", "索引",
    "embedding", "向量", "图", "节点", "拓扑", "一致性", "事务", "回滚",
    "审批", "工单", "mes", "opc", "rest", "语义层", "参数", "权重", "模型",
    "gpu", "hbm", "互联", "带宽", "framework", "pipeline", "latency",
    "throughput", "mechanism", "implementation", "architecture",
)

# Vague-filler cues: abstract talk without concrete referents.
_VAGUE_CUES = (
    "值得关注", "值得期待", "具有重要意义", "未来发展", "前景广阔", "潜力巨大",
    "总的来说", "综上所述", "可以看到", "不难发现", "其实", "可以说",
    "as we can see", "in conclusion", "it is important",
)


@dataclass(frozen=True)
class SemanticVerdict:
    kind: str
    confidence: str
    reason: str
    judge: str  # "llm" | "rules"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "confidence": self.confidence,
            "reason": self.reason,
            "judge": self.judge,
        }


def classify_semantic_quality(
    claim: str,
    topic: str,
    evidence_titles: list[str] | None = None,
    llm_runner: LLMRunner | None = None,
    llm_payload: dict[str, Any] | None = None,
) -> SemanticVerdict:
    """Classify a candidate's semantic quality.

    ``llm_runner`` is the primary judge; when absent or when it returns
    unparsable output, the deterministic rule fallback is used.
    """
    if llm_runner is not None:
        try:
            payload = dict(llm_payload or {})
            payload.setdefault(
                "topic", topic,
            )
            payload.setdefault("claim", claim)
            payload.setdefault(
                "evidence_titles", " ".join(evidence_titles or []),
            )
            instructions = SEMANTIC_JUDGE_INSTRUCTIONS.format(
                topic=topic,
                claim=claim,
                evidence_titles="\n".join(f"- {title}" for title in (evidence_titles or [])[:8]) or "(无)",
            )
            raw = llm_runner(instructions, payload)
            verdict = _parse_llm_verdict(raw)
            if verdict is not None:
                return verdict
        except Exception:
            pass  # fall through to rule judge
    return _rule_verdict(claim, topic)


def _parse_llm_verdict(raw: str) -> SemanticVerdict | None:
    import json

    match = re.search(r"\{[^{}]*\"kind\"\s*:\s*\"[^\"]*\"[^{}]*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    kind = str(data.get("kind", "")).strip().lower()
    if kind not in SEMANTIC_KINDS:
        return None
    return SemanticVerdict(
        kind=kind,
        confidence=str(data.get("confidence") or "medium"),
        reason=str(data.get("reason") or "classified by LLM"),
        judge="llm",
    )


def _rule_verdict(claim: str, topic: str) -> SemanticVerdict:
    text = f"{claim}\n{topic}".lower()
    if any(cue in text for cue in _MARKETING_CUES):
        return SemanticVerdict(
            kind="marketing",
            confidence="high" if _marketing_density(text) >= 2 else "medium",
            reason="marketing/self-media rhetoric cues detected",
            judge="rules",
        )
    technical_hits = sum(1 for cue in _TECHNICAL_CUES if cue in text)
    if technical_hits >= 2:
        return SemanticVerdict(
            kind="technical",
            confidence="medium",
            reason=f"technical mechanism cues present ({technical_hits})",
            judge="rules",
        )
    if len(claim.strip()) < 20 or any(cue in text for cue in _VAGUE_CUES):
        return SemanticVerdict(
            kind="vague",
            confidence="medium",
            reason="claim is too short or contains filler phrasing",
            judge="rules",
        )
    return SemanticVerdict(
        kind="technical",
        confidence="low",
        reason="no marketing cues; default technical by rule fallback",
        judge="rules",
    )


def _marketing_density(text: str) -> int:
    return sum(1 for cue in _MARKETING_CUES if cue in text)


def is_noise_verdict(verdict: SemanticVerdict) -> bool:
    """Whether a semantic verdict should reject a candidate as noise."""
    return verdict.kind in {"marketing", "vague"}
