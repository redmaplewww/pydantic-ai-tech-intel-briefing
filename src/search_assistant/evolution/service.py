from __future__ import annotations

import hashlib
import re
import uuid
from datetime import UTC, datetime
from typing import Any, Callable
from urllib.parse import urlparse

from search_assistant.contracts import (
    DailyBriefing,
    DomainKnowledgeCandidate,
    DomainKnowledgeEvidence,
)
from search_assistant.evolution.semantics import SemanticVerdict, classify_semantic_quality, is_noise_verdict
from search_assistant.knowledge_graph.embedding import EmbeddingProvider, build_embedding_provider
from search_assistant.knowledge_graph.service import DomainKnowledgeGraphService
from search_assistant.memory.store import MemoryStore


KNOWLEDGE_LAYER_VALIDATED = "validated_knowledge"
KNOWLEDGE_LAYER_WEAK_SIGNAL = "weak_signal"
KNOWLEDGE_LAYER_REJECTED = "rejected_noise"
KNOWLEDGE_LAYER_UNREVIEWED = "unreviewed_candidate"

# Domains whose content is treated as primary technical evidence: papers,
# source repositories, model hubs, and standards bodies.  One such URL counts
# as three generic URLs in the authority gate.
_PRIMARY_SOURCE_DOMAINS = frozenset(
    {
        "arxiv.org",
        "github.com",
        "gitlab.com",
        "huggingface.co",
        "hf-mirror.com",
        "modelscope.cn",
        "paperswithcode.com",
        "openreview.net",
        "doi.org",
        "ieee.org",
        "acm.org",
        "nature.com",
        "science.org",
        "springer.com",
        "sciencedirect.com",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "arxiv-vanity.com",
        "pytorch.org",
        "tensorflow.org",
        "w3.org",
        "ietf.org",
        "oasis-open.org",
    }
)

# Marketing-funnel domains that contribute only a third of a generic URL.
_WEAK_SOURCE_DOMAINS = frozenset(
    {
        "mp.weixin.qq.com",
        "weixin.qq.com",
        "toutiao.com",
        "xiaohongshu.com",
        "xhslink.com",
        "baijiahao.baidu.com",
        "sohu.com",
    }
)

_PRIMARY_SOURCE_WEIGHT = 3.0
_WEAK_SOURCE_WEIGHT = 1.0 / 3.0
_NEUTRAL_SOURCE_WEIGHT = 1.0
_MIN_EFFECTIVE_SOURCE_COUNT = 2.0

# Minimum number of distinct briefing trajectories that must support a
# candidate.  Lowered to 1: with semantic quality gating and user-confirmed
# promotion in place, a single well-sourced briefing is enough to become a
# reviewable candidate; the marketing/vague-content gate and the
# pending-user-confirm step provide the noise control that a second
# trajectory used to provide.
_MIN_INDEPENDENT_TRAJECTORIES = 1

# How old evidence must be before a candidate can no longer validate.
_EVIDENCE_STALE_DAYS = 30

# Fuzzy-merge thresholds: a new claim merges into an existing candidate when
# its normalized topic matches and it shares at least two claim tokens with at
# least 20% of the smaller token set.  Claims are long paragraphs; for the
# tracked-topic-pool usage pattern, same-topic rephrased claims should merge so
# the cross-trajectory gate can fire.  Chinese claims are split on common
# connectives so shared phrases ("低延时", "高可靠") actually produce shared
# tokens instead of whole-run strings.
_MIN_SHARED_CLAIM_TOKENS = 2
_CLAIM_OVERLAP_RATIO = 0.2
_CJK_TRIGRAM_JACCARD_MIN = 0.14

# Connectives and function words used to split CJK runs into phrase tokens.
_CJK_SEPARATORS = (
    "而不是", "意味着", "并且", "以及", "通过", "作为", "提出", "认为", "明确",
    "需要", "要求", "说明", "不能", "可以", "本批", "材料", "主要", "证据",
    "包括", "部署", "写入", "直接", "以及", "和", "与", "为", "以", "是", "在",
    "把", "将", "从", "到", "对", "被", "使", "让", "而", "或", "也", "都",
    "仍", "需", "该", "这", "那", "其", "中", "的", "了",
)
_CJK_SEPARATOR_PATTERN = "|".join(
    re.escape(word) for word in sorted(_CJK_SEPARATORS, key=len, reverse=True)
)


def _normalize_text(value: str) -> str:
    """Lowercase and collapse separators so phrasing variants compare fairly."""
    return re.sub(r"\s+|[,，。；;:：、.!?！？()（）\[\]【】\"']", " ", value).strip().lower()


def _topic_similar(left: str, right: str) -> bool:
    """Topics are similar when normalized equal, one contains the other, or the
    token overlap is high (>= 0.8 of the smaller set)."""
    if not left or not right:
        return False
    a, b = _normalize_text(left), _normalize_text(right)
    if a == b or a in b or b in a:
        return True
    ta, tb = set(re.findall(r"[a-z0-9]+", a)) | set(re.findall(r"[\u4e00-\u9fff]{2,}", a)), set(
        re.findall(r"[a-z0-9]+", b)
    ) | set(re.findall(r"[\u4e00-\u9fff]{2,}", b))
    if not ta or not tb:
        return False
    return len(ta & tb) / max(1, min(len(ta), len(tb))) >= 0.8


def _claim_tokens(claim: str) -> set[str]:
    """Claim tokens: latin words plus CJK phrases.

    CJK runs are split on common connectives so that "低延时、高可靠、可审计"
    yields "低延时" / "高可靠" / "可审计" tokens instead of one opaque run.
    """
    tokens = set(re.findall(r"[a-z0-9]+", claim.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", claim):
        for piece in re.split(_CJK_SEPARATOR_PATTERN, run):
            if len(piece) >= 2:
                tokens.add(piece)
    return tokens


def _cjk_char_sequence(text: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fff]", text.lower()))


def _cjk_trigrams(text: str) -> set[str]:
    seq = _cjk_char_sequence(text)
    return {seq[index : index + 3] for index in range(len(seq) - 2)}


def _claims_similar(left: str, right: str) -> bool:
    a, b = _claim_tokens(left), _claim_tokens(right)
    if a and b:
        shared = len(a & b)
        ratio = shared / max(1, min(len(a), len(b)))
        if shared >= _MIN_SHARED_CLAIM_TOKENS and ratio >= _CLAIM_OVERLAP_RATIO:
            return True
    # Chinese fallback: whole-run tokenization still fragments rephrased
    # sentences, so also compare character trigram overlap.  Shared phrases
    # ("低延时", "高可靠", "可审计") produce shared trigrams; unrelated
    # same-topic claims share only topic words and stay below the threshold.
    left_seq, right_seq = _cjk_char_sequence(left), _cjk_char_sequence(right)
    if len(left_seq) >= 15 and len(right_seq) >= 15:
        left_grams, right_grams = _cjk_trigrams(left), _cjk_trigrams(right)
        if left_grams and right_grams:
            jaccard = len(left_grams & right_grams) / len(left_grams | right_grams)
            if jaccard >= _CJK_TRIGRAM_JACCARD_MIN:
                return True
    return False


def source_authority_weight(url: str) -> float:
    """Weight an evidence URL by source authority.

    Primary sources (papers, repositories, model hubs) count three times as
    much as generic URLs; marketing-funnel domains count one third.  Hosts are
    matched by exact suffix so subdomains (``x.github.io``, ``y.arxiv.org``)
    inherit the parent's weight.
    """
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return _NEUTRAL_SOURCE_WEIGHT
    for suffix in _PRIMARY_SOURCE_DOMAINS:
        if host == suffix or host.endswith("." + suffix):
            return _PRIMARY_SOURCE_WEIGHT
    for suffix in _WEAK_SOURCE_DOMAINS:
        if host == suffix or host.endswith("." + suffix):
            return _WEAK_SOURCE_WEIGHT
    return _NEUTRAL_SOURCE_WEIGHT


def evidence_source_domains(evidence_urls: set[str]) -> set[str]:
    """Distinct netlocs across a candidate's evidence URLs."""
    domains: set[str] = set()
    for url in evidence_urls:
        try:
            host = urlparse(url).netloc.lower()
        except ValueError:
            continue
        if host:
            domains.add(host)
    return domains


class EvolutionDiagnosisService:
    """Route structured evaluation failures to a reviewable update carrier."""

    def diagnose(self, item: dict[str, Any], trajectory_id: str) -> dict[str, Any]:
        result_flags = list(item["result_verification"]["flags"])
        process_flags = list(item["process_verification"]["flags"])
        quality_flags = list(item["quality_verification"]["flags"])
        all_flags = list(dict.fromkeys(result_flags + process_flags + quality_flags))

        if "blocked_answer" in all_flags or "task_blocked" in all_flags:
            carrier = "program_harness"
            target = "failure recovery and safe fallback routing"
        elif any(flag in all_flags for flag in ("required_search_not_executed", "search_returned_no_sources")):
            carrier = "prompt_skill"
            target = "search planning and source acquisition"
        elif any(flag in all_flags for flag in ("unverified_claims", "review_rejected", "low_confidence")):
            carrier = "prompt_skill"
            target = "claim verification and final review"
        elif "missing_calibration" in all_flags:
            carrier = "prompt_skill"
            target = "calibration policy"
        else:
            carrier = "none"
            target = "no candidate update"

        evidence_refs = [f"trajectory:{trajectory_id}", f"question:{item['question_id']}"]
        evidence_refs.extend(f"source:{url}" for url in item.get("source_urls", []))
        return {
            "root_causes": all_flags,
            "recommended_update_carrier": carrier,
            "target": target,
            "evidence_refs": evidence_refs,
            "confidence": "high" if len(all_flags) >= 2 else "medium" if all_flags else "high",
            "candidate_only": True,
        }


class DomainKnowledgeCandidateService:
    """Turn briefing evidence into deduplicated, reviewable knowledge candidates."""

    def __init__(
        self,
        store: MemoryStore,
        graph_service: DomainKnowledgeGraphService | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        llm_runner: Callable[[str, dict[str, Any]], str] | None = None,
    ):
        self.store = store
        self.embedding_provider = embedding_provider or build_embedding_provider()
        self.graph_service = graph_service or DomainKnowledgeGraphService(
            store, embedding_provider=self.embedding_provider
        )
        # Semantic quality judge: LLM primary (injected), rule fallback in
        # evolution.semantics.  None keeps the gate offline with rules only.
        self.llm_runner = llm_runner

    def capture_briefing(self, briefing: DailyBriefing) -> list[str]:
        if not briefing.sources:
            return []

        source_by_url = {source.url: source for source in briefing.sources}
        candidate_ids: list[str] = []
        for theme in briefing.synthesis.themes:
            claim = (theme.analysis or theme.what_is_happening or briefing.synthesis.analysis_judgment).strip()
            if not claim:
                continue
            evidence_urls = [url for url in theme.source_urls if url in source_by_url]
            if not evidence_urls:
                evidence_urls = [source.url for source in briefing.sources[:3]]
            evidence = [
                DomainKnowledgeEvidence(
                    title=source_by_url[url].title,
                    url=url,
                    provider=source_by_url[url].provider,
                    retrieved_at=source_by_url[url].retrieved_at,
                )
                for url in dict.fromkeys(evidence_urls)
            ]
            now = datetime.now(UTC).isoformat()
            fingerprint = self._fingerprint(briefing.topic, claim)
            # Fuzzy merge: when a rephrased claim from the same tracked topic
            # already exists, reuse its fingerprint so the store merge path
            # accumulates evidence instead of creating a parallel candidate.
            similar = self._find_similar_candidate(briefing.topic, claim)
            if similar is not None:
                fingerprint = str(similar["fingerprint"])
            candidate = DomainKnowledgeCandidate(
                id=f"knowledge_{uuid.uuid4().hex}",
                topic=briefing.topic,
                claim=claim,
                applies_when=(
                    f"Use as planning context when researching {briefing.topic}; "
                    "re-open original sources before making current factual claims."
                ),
                evidence=evidence,
                contradictions=[],
                confidence=self._confidence(len(evidence)),
                status="candidate",
                source_ids=[briefing.id, *[source_by_url[url].id for url in dict.fromkeys(evidence_urls)]],
                fingerprint=fingerprint,
                created_at=now,
                updated_at=now,
            )
            candidate_ids.append(self.store.upsert_domain_knowledge_candidate(candidate)[0])
        unique_ids = list(dict.fromkeys(candidate_ids))
        self.link_candidates_to_knowledge_graphs(unique_ids)
        return unique_ids

    def link_candidates_to_knowledge_graphs(
        self,
        candidate_ids: list[str] | None = None,
        limit_per_candidate: int = 1,
    ) -> dict[str, Any]:
        """Attach reviewable candidates to reviewed graph entities without promoting them."""

        seed_result = self._ensure_default_graphs()
        requested_ids = {candidate_id for candidate_id in candidate_ids or []}
        candidates = [
            candidate
            for candidate in self.store.list_domain_knowledge_candidates()
            if not requested_ids or candidate["id"] in requested_ids
        ]
        existing_keys = {
            (
                str(link["candidate_id"]),
                str(link["graph_id"]),
                str(link["entity_id"] or ""),
                str(link["relation_id"] or ""),
                str(link["link_type"]),
            )
            for link in self.store.list_domain_candidate_graph_links()
        }

        created_links: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate["status"] == "deprecated":
                continue
            query = self._candidate_graph_query(candidate)
            hits = self.graph_service.query_relevant(query, limit=limit_per_candidate, min_score=5.0)
            for hit in hits:
                key = (
                    str(candidate["id"]),
                    hit.graph_id,
                    hit.entity_id,
                    "",
                    "candidate_matches_entity",
                )
                if key in existing_keys:
                    continue
                link_id = self.store.link_domain_candidate_to_graph(
                    candidate_id=str(candidate["id"]),
                    graph_id=hit.graph_id,
                    entity_id=hit.entity_id,
                    link_type="candidate_matches_entity",
                    note=(
                        "Matched by self-evolution candidate text; "
                        f"score={hit.score}; entity={hit.entity_name}"
                    ),
                )
                existing_keys.add(key)
                created_links.append(
                    {
                        "id": link_id,
                        "candidate_id": str(candidate["id"]),
                        "graph_id": hit.graph_id,
                        "entity_id": hit.entity_id,
                        "score": hit.score,
                    }
                )

        return {
            "seeded_graphs": seed_result["seeded_graphs"],
            "candidates_considered": len(candidates),
            "created_links": len(created_links),
            "links": created_links,
        }

    def record_validation_gate(self, candidate_id: str) -> dict[str, Any]:
        """Record the automatic traceability gate without promoting the candidate."""
        return self._run_validation_gate(candidate_id, promote=False)

    def validate(self, candidate_id: str) -> dict[str, Any]:
        return self._run_validation_gate(candidate_id, promote=True)

    def _promote_to_pending(self, candidate_id: str) -> dict[str, Any]:
        """Run the full gate and, on success, move the candidate to
        ``pending_user_confirm`` so a user decision is required before it
        becomes ``validated`` (used by automatic distillation)."""
        return self._run_validation_gate(candidate_id, promote="pending")

    def _semantic_verdict(self, candidate: dict[str, Any]) -> SemanticVerdict | None:
        """Classify the candidate's claim semantics (technical vs marketing/vague).

        Uses the injected LLM runner when available; otherwise the rule
        fallback in ``evolution.semantics``.  Returns None only when the
        candidate has no claim text at all (the no-original-source gate
        already covers the empty case).
        """
        claim = str(candidate.get("claim") or "").strip()
        if not claim:
            return None
        evidence_titles = [
            str(item.get("title") or "")
            for item in (candidate.get("evidence") or [])
            if str(item.get("title") or "").strip()
        ]
        return classify_semantic_quality(
            claim=claim,
            topic=str(candidate.get("topic") or ""),
            evidence_titles=evidence_titles,
            llm_runner=self.llm_runner,
        )

    def list_candidates(self, status: str | None = None, layer: str | None = None) -> list[dict[str, Any]]:
        candidates = self.store.list_domain_knowledge_candidates(status)
        latest_layers = self._latest_validation_layers()
        enriched: list[dict[str, Any]] = []
        for candidate in candidates:
            validation = latest_layers.get(str(candidate["id"]), {})
            item = {
                **candidate,
                "knowledge_layer": validation.get("knowledge_layer", KNOWLEDGE_LAYER_UNREVIEWED),
                "knowledge_layer_reason": validation.get("reason", "validation gate has not run"),
                "knowledge_layer_next_action": validation.get(
                    "next_action",
                    "run knowledge-candidate-validate or generate a new briefing",
                ),
            }
            if layer is None or item["knowledge_layer"] == layer:
                enriched.append(item)
        return enriched

    def _run_validation_gate(self, candidate_id: str, promote: bool | str) -> dict[str, Any]:
        candidate = self.store.get_domain_knowledge_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"domain knowledge candidate not found: {candidate_id}")
        evidence_urls = {
            str(item.get("url", ""))
            for item in candidate["evidence"]
            if str(item.get("url", "")).startswith(("http://", "https://"))
        }
        source_domains = evidence_source_domains(evidence_urls)
        effective_source_count = round(
            sum(source_authority_weight(url) for url in evidence_urls), 3
        )
        failures: list[str] = []
        if not evidence_urls:
            failures.append("no_original_source")
        elif len(evidence_urls) < 2 or len(source_domains) < 2:
            # "Independent" means at least two deduplicated URLs from at least
            # two distinct domains; two URLs from one domain are one source.
            failures.append("fewer_than_two_independent_sources")
        if effective_source_count < _MIN_EFFECTIVE_SOURCE_COUNT:
            # Authority gate: papers/repositories/model hubs count triple,
            # marketing-funnel domains count one third, so a marketing-heavy
            # evidence set fails even when it has enough raw URLs.
            failures.append("insufficient_source_authority")
        if candidate["contradictions"]:
            failures.append("unresolved_contradictions")
        independent_trajectories = self.independent_trajectory_count(candidate)
        if independent_trajectories < _MIN_INDEPENDENT_TRAJECTORIES:
            failures.append("fewer_than_two_independent_trajectories")
        oldest_evidence = self._oldest_evidence_retrieved_at(candidate)
        if (
            oldest_evidence is not None
            and (datetime.now(UTC) - oldest_evidence).days > _EVIDENCE_STALE_DAYS
        ):
            failures.append("stale_evidence")
        if candidate["confidence"] == "low":
            failures.append("low_confidence")
        semantic = self._semantic_verdict(candidate)
        if semantic is not None and is_noise_verdict(semantic):
            failures.append(f"marketing_or_vague_content:{semantic.kind}")
        knowledge_layer = self._knowledge_layer(failures)
        layer_metadata = self._layer_metadata(knowledge_layer)
        if failures:
            self.store.add_gate_record(
                gate_type="domain_knowledge_candidate_validation",
                subject_type="domain_knowledge_candidate",
                subject_id=candidate_id,
                result="failed",
                reason=", ".join(failures),
                evidence_refs=self._candidate_evidence_refs(candidate),
                metadata={
                    "topic": candidate["topic"],
                    "confidence": candidate["confidence"],
                    "evidence_url_count": len(evidence_urls),
                    "source_domain_count": len(source_domains),
                    "effective_source_count": effective_source_count,
                    "independent_trajectory_count": independent_trajectories,
                    "candidate_status": candidate["status"],
                    "knowledge_layer": knowledge_layer,
                    "semantic": semantic.to_dict() if semantic is not None else None,
                    **layer_metadata,
                },
            )
            return {
                "validated": False,
                "candidate_id": candidate_id,
                "failures": failures,
                "source_domain_count": len(source_domains),
                "effective_source_count": effective_source_count,
                "independent_trajectory_count": independent_trajectories,
                "knowledge_layer": knowledge_layer,
                **layer_metadata,
            }

        if promote is True:
            self.store.update_domain_knowledge_candidate_status(
                candidate_id,
                "validated",
                "passed traceability gate: multiple independent domains with sufficient source authority "
                "across at least one independent trajectory, no unresolved contradictions, non-low confidence",
            )
        elif promote == "pending" and str(candidate["status"]) != "validated":
            self.store.update_domain_knowledge_candidate_status(
                candidate_id,
                "pending_user_confirm",
                "passed every validation gate; awaiting explicit user confirmation before release",
            )
        self.store.add_gate_record(
            gate_type="domain_knowledge_candidate_validation",
            subject_type="domain_knowledge_candidate",
            subject_id=candidate_id,
            result="passed",
            reason="multiple independent domains with sufficient source authority across at least one "
            "independent trajectory, no unresolved contradictions, non-low confidence",
            evidence_refs=self._candidate_evidence_refs(candidate),
            metadata={
                "topic": candidate["topic"],
                "confidence": candidate["confidence"],
                "evidence_url_count": len(evidence_urls),
                "source_domain_count": len(source_domains),
                "effective_source_count": effective_source_count,
                "independent_trajectory_count": independent_trajectories,
                "candidate_status": candidate["status"],
                "knowledge_layer": knowledge_layer,
                "semantic": semantic.to_dict() if semantic is not None else None,
                **layer_metadata,
            },
        )
        return {
            "validated": True,
            "candidate_id": candidate_id,
            "failures": [],
            "source_domain_count": len(source_domains),
            "effective_source_count": effective_source_count,
            "independent_trajectory_count": independent_trajectories,
            "knowledge_layer": knowledge_layer,
            **layer_metadata,
        }

    def distill_candidates(self, limit: int | None = None) -> dict[str, Any]:
        """Re-run validation gates for non-deprecated candidates.

        Called by the offline evolution loop after trajectories are verified.
        A candidate that passes every gate moves to status
        ``pending_user_confirm`` — it is *eligible* for promotion but is not
        marked ``validated`` until a user explicitly confirms it in chat or
        via ``knowledge-candidate-confirm``.  Deprecated candidates are never
        promoted and already-validated candidates are re-gated without
        downgrading.
        """
        candidates = [
            candidate
            for candidate in self.store.list_domain_knowledge_candidates()
            if str(candidate["status"]) != "deprecated"
        ]
        if limit is not None:
            candidates = candidates[:limit]
        results = []
        for candidate in candidates:
            status = str(candidate["status"])
            if status == "validated":
                results.append(self.record_validation_gate(str(candidate["id"])))
            else:
                # Full gate run; on success the candidate is promoted to
                # pending_user_confirm, never straight to validated.
                results.append(self._promote_to_pending(str(candidate["id"])))
        layer_counts: dict[str, int] = {}
        for result in results:
            layer = str(result["knowledge_layer"])
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        return {
            "considered": len(candidates),
            "layer_counts": layer_counts,
            "validated": layer_counts.get(KNOWLEDGE_LAYER_VALIDATED, 0),
            "weak_signal": layer_counts.get(KNOWLEDGE_LAYER_WEAK_SIGNAL, 0),
            "rejected_noise": layer_counts.get(KNOWLEDGE_LAYER_REJECTED, 0),
            "details": results,
        }

    def confirm_candidate(
        self,
        candidate_id: str,
        reviewer: str = "user-confirmed",
        reason: str = "user confirmed after review",
    ) -> dict[str, Any]:
        """Promote a pending candidate to ``validated`` after explicit user
        confirmation (chat command or CLI).  Records a human-review gate and a
        project ledger entry so the user decision is auditable."""
        candidate = self.store.get_domain_knowledge_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"domain knowledge candidate not found: {candidate_id}")
        status = str(candidate["status"])
        if status == "validated":
            return {"confirmed": True, "already_validated": True, "candidate_id": candidate_id}
        if status != "pending_user_confirm":
            raise ValueError(
                f"candidate {candidate_id} is {status}; only pending_user_confirm candidates "
                "can be confirmed for release"
            )
        self.store.update_domain_knowledge_candidate_status(
            candidate_id,
            "validated",
            reason or "user confirmed after review",
        )
        gate_id = self.store.add_gate_record(
            gate_type="domain_knowledge_candidate_human_review",
            subject_type="domain_knowledge_candidate",
            subject_id=candidate_id,
            result="passed",
            reason=reason or "user confirmed after review",
            evidence_refs=self._candidate_evidence_refs(candidate),
            metadata={"reviewer": reviewer, "topic": candidate["topic"], "confirmation": "chat-or-cli"},
        )
        ledger_id = self.store.add_project_ledger_entry(
            entry_type="knowledge_release",
            subject=candidate_id,
            status="approved",
            summary=f"Domain knowledge candidate confirmed by user for reviewed use: {candidate['topic']}",
            evidence_refs=[gate_id, *self._candidate_evidence_refs(candidate)],
            risk="User confirmation does not replace fresh source verification before current factual claims.",
            rollback="knowledge-candidate-deprecate to remove the candidate.",
            metadata={"reviewer": reviewer, "candidate_id": candidate_id, "topic": candidate["topic"]},
        )
        return {
            "confirmed": True,
            "already_validated": False,
            "candidate_id": candidate_id,
            "gate_id": gate_id,
            "ledger_id": ledger_id,
            "topic": candidate["topic"],
        }

    @staticmethod
    def independent_trajectory_count(candidate: dict[str, Any]) -> int:
        """Count distinct briefing runs (trajectories) that support a candidate.

        ``capture_briefing`` records the briefing id (``brief_...``) as the
        first source id; merged candidates accumulate every briefing id that
        produced the same claim, so this is the cross-trajectory support count
        used by the validation gate.
        """
        source_ids = candidate.get("source_ids") or []
        return len(
            {
                str(source_id)
                for source_id in source_ids
                if re.match(r"^brief", str(source_id), flags=re.IGNORECASE)
            }
        )

    def approve(
        self,
        candidate_id: str,
        reviewer: str = "local-reviewer",
        reason: str = "approved after human review",
        eval_gate: dict[str, Any] | None = None,
        require_eval_gate: bool = True,
    ) -> dict[str, Any]:
        eval_release_gate: dict[str, Any] | None = None
        if require_eval_gate:
            eval_release_gate = self._record_eval_release_gate(candidate_id, eval_gate)
            if not eval_release_gate["passed"]:
                return {
                    "approved": False,
                    "candidate_id": candidate_id,
                    "gate_id": eval_release_gate["gate_id"],
                    "validation": None,
                    "eval_gate": eval_release_gate,
                }
        elif eval_gate is not None:
            eval_release_gate = self._record_eval_release_gate(candidate_id, eval_gate)

        validation = self.validate(candidate_id)
        candidate = self.store.get_domain_knowledge_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"domain knowledge candidate not found: {candidate_id}")
        if not validation["validated"]:
            gate_id = self.store.add_gate_record(
                gate_type="domain_knowledge_candidate_human_review",
                subject_type="domain_knowledge_candidate",
                subject_id=candidate_id,
                result="failed",
                reason="human approval blocked because validation gate failed: "
                + ", ".join(validation["failures"]),
                evidence_refs=self._candidate_evidence_refs(candidate),
                metadata={"reviewer": reviewer, "topic": candidate["topic"]},
            )
            return {
                "approved": False,
                "candidate_id": candidate_id,
                "gate_id": gate_id,
                "validation": validation,
                "eval_gate": eval_release_gate or eval_gate,
            }

        gate_id = self.store.add_gate_record(
            gate_type="domain_knowledge_candidate_human_review",
            subject_type="domain_knowledge_candidate",
            subject_id=candidate_id,
            result="passed",
            reason=reason.strip() or "approved after human review",
            evidence_refs=self._candidate_evidence_refs(candidate),
            metadata={"reviewer": reviewer, "topic": candidate["topic"]},
        )
        self.store.add_project_ledger_entry(
            entry_type="knowledge_release",
            subject=candidate_id,
            status="approved",
            summary=f"Domain knowledge candidate approved for reviewed use: {candidate['topic']}",
            evidence_refs=[
                gate_id,
                *([str(eval_release_gate["gate_id"])] if eval_release_gate else []),
                *self._candidate_evidence_refs(candidate),
            ],
            risk="Candidate is planning context only; source URLs must be reopened before current factual claims.",
            rollback="Run knowledge-candidate-deprecate with a reason to remove the candidate from active use.",
            metadata={"reviewer": reviewer, "claim": candidate["claim"], "eval_gate": eval_release_gate or eval_gate or {}},
        )
        return {
            "approved": True,
            "candidate_id": candidate_id,
            "gate_id": gate_id,
            "validation": validation,
            "eval_gate": eval_release_gate or eval_gate,
        }

    def deprecate(self, candidate_id: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("deprecation reason must not be empty")
        candidate = self.store.get_domain_knowledge_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"domain knowledge candidate not found: {candidate_id}")
        self.store.update_domain_knowledge_candidate_status(candidate_id, "deprecated", reason.strip())
        gate_id = self.store.add_gate_record(
            gate_type="domain_knowledge_candidate_rollback",
            subject_type="domain_knowledge_candidate",
            subject_id=candidate_id,
            result="passed",
            reason=reason.strip(),
            evidence_refs=self._candidate_evidence_refs(candidate),
            metadata={"topic": candidate["topic"]},
        )
        self.store.add_project_ledger_entry(
            entry_type="knowledge_rollback",
            subject=candidate_id,
            status="deprecated",
            summary=f"Domain knowledge candidate deprecated: {candidate['topic']}",
            evidence_refs=[gate_id, *self._candidate_evidence_refs(candidate)],
            risk="Deprecated candidates remain in the audit trail but should not guide new synthesis.",
            rollback="Create a fresh candidate from newer evidence and approve it through validation and human review.",
            metadata={"reason": reason.strip(), "claim": candidate["claim"]},
        )

    def _record_eval_release_gate(self, candidate_id: str, eval_gate: dict[str, Any] | None) -> dict[str, Any]:
        candidate = self.store.get_domain_knowledge_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"domain knowledge candidate not found: {candidate_id}")
        gate = eval_gate or {
            "passed": False,
            "reason": "evaluation report is required before knowledge release",
            "report_path": "",
        }
        passed = bool(gate.get("passed"))
        waived = bool(gate.get("waived"))
        gate_id = self.store.add_gate_record(
            gate_type="domain_knowledge_candidate_eval_release",
            subject_type="domain_knowledge_candidate",
            subject_id=candidate_id,
            result="waived" if waived else "passed" if passed else "failed",
            reason=str(
                gate.get("reason")
                or ("evaluation gate waived" if waived else "evaluation gate passed" if passed else "evaluation gate failed")
            ),
            evidence_refs=[
                ref
                for ref in (
                    str(gate.get("report_path") or ""),
                    *self._candidate_evidence_refs(candidate),
                )
                if ref
            ],
            metadata={"topic": candidate["topic"], "eval_gate": gate},
        )
        return {
            "passed": passed,
            "waived": waived,
            "gate_id": gate_id,
            "reason": str(gate.get("reason") or ""),
            "report_path": str(gate.get("report_path") or ""),
        }

    def _ensure_default_graphs(self) -> dict[str, object]:
        if self.graph_service.list_graphs():
            return {"seeded_graphs": 0, "graphs": self.graph_service.list_graphs()}
        return self.graph_service.seed_default_graphs()

    @staticmethod
    def _knowledge_layer(failures: list[str]) -> str:
        if not failures:
            return KNOWLEDGE_LAYER_VALIDATED
        if (
            "no_original_source" in failures
            or "unresolved_contradictions" in failures
            or any(failure.startswith("marketing_or_vague_content") for failure in failures)
        ):
            return KNOWLEDGE_LAYER_REJECTED
        return KNOWLEDGE_LAYER_WEAK_SIGNAL

    @staticmethod
    def _layer_metadata(layer: str) -> dict[str, str]:
        if layer == KNOWLEDGE_LAYER_VALIDATED:
            return {
                "intended_use": "stable planning context after eval and human review gates pass",
                "next_action": "eligible for knowledge-candidate-approve after eval gate passes",
            }
        if layer == KNOWLEDGE_LAYER_WEAK_SIGNAL:
            return {
                "intended_use": "exploratory clue for the next search plan; do not use as trusted knowledge",
                "next_action": "search for independent corroborating sources before promotion",
            }
        return {
            "intended_use": "audit trail only; do not use for planning or synthesis",
            "next_action": "ignore unless a human reviewer supplies corrected evidence",
        }

    def _latest_validation_layers(self) -> dict[str, dict[str, str]]:
        latest: dict[str, dict[str, str]] = {}
        for gate in self.store.list_gate_records(gate_type="domain_knowledge_candidate_validation"):
            candidate_id = str(gate["subject_id"])
            if candidate_id in latest:
                continue
            metadata = gate.get("metadata") or {}
            latest[candidate_id] = {
                "knowledge_layer": str(metadata.get("knowledge_layer") or KNOWLEDGE_LAYER_UNREVIEWED),
                "reason": str(gate.get("reason") or ""),
                "next_action": str(metadata.get("next_action") or ""),
            }
        return latest

    @staticmethod
    def _candidate_graph_query(candidate: dict[str, Any]) -> str:
        evidence_titles = " ".join(str(item.get("title", "")) for item in candidate.get("evidence", []))
        return " ".join(
            part.strip()
            for part in (
                str(candidate.get("topic", "")),
                str(candidate.get("claim", "")),
                str(candidate.get("applies_when", "")),
                evidence_titles,
            )
            if part and part.strip()
        )

    @staticmethod
    def _fingerprint(topic: str, claim: str) -> str:
        normalized = re.sub(r"\s+", " ", f"{topic.strip().lower()}\n{claim.strip().lower()}")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _find_similar_candidate(self, topic: str, claim: str) -> dict[str, Any] | None:
        """Find an existing candidate whose topic matches and whose claim is a
        rephrasing of ``claim``, so the cross-trajectory gate can fire for
        repeated tracking of the same topic even when wording shifts.

        Topic matching is literal-first (``_topic_similar``) with an embedding
        fallback (``_topic_semantically_similar``) so rephrased topic strings
        such as "AI Agent Harness 上下文工程 工具调用可靠性" vs "上下文工程"
        still merge; the embedding channel is a no-op when the provider is
        disabled or unavailable.
        """
        for candidate in self.store.list_domain_knowledge_candidates():
            if str(candidate["status"]) == "deprecated":
                continue
            if not (
                _topic_similar(topic, str(candidate.get("topic", "")))
                or self._topic_semantically_similar(topic, str(candidate.get("topic", "")))
            ):
                continue
            if _claims_similar(claim, str(candidate.get("claim", ""))):
                return candidate
        return None

    def _topic_semantically_similar(self, left: str, right: str) -> bool:
        """Embedding-based topic similarity, used when literal matching fails.

        Returns False when either topic is empty or the embedding provider
        returns empty vectors (disabled/unavailable), so the caller falls back
        to literal matching without raising.
        """
        if not left.strip() or not right.strip():
            return False
        try:
            vectors = self.embedding_provider.embed([left, right])
        except Exception:
            return False
        if len(vectors) != 2 or not vectors[0] or not vectors[1]:
            return False
        from search_assistant.knowledge_graph.embedding import cosine_similarity

        return cosine_similarity(vectors[0], vectors[1]) >= 0.75

    @staticmethod
    def _oldest_evidence_retrieved_at(candidate: dict[str, Any]):
        """Earliest evidence retrieval timestamp, or None when unparsable."""
        dates: list[datetime] = []
        for item in candidate.get("evidence") or []:
            raw = str(item.get("retrieved_at") or "").strip()
            if not raw:
                continue
            try:
                dates.append(datetime.fromisoformat(raw.replace("Z", "+00:00")))
            except ValueError:
                continue
        return min(dates) if dates else None

    @staticmethod
    def _confidence(evidence_count: int) -> str:
        if evidence_count >= 3:
            return "high"
        if evidence_count >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _candidate_evidence_refs(candidate: dict[str, Any]) -> list[str]:
        refs = [f"candidate:{candidate['id']}"]
        refs.extend(str(source_id) for source_id in candidate.get("source_ids", []))
        refs.extend(
            str(item.get("url"))
            for item in candidate.get("evidence", [])
            if str(item.get("url", "")).startswith(("http://", "https://"))
        )
        return list(dict.fromkeys(ref for ref in refs if ref))
