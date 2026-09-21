from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any

from search_assistant.contracts import AnswerPackage, Classification, IncomingMessage, SearchRecord, SourceEvidence
from search_assistant.memory.store import MemoryStore
from search_assistant.reports.service import ReportService
from search_assistant.search.provider import SearchClient
from search_assistant.skills.service import SkillDraftService
from search_assistant.workflow.grounding import (
    GroundingDecision,
    QuestionProfile,
    build_question_profile,
    classify_question_profile,
    decide_grounding_policy,
)
from search_assistant.verification.policy import (
    extract_key_claims,
    requires_calibration,
    requires_verification,
    verify_claims_against_sources,
)
from search_assistant.runtime import AgentRuntime


@dataclass(frozen=True)
class SearchAudit:
    executed: bool
    queries: list[str]
    sources: list[SourceEvidence]
    engines: list[str] = field(default_factory=list)
    skipped_reason: str | None = None


@dataclass(frozen=True)
class GroundingReport:
    answer: str
    unverified_claims: list[str] = field(default_factory=list)
    fallback_used: bool = False
    enforcement_actions: list[str] = field(default_factory=list)


class SearchAssistantWorkflow:
    def __init__(
        self,
        store: MemoryStore,
        runtime: AgentRuntime,
        search_client: SearchClient | None = None,
        search_budget_seconds: float = 45.0,
        skill_drafts_dir: str | Path = "skills/drafts",
        report_output_dir: str | Path = "reports",
        admin_user_ids: set[str] | None = None,
    ):
        self.store = store
        self.runtime = runtime
        self.search_client = search_client
        self.search_budget_seconds = search_budget_seconds
        self.skill_drafts_dir = Path(skill_drafts_dir)
        self.report_output_dir = Path(report_output_dir)
        self.admin_user_ids = admin_user_ids

    def answer(self, message: IncomingMessage) -> AnswerPackage:
        existing = self.store.latest_answer_for_dedupe_key(message.dedupe_key)
        if existing is None:
            existing = self.store.latest_answer_for_message_id(message.message_id)
        if existing is not None:
            return existing

        question_id = self.store.record_interaction(message)
        if self._is_skill_promote_request(message.text):
            if not self._can_manage_skills(message):
                return self._answer_skill_permission_denied(message, question_id)
            return self._answer_skill_promote_command(message, question_id)
        if self._is_skill_list_request(message.text):
            return self._answer_skill_list_command(message, question_id)
        if self._is_skill_generation_request(message.text):
            return self._answer_skill_generation_command(message, question_id)
        if self._is_candidate_confirm_request(message.text):
            if not self._can_manage_skills(message):
                return self._answer_skill_permission_denied(message, question_id)
            return self._answer_candidate_confirm_command(message, question_id)
        if self._is_candidate_pending_list_request(message.text):
            return self._answer_candidate_pending_list_command(message, question_id)
        if self._is_evolution_request(message.text):
            return self._answer_evolution_command(message, question_id)
        if self._is_learning_report_request(message.text):
            return self._answer_learning_report_command(message, question_id)
        question_text = message.text
        if self._is_user_feedback(message.text):
            embedded_question = self._extract_answerable_question_from_feedback(message.text)
            if embedded_question is None:
                return self._answer_feedback_command(message, question_id)
            question_text = embedded_question

        question_profile = build_question_profile(question_text)
        classification = self._classify(question_profile)
        grounding_decision = decide_grounding_policy(question_profile, classification)
        answer_strategy = self._answer_strategy(question_text, classification)
        memory_context = self._memory_context(message.user_id, message.chat_id)
        experience_context = self._experience_context(message.user_id, message.chat_id)
        active_skills = self._active_skill_context(message.user_id, message.chat_id)
        planning_context: dict[str, object] = {
            "question_id": question_id,
            "classification": classification,
            "question_profile": question_profile.to_context(),
            "grounding_policy": grounding_decision.to_context(),
            "answer_strategy": answer_strategy,
            "memory": memory_context,
            "experience": experience_context,
            "active_skills": active_skills,
        }
        search_audit = self._search_sources(question_text, classification, planning_context)
        search_results = search_audit.sources
        context: dict[str, object] = {
            "question_id": question_id,
            "question": question_text,
            "classification": classification,
            "question_profile": question_profile.to_context(),
            "grounding_policy": grounding_decision.to_context(),
            "answer_strategy": answer_strategy,
            "memory": memory_context,
            "experience": experience_context,
            "active_skills": active_skills,
            "search_queries": search_audit.queries,
            "search_results": [result.model_dump(mode="json") for result in search_results],
        }

        calibration: dict[str, object] | None = None
        draft: str | None = None
        low_relevance_issue = self._low_source_relevance_issue(
            question_text,
            classification,
            search_audit,
            grounding_decision,
        )
        allow_foundational_fallback = self._allow_foundational_fallback(grounding_decision, low_relevance_issue)
        fallback_used = False
        grounding_policy_claims: list[str] = []
        grounding_enforcement_actions: list[str] = []
        if low_relevance_issue and not allow_foundational_fallback:
            final_answer = self._low_source_relevance_answer(low_relevance_issue)
            review: dict[str, object] | None = {
                "ran": True,
                "approved": False,
                "issues": [low_relevance_issue],
                "revision": final_answer,
            }
            review_failed = True
            verified_claims = []
            unverified_claims = [low_relevance_issue]
        else:
            if grounding_decision.category == "foundational":
                context["foundational_answer_policy"] = (
                    "For stable concept or understanding-check questions, do not refuse solely because "
                    "live search evidence is thin. Give the basic conceptual answer, keep uncertainty visible, "
                    "and avoid current facts, exact numbers, vendor-specific claims, and unrelated citations "
                    "unless the search results support them."
                )
            if low_relevance_issue and allow_foundational_fallback:
                context["source_relevance_issue"] = low_relevance_issue
                context["allow_foundational_fallback"] = True
            draft = self.runtime.generate_answer(question_text, context)
            _, draft_unverified_claims = self._verify_answer_claims(draft, classification, search_results)
            if low_relevance_issue and allow_foundational_fallback and low_relevance_issue not in draft_unverified_claims:
                draft_unverified_claims.append(low_relevance_issue)
            if requires_calibration(classification, bool(draft_unverified_claims)):
                calibration = self.runtime.calibrate(
                    draft,
                    {
                        **context,
                        "draft": draft,
                        "unverified_claims": draft_unverified_claims,
                        "search_results": [result.model_dump(mode="json") for result in search_results],
                    },
                )

            final_answer = draft
            if calibration and isinstance(calibration.get("revision"), str) and calibration["revision"].strip():
                final_answer = calibration["revision"].strip()
            _, pre_review_unverified_claims = self._verify_answer_claims(final_answer, classification, search_results)
            review = self.runtime.review_answer(
                final_answer,
                {
                    **context,
                    "answer": final_answer,
                    "calibration": calibration,
                    "unverified_claims": pre_review_unverified_claims,
                    "search_results": [result.model_dump(mode="json") for result in search_results],
                },
            )
            review_failed = bool(review and review.get("ran") and not review.get("approved", False))
            if review_failed:
                final_answer = self._review_rejection_answer(review)
            elif review and isinstance(review.get("revision"), str) and review["revision"].strip():
                final_answer = review["revision"].strip()
            enforcement = self._enforce_grounding_policy(
                final_answer,
                question_text,
                grounding_decision,
                search_results,
                low_relevance_issue=low_relevance_issue,
                review_failed=review_failed,
            )
            final_answer = enforcement.answer
            fallback_used = fallback_used or enforcement.fallback_used
            grounding_policy_claims = enforcement.unverified_claims
            grounding_enforcement_actions = enforcement.enforcement_actions
            verified_claims, unverified_claims = self._verify_answer_claims(final_answer, classification, search_results)
            if low_relevance_issue and allow_foundational_fallback and low_relevance_issue not in unverified_claims:
                unverified_claims.append(low_relevance_issue)
            for policy_claim in grounding_policy_claims:
                if policy_claim not in unverified_claims:
                    unverified_claims.append(policy_claim)

        visible_answer = self._append_search_record(final_answer, search_audit)
        package = AnswerPackage(
            question_id=question_id,
            answer_text=visible_answer,
            classification=classification,
            confidence="low" if review_failed or fallback_used else self._confidence(classification, bool(unverified_claims)),
            verified_claims=verified_claims,
            unverified_claims=unverified_claims,
            sources=search_results,
            calibration=calibration,
            review=review,
            memory_updates=self._memory_updates(message, question_id),
            search_record=self._search_record_from_audit(search_audit),
            trajectory_context={
                "draft_answer": draft,
                "active_skills": [
                    {"name": skill.get("name", ""), "path": skill.get("path", "")}
                    for skill in active_skills
                ],
                "answer_strategy": answer_strategy,
                "question_profile": question_profile.to_context(),
                "grounding_policy": grounding_decision.to_context(),
                "grounding_report": {
                    "fallback_used": fallback_used,
                    "unverified_claims": grounding_policy_claims,
                    "enforcement_actions": grounding_enforcement_actions,
                },
                "runtime_metadata": {
                    "runtime_class": type(self.runtime).__name__,
                    "model": getattr(self.runtime, "model", None),
                    "provider": getattr(self.runtime, "provider", None),
                    "search_client_class": type(self.search_client).__name__ if self.search_client is not None else None,
                },
                "execution_flags": {
                    "review_failed": review_failed,
                    "fallback_used": fallback_used,
                    "low_relevance_issue": low_relevance_issue,
                    "grounding_category": grounding_decision.category,
                },
            },
        )
        self.store.record_answer(package)
        self.store.add_experience_item(
            title=f"Answer pattern: {package.classification}",
            body=self._experience_body(message, package),
            source_ids=[package.question_id],
            user_id=message.user_id,
            chat_id=message.chat_id,
        )
        if self._is_user_feedback(message.text):
            self.store.add_experience_item(
                title="User feedback: search and answer correction",
                body=self._feedback_experience_body(message, package, search_audit),
                source_ids=[package.question_id],
                user_id=message.user_id,
                chat_id=message.chat_id,
            )
        for update in package.memory_updates:
            if isinstance(update, dict):
                self.store.add_memory_item(
                    update["kind"], update["content"], question_id, user_id=message.user_id, chat_id=message.chat_id
                )
            else:
                self.store.add_memory_item(
                    update.kind, update.content, question_id, user_id=message.user_id, chat_id=message.chat_id
                )
        return package

    def _answer_learning_report_command(self, message: IncomingMessage, question_id: str) -> AnswerPackage:
        audit = SearchAudit(
            executed=False,
            queries=[],
            sources=[],
            engines=self._search_engine_names(),
            skipped_reason="learning report command does not trigger web search",
        )
        markdown = ReportService(self.store, output_dir=self.report_output_dir).generate_markdown()
        report_path = self.report_output_dir / "learning-report.md"
        final_answer = self._learning_report_acknowledgement_answer(markdown, report_path)
        package = AnswerPackage(
            question_id=question_id,
            answer_text=self._append_search_record(final_answer, audit),
            classification="simple",
            confidence="high",
            verified_claims=[],
            unverified_claims=[],
            sources=[],
            calibration=None,
            review={
                "ran": False,
                "approved": True,
                "issues": [],
                "revision": final_answer,
                "reason": "learning report command generated persisted report without ordinary QA generation",
            },
            memory_updates=[],
            search_record=self._search_record_from_audit(audit),
        )
        self.store.record_answer(package)
        self.store.add_experience_item(
            title="User requested learning report",
            body=(
                f"user_request={message.text}\n"
                f"report_path={report_path}\n"
                f"report_chars={len(markdown)}\n"
                "future_rule=User can explicitly request a persisted learning report from chat or Feishu."
            ),
            source_ids=[package.question_id],
            user_id=message.user_id,
            chat_id=message.chat_id,
        )
        return package

    def _answer_evolution_command(self, message: IncomingMessage, question_id: str) -> AnswerPackage:
        audit = SearchAudit(
            executed=False,
            queries=[],
            sources=[],
            engines=self._search_engine_names(),
            skipped_reason="self-evolution command does not trigger web search",
        )
        markdown = ReportService(self.store, output_dir=self.report_output_dir).generate_markdown()
        report_path = self.report_output_dir / "learning-report.md"
        skill_service = self._skill_service_for(message)
        refreshed_skill_paths = skill_service.refresh_reviewable_drafts()
        skill_paths = skill_service.auto_create_from_experience(refresh_existing=False)
        final_answer = self._evolution_acknowledgement_answer(
            report_path,
            markdown,
            skill_paths,
            refreshed_skill_paths,
        )
        package = AnswerPackage(
            question_id=question_id,
            answer_text=self._append_search_record(final_answer, audit),
            classification="simple",
            confidence="high",
            verified_claims=[],
            unverified_claims=[],
            sources=[],
            calibration=None,
            review={
                "ran": False,
                "approved": True,
                "issues": [],
                "revision": final_answer,
                "reason": "self-evolution command generated report and review-only skill drafts",
            },
            memory_updates=[],
            search_record=self._search_record_from_audit(audit),
        )
        self.store.record_answer(package)
        self.store.add_experience_item(
            title="User requested self-evolution",
            body=(
                f"user_request={message.text}\n"
                f"report_path={report_path}\n"
                f"report_chars={len(markdown)}\n"
                f"draft_count={len(skill_paths)}\n"
                f"refreshed_draft_count={len(refreshed_skill_paths)}\n"
                f"draft_paths={json.dumps(skill_paths, ensure_ascii=False)}\n"
                f"refreshed_draft_paths={json.dumps(refreshed_skill_paths, ensure_ascii=False)}\n"
                "future_rule=User can explicitly run the self-evolution cycle from chat or Feishu."
            ),
            source_ids=[package.question_id],
            user_id=message.user_id,
            chat_id=message.chat_id,
        )
        return package

    def _answer_feedback_command(self, message: IncomingMessage, question_id: str) -> AnswerPackage:
        audit = SearchAudit(
            executed=False,
            queries=[],
            sources=[],
            engines=self._search_engine_names(),
            skipped_reason="反馈/修正指令不触发联网搜索",
        )
        final_answer = self._feedback_acknowledgement_answer(message.text)
        package = AnswerPackage(
            question_id=question_id,
            answer_text=self._append_search_record(final_answer, audit),
            classification="simple",
            confidence="high",
            verified_claims=[],
            unverified_claims=[],
            sources=[],
            calibration=None,
            review={
                "ran": False,
                "approved": True,
                "issues": [],
                "revision": final_answer,
                "reason": "user feedback command stored without ordinary QA generation",
            },
            memory_updates=self._memory_updates(message, question_id),
            search_record=self._search_record_from_audit(audit),
        )
        self.store.record_answer(package)
        self.store.add_experience_item(
            title="User feedback: search and answer correction",
            body=self._feedback_experience_body(message, package, audit),
            source_ids=[package.question_id],
            user_id=message.user_id,
            chat_id=message.chat_id,
        )
        for update in package.memory_updates:
            if isinstance(update, dict):
                self.store.add_memory_item(
                    update["kind"], update["content"], question_id, user_id=message.user_id, chat_id=message.chat_id
                )
            else:
                self.store.add_memory_item(
                    update.kind, update.content, question_id, user_id=message.user_id, chat_id=message.chat_id
                )
        return package

    def _answer_skill_list_command(self, message: IncomingMessage, question_id: str) -> AnswerPackage:
        audit = SearchAudit(
            executed=False,
            queries=[],
            sources=[],
            engines=self._search_engine_names(),
            skipped_reason="skill list command does not trigger web search",
        )
        result = self._skill_service_for(message).list_review_status()
        final_answer = self._skill_list_answer(result)
        package = AnswerPackage(
            question_id=question_id,
            answer_text=self._append_search_record(final_answer, audit),
            classification="simple",
            confidence="high",
            verified_claims=[],
            unverified_claims=[],
            sources=[],
            calibration=None,
            review={
                "ran": False,
                "approved": True,
                "issues": [],
                "revision": final_answer,
                "reason": "user requested skill review status list",
            },
            memory_updates=[],
            search_record=self._search_record_from_audit(audit),
        )
        self.store.record_answer(package)
        self.store.add_experience_item(
            title="User requested skill list",
            body=(
                f"user_request={message.text}\n"
                f"skill_counts={json.dumps(result.get('counts', {}), ensure_ascii=False)}\n"
                "future_rule=User can inspect review-only and promoted skill status from chat."
            ),
            source_ids=[package.question_id],
            user_id=message.user_id,
            chat_id=message.chat_id,
        )
        return package

    def _answer_skill_promote_command(self, message: IncomingMessage, question_id: str) -> AnswerPackage:
        audit = SearchAudit(
            executed=False,
            queries=[],
            sources=[],
            engines=self._search_engine_names(),
            skipped_reason="skill promote command does not trigger web search",
        )
        target = self._skill_promote_target(message.text)
        promoted_path: str | None = None
        error: str | None = None
        if not target:
            error = "没有识别到要启用的 skill 名称或 slug。"
        else:
            try:
                promoted_path = self._skill_service_for(message).promote(target)
            except (KeyError, ValueError, FileExistsError, FileNotFoundError) as exc:
                error = str(exc)

        final_answer = self._skill_promote_answer(target, promoted_path, error)
        package = AnswerPackage(
            question_id=question_id,
            answer_text=self._append_search_record(final_answer, audit),
            classification="simple",
            confidence="high" if promoted_path else "low",
            verified_claims=[],
            unverified_claims=[] if promoted_path else [error or "skill promotion failed"],
            sources=[],
            calibration=None,
            review={
                "ran": False,
                "approved": bool(promoted_path),
                "issues": [] if promoted_path else [error or "skill promotion failed"],
                "revision": final_answer,
                "reason": "user requested skill promotion from chat",
            },
            memory_updates=[],
            search_record=self._search_record_from_audit(audit),
        )
        self.store.record_answer(package)
        self.store.add_experience_item(
            title="User requested skill promotion",
            body=(
                f"user_request={message.text}\n"
                f"target={target or ''}\n"
                f"promoted_path={promoted_path or ''}\n"
                f"error={error or ''}\n"
                "future_rule=Only explicitly promoted skills are loaded as active reviewed guidance."
            ),
            source_ids=[package.question_id],
            user_id=message.user_id,
            chat_id=message.chat_id,
        )
        return package

    def _answer_skill_generation_command(self, message: IncomingMessage, question_id: str) -> AnswerPackage:
        audit = SearchAudit(
            executed=False,
            queries=[],
            sources=[],
            engines=self._search_engine_names(),
            skipped_reason="skill generation command does not trigger web search",
        )
        title = self._skill_generation_title(message.text)
        source_ids = self._skill_source_ids_for_draft(message.user_id, message.chat_id)
        draft_path = self._skill_service_for(message).create_from_experience(
            title,
            source_ids=source_ids,
        )
        final_answer = self._skill_generation_acknowledgement_answer(title, draft_path, source_ids)
        package = AnswerPackage(
            question_id=question_id,
            answer_text=self._append_search_record(final_answer, audit),
            classification="simple",
            confidence="high",
            verified_claims=[],
            unverified_claims=[],
            sources=[],
            calibration=None,
            review={
                "ran": False,
                "approved": True,
                "issues": [],
                "revision": final_answer,
                "reason": "user requested review-only skill draft generation",
            },
            memory_updates=[],
            search_record=self._search_record_from_audit(audit),
        )
        self.store.record_answer(package)
        self.store.add_experience_item(
            title="User requested skill draft",
            body=(
                f"user_request={message.text}\n"
                f"skill_title={title}\n"
                f"draft_path={draft_path}\n"
                f"source_ids={json.dumps(source_ids, ensure_ascii=False)}\n"
                "future_rule=User can explicitly request review-only skill draft generation from chat."
            ),
            source_ids=[package.question_id],
            user_id=message.user_id,
            chat_id=message.chat_id,
        )
        return package

    def _classify(self, profile: QuestionProfile) -> Classification:
        return classify_question_profile(profile)

    def _answer_strategy(self, text: str, classification: Classification) -> dict[str, object]:
        lowered = text.lower()
        base: dict[str, object] = {
            "mode": "direct_answer",
            "style": "concise_verified_partner",
            "stance": "answer_first_and_evidence_grounded",
            "visible_reasoning": [
                "direct_judgment",
                "reasoning_snapshot",
                "evidence_basis",
                "uncertainty",
                "next_step",
            ],
            "reasoning_contract": "must_show_compact_user_visible_reasoning",
            "presentation_contract": "natural_technical_partner_not_mechanical_checklist",
            "section_policy": "required_sections_are_reasoning_moves_not_mandatory_headings",
            "preferred_response_shape": "answer_first_then_reasoning_snapshot_then_next_check",
            "required_sections": [
                "direct_judgment",
                "evidence_basis",
                "uncertainty",
                "next_step",
            ],
            "minimum_reasoning_detail": "state_evidence_and_uncertainty_before_final_recommendation",
            "hidden_reasoning_policy": "do_not_reveal_chain_of_thought",
            "voice_guidance": [
                "answer like a collaborative technical partner",
                "show what you checked and why the conclusion follows",
                "push back on weak premises without sounding bureaucratic",
            ],
            "max_clarifying_questions": 1,
        }
        if self._is_brainstorming_discussion(text, lowered):
            return {
                **base,
                "mode": "brainstorming_discussion",
                "style": "technical_partner",
                "stance": "exploratory_but_evidence_grounded",
                "visible_reasoning": [
                    "direct_judgment",
                    "reasoning_snapshot",
                    "competing_hypotheses",
                    "evidence_check",
                    "pushback",
                    "decision_criteria",
                    "next_verification",
                ],
                "required_sections": [
                    "direct_judgment",
                    "competing_hypotheses",
                    "evidence_check",
                    "pushback",
                    "decision_criteria",
                    "next_verification",
                ],
                "preferred_response_shape": "judgment_then_hypotheses_then_collaborative_pushback",
                "minimum_reasoning_detail": "compare_hypotheses_and_explain_tradeoffs_before_recommendation",
            }
        if self._is_technical_explanation(text, lowered, classification):
            return {
                **base,
                "mode": "technical_explanation",
                "style": "technical_partner",
                "stance": "explain_mechanism_with_assumptions",
                "visible_reasoning": [
                    "direct_judgment",
                    "reasoning_snapshot",
                    "key_assumptions",
                    "evidence_check",
                    "mechanistic_trace",
                    "reasoning_path",
                    "counterpoints_or_risks",
                    "next_verification",
                ],
                "required_sections": [
                    "direct_judgment",
                    "key_assumptions",
                    "evidence_check",
                    "reasoning_path",
                    "counterpoints_or_risks",
                    "next_verification",
                ],
                "minimum_reasoning_detail": "explain_mechanism_or_calculation_before_final_recommendation",
            }
        return base

    def _active_skill_context(
        self,
        user_id: str,
        chat_id: str,
        max_skills: int = 5,
        max_chars_per_skill: int = 5000,
    ) -> list[dict[str, str]]:
        active_skills: list[dict[str, str]] = []
        for draft in self.store.list_skill_drafts(user_id, chat_id):
            if draft.get("review_status") != "promoted":
                continue
            path = Path(str(draft.get("path", "")))
            if path.name != "SKILL.md" or not path.exists():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            active_skills.append(
                {
                    "name": str(draft.get("name", "")),
                    "path": str(path),
                    "content": content[:max_chars_per_skill],
                }
            )
            if len(active_skills) >= max_skills:
                break
        return active_skills

    def _memory_context(
        self,
        user_id: str,
        chat_id: str,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        """Return layered memory items for planning, falling back to legacy tables."""
        items = self.store.list_layered_memory_items(
            layer=None, user_id=user_id, chat_id=chat_id, limit=limit
        )
        if items:
            return [item.model_dump(mode="json") for item in items]
        return self.store.list_memory_items(user_id, chat_id)

    def _experience_context(
        self,
        user_id: str,
        chat_id: str,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        """Return run_experience layer items, falling back to legacy experience tables."""
        items = self.store.list_layered_memory_items(
            layer="run_experience", user_id=user_id, chat_id=chat_id, limit=limit
        )
        if items:
            return [item.model_dump(mode="json") for item in items]
        return self.store.list_experience_items(user_id, chat_id)

    def _can_manage_skills(self, message: IncomingMessage) -> bool:
        return self.admin_user_ids is None or message.user_id in self.admin_user_ids

    def _skill_service_for(self, message: IncomingMessage) -> SkillDraftService:
        return SkillDraftService(
            self.store,
            drafts_dir=self.skill_drafts_dir,
            user_id=message.user_id,
            chat_id=message.chat_id,
        )

    def _answer_skill_permission_denied(self, message: IncomingMessage, question_id: str) -> AnswerPackage:
        final_answer = "当前账号没有启用 Skill 的权限。Skill 变更需要由配置中的管理员审核并执行。"
        audit = SearchAudit(
            executed=False,
            queries=[],
            sources=[],
            engines=self._search_engine_names(),
            skipped_reason="skill promotion requires administrator authorization",
        )
        package = AnswerPackage(
            question_id=question_id,
            answer_text=self._append_search_record(final_answer, audit),
            classification="simple",
            confidence="low",
            verified_claims=[],
            unverified_claims=["skill promotion was denied for this user"],
            sources=[],
            calibration=None,
            review={"ran": False, "approved": False, "issues": ["unauthorized skill promotion"], "revision": final_answer},
            memory_updates=[],
            search_record=self._search_record_from_audit(audit),
        )
        self.store.record_answer(package)
        self.store.add_experience_item(
            title="Denied skill promotion request",
            body=f"user_id={message.user_id}\nrequest={message.text}",
            source_ids=[question_id],
            user_id=message.user_id,
            chat_id=message.chat_id,
        )
        return package

    def _is_brainstorming_discussion(self, text: str, lowered: str) -> bool:
        markers = (
            "brainstorm",
            "debate",
            "discuss",
            "push back",
            "challenge",
            "hypothesis",
            "assumption",
            "我觉得",
            "我现在觉得",
            "你认为",
            "你怎么看",
            "怎么看",
            "有没有可能",
            "能不能这样理解",
            "讨论",
            "聊一下",
            "头脑风暴",
            "争辩",
            "反驳",
            "假设",
            "脑暴",
        )
        return any(marker in lowered or marker in text for marker in markers)

    def _is_technical_explanation(self, text: str, lowered: str, classification: Classification) -> bool:
        if classification not in {"hard", "research"}:
            return False
        mechanism_signals = (
            "why",
            "how does",
            "mechanism",
            "principle",
            "bottleneck",
            "bandwidth",
            "throughput",
            "tok/s",
            "tokens per second",
            "decode",
            "prefill",
            "为什么",
            "原理",
            "机制",
            "瓶颈",
            "带宽",
            "吞吐",
            "怎么",
            "如何",
            "哪里",
            "解码",
            "推理",
        )
        technical_terms = (
            "gpu",
            "cxl",
            "hbm",
            "kv cache",
            "tensor core",
            "model",
            "模型",
            "显存",
            "内存",
            "算力",
        )
        has_mechanism_signal = any(signal in lowered or signal in text for signal in mechanism_signals)
        has_technical_subject = any(term in lowered or term in text for term in technical_terms)
        return has_mechanism_signal and (has_technical_subject or self._has_specific_technical_entity(text))

    def _has_specific_technical_entity(self, text: str) -> bool:
        return bool(
            re.search(r"(?<![A-Za-z0-9])[A-Z]{1,8}\d{2,}[A-Za-z0-9-]*(?![A-Za-z0-9])", text)
            or re.search(r"(?<![A-Za-z0-9])[A-Z]{2,}\s+[A-Z]?\d{2,}[A-Za-z0-9-]*(?![A-Za-z0-9])", text)
            or re.search(
                r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]+[-\s]?v\d+[A-Za-z0-9-]*(?![A-Za-z0-9])",
                text,
                flags=re.IGNORECASE,
            )
            or re.search(r"(?<![A-Za-z0-9])[A-Za-z]+-\d+[A-Za-z0-9-]*(?![A-Za-z0-9])", text)
        )

    def _confidence(self, classification: Classification, has_unverified_claims: bool) -> str:
        if classification == "high_stakes" or has_unverified_claims:
            return "low"
        if classification in {"hard", "research"}:
            return "medium"
        return "high"

    def _verify_answer_claims(
        self,
        answer_text: str,
        classification: Classification,
        sources: list[SourceEvidence],
    ):
        claims = extract_key_claims(answer_text)
        if not requires_verification(answer_text, classification):
            return [], []
        return verify_claims_against_sources(claims, sources)

    def _memory_updates(self, message: IncomingMessage, question_id: str) -> list[dict[str, str]]:
        lowered = message.text.lower()
        updates: list[dict[str, str]] = []
        if any(word in lowered for word in ("feishu", "飞书", "bot", "机器人")):
            updates.append({"kind": "topic", "content": "Feishu integration", "source_id": question_id})
        if "agent framework" in lowered:
            updates.append({"kind": "topic", "content": "Microsoft Agent Framework", "source_id": question_id})
        if any(word in lowered for word in ("学习", "learning", "路线", "方向", "下一步", "next step")):
            updates.append({"kind": "intent", "content": "Learning direction planning", "source_id": question_id})
        if any(word in lowered for word in ("verify", "verification", "核验", "验证", "证据", "source")):
            updates.append({"kind": "practice", "content": "Verification practice", "source_id": question_id})
        if any(word in lowered for word in ("api", "version", "版本")):
            updates.append({"kind": "topic", "content": "API reliability", "source_id": question_id})
        if (
            "model deployment" in lowered
            or ("deploy" in lowered and "model" in lowered)
            or ("部署" in lowered and "模型" in message.text)
        ):
            updates.append(
                {
                    "kind": "topic",
                    "content": "AI infrastructure and model deployment",
                    "source_id": question_id,
                }
            )
        if any(word in lowered for word in ("cxl", "memory pooling", "memory pool", "gpu", "accelerator")):
            updates.append({"kind": "topic", "content": "AI memory architecture", "source_id": question_id})
        if any(
            word in lowered
            for word in ("world model", "physical ai", "embodied ai", "embodied physical ai", "gr00t", "genie 3")
        ):
            updates.append({"kind": "topic", "content": "Physical AI and world models", "source_id": question_id})
        return updates

    def _experience_body(self, message: IncomingMessage, package: AnswerPackage) -> str:
        return (
            f"question={message.text}\n"
            f"classification={package.classification}\n"
            f"confidence={package.confidence}\n"
            f"sources={len(package.sources)}\n"
            f"verified_claims={len(package.verified_claims)}\n"
            f"unverified_claims={len(package.unverified_claims)}\n"
            f"calibration_ran={bool((package.calibration or {}).get('ran'))}\n"
            f"review_ran={bool((package.review or {}).get('ran'))}\n"
            f"review_approved={bool((package.review or {}).get('approved'))}"
        )

    def _review_rejection_answer(self, review: dict[str, object]) -> str:
        issues = review.get("issues") if isinstance(review, dict) else []
        if not isinstance(issues, list):
            issues = [str(issues)]
        clean_issues = [str(issue).strip() for issue in issues if str(issue).strip()]
        lines = [
            "最终审查未通过：这轮回答不能作为可靠结论直接发送。",
            "",
            "主要原因:",
        ]
        if clean_issues:
            lines.extend(f"- {issue}" for issue in clean_issues)
        else:
            lines.append("- 审查 agent 未给出可验证通过的结论。")
        lines.extend(
            [
                "",
                "处理结果: 已阻断原回答。请补充更相关的搜索结果或改写问题后重新提问；在证据不足前，我不会用模型记忆硬给确定结论。",
            ]
        )
        return "\n".join(lines)

    def _foundational_fallback_answer(self, question: str, issue: str | None) -> str:
        focus = self._concept_focus_phrase(question)
        lines = [
            "可以这么理解，但要保留低置信边界：这个问题不应该只因为搜索证据弱就被阻断。",
            "",
            "基础解释:",
            f"- 先把问题聚焦在“{focus}”这个概念或理解关系上，而不是扩展成未经核实的事实结论。",
            "- 可以先区分主体、组成部分、工作方式、相互关系和边界条件，再说明哪些部分只是稳定概念，哪些部分需要来源支持。",
            "- 只要问题不涉及最新事实、具体数字、厂商规格、版本事实或高风险决策，就允许使用稳定模型知识给出概念层面的解释。",
            "- 如果问题后续转向参数、性能、发布时间、兼容性或部署可行性，就必须重新检索并核验证据。",
            "",
            "低置信说明: 本轮联网搜索证据弱，以上只作为基础解释；具体事实、参数、版本、性能和部署结论仍需重新检索并核验。",
        ]
        if issue:
            lines.extend(["", f"搜索校准: {issue}"])
        return "\n".join(lines)

    def _concept_focus_phrase(self, question: str) -> str:
        cleaned = question.strip()
        cleaned = re.sub(r"^\s*(?:请|请问|麻烦|帮我)?(?:能否|能不能|可不可以)?(?:解释|介绍|说明)(?:一下)?[:：]?\s*", "", cleaned)
        cleaned = re.sub(r"^\s*(?:what\s+is|can\s+you\s+explain|explain)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[？?。!！]+$", "", cleaned).strip()
        if not cleaned:
            return "问题中的核心概念"
        return cleaned[:80]

    def _remove_external_knowledge_sentences(self, answer: str) -> str:
        marker_pattern = (
            r"来自外部知识|外部知识|from external knowledge|outside (?:the )?retrieved sources|"
            r"model memory|模型记忆|行业常识|基于行业|逻辑推断|logical inference|"
            r"industry knowledge|industry common sense|not in search results|搜索结果中没有"
        )
        marker_re = re.compile(marker_pattern, flags=re.IGNORECASE)

        def keep_line(line: str) -> bool:
            stripped = line.strip()
            if not marker_re.search(stripped):
                return True
            if stripped.startswith("#"):
                return False
            has_sentence_punctuation = bool(re.search(r"[。！？.!?]", stripped))
            return has_sentence_punctuation or len(stripped) >= 160

        answer = "\n".join(line for line in answer.splitlines() if keep_line(line))
        sentence_pattern = re.compile(
            rf"[^。！？.!?\n]*(?:{marker_pattern})[^。！？.!?\n]*(?:[。！？.!?]|$)",
            flags=re.IGNORECASE,
        )
        cleaned = sentence_pattern.sub("", answer)
        cleaned = re.sub(r"[ \t]+([。！？.!?])", r"\1", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _remove_malformed_partial_lines(self, answer: str) -> str:
        cleaned_lines: list[str] = []
        for raw_line in answer.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append(line)
                continue
            if re.fullmatch(r"\d+[.)]", stripped):
                continue
            if self._looks_like_malformed_partial_line(stripped):
                continue
            cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _looks_like_malformed_partial_line(self, line: str) -> bool:
        if line.count("（") > line.count("）") or line.count("(") > line.count(")"):
            return True
        return bool(re.search(r"(?:^|[\s，,（(])(?:[A-Z]{2,}|[A-Z][A-Za-z]+)\s*\d+(?:\.\d*)?\.$", line))

    def _is_distributed_model_understanding_question(self, question: str) -> bool:
        lowered = question.lower()
        has_distributed = any(term in lowered or term in question for term in ("distributed", "分布式", "多节点"))
        has_model = any(term in lowered or term in question for term in ("llm", "large model", "大模型", "模型"))
        has_inference = any(term in lowered or term in question for term in ("inference", "推理", "节点", "互联"))
        return has_distributed and has_model and has_inference

    def _looks_like_source_insufficiency_refusal(self, answer: str) -> bool:
        lowered = answer.lower()
        source_terms = (
            "搜索结果相关性不足",
            "搜索内容不足",
            "搜索证据不足",
            "搜索未返回可用结果",
            "检索结果不足",
            "检索内容不足",
            "检索证据不足",
            "当前检索结果",
            "检索结果",
            "来源不足",
            "证据不足",
            "证据不充分",
            "证据不够充分",
            "证据有限",
            "资料不足",
            "资料不充分",
            "信息不足",
            "信息不充分",
            "搜索内容有限",
            "搜索结果有限",
            "检索内容有限",
            "检索结果有限",
            "来源有限",
            "source relevance",
            "insufficient source",
            "insufficient evidence",
            "insufficient retrieval",
            "limited evidence",
            "weak evidence",
        )
        refusal_terms = (
            "不能回答",
            "无法回答",
            "不应该回答",
            "不能给出",
            "无法给出",
            "无法确认",
            "不能确认",
            "无法判断",
            "不能判断",
            "无法确定",
            "不能确定",
            "难以判断",
            "暂时不能判断",
            "暂时无法判断",
            "不建议直接判断",
            "不建议直接给出结论",
            "已阻断",
            "阻断",
            "refuse",
            "cannot answer",
            "can't answer",
            "blocked",
            "cannot confirm",
            "cannot determine",
        )
        has_source_issue = any(term in lowered or term in answer for term in source_terms)
        has_refusal = any(term in lowered or term in answer for term in refusal_terms)
        return has_source_issue and has_refusal

    def _needs_foundational_policy_fallback(self, question: str, answer: str) -> bool:
        if self._looks_like_source_insufficiency_refusal(answer):
            return True
        if not self._looks_like_source_insufficiency_stall(answer):
            return False
        if self._looks_like_source_insufficiency_deferral(answer):
            return True
        return not self._has_foundational_concept_content(question, answer)

    def _looks_like_source_insufficiency_stall(self, answer: str) -> bool:
        lowered = answer.lower()
        source_terms = (
            "搜索结果相关性不足",
            "搜索内容不足",
            "搜索证据不足",
            "搜索未返回可用结果",
            "检索结果不足",
            "检索内容不足",
            "检索证据不足",
            "当前检索结果",
            "来源不足",
            "证据不足",
            "证据不充分",
            "证据不够充分",
            "证据有限",
            "资料不足",
            "资料不充分",
            "信息不足",
            "信息不充分",
            "搜索内容有限",
            "搜索结果有限",
            "检索内容有限",
            "检索结果有限",
            "来源有限",
            "source relevance",
            "insufficient source",
            "insufficient evidence",
            "insufficient retrieval",
            "limited evidence",
            "weak evidence",
        )
        stall_terms = (
            "建议",
            "补充",
            "重新",
            "进一步",
            "更多",
            "后再",
            "无法",
            "不能",
            "不足",
            "不充分",
            "有限",
            "暂时",
            "暂不",
            "缺少",
            "缺乏",
            "recommend",
            "need more",
            "more relevant",
            "further search",
            "not enough",
            "lack",
            "missing",
        )
        has_source_issue = any(term in lowered or term in answer for term in source_terms)
        has_stall = any(term in lowered or term in answer for term in stall_terms)
        return has_source_issue and has_stall

    def _looks_like_source_insufficiency_deferral(self, answer: str) -> bool:
        lowered = answer.lower()
        deferral_patterns = (
            r"不建议(?:直接)?(?:判断|给出判断|下结论|给出结论|回答)",
            r"暂(?:时)?(?:不|不能|无法)(?:判断|给出判断|下结论|给出结论|回答)",
            r"(?:建议)?[^。！？.!?\n]{0,40}(?:补充|重新|进一步|更多)[^。！？.!?\n]{0,40}后再(?:判断|回答|确认|下结论)",
            r"need\s+more[^.?!\n]{0,80}before\s+(?:answering|judging|concluding)",
            r"answer[^.?!\n]{0,40}after\s+(?:more|additional|further)\s+(?:sources|evidence|search)",
        )
        return any(re.search(pattern, lowered) or re.search(pattern, answer) for pattern in deferral_patterns)

    def _has_foundational_concept_content(self, question: str, answer: str) -> bool:
        if self._is_distributed_model_understanding_question(question):
            lowered = answer.lower()
            concept_terms = (
                "节点",
                "推理",
                "互联",
                "通信",
                "协同",
                "分工",
                "并行",
                "张量",
                "流水线",
                "专家",
                "node",
                "inference",
                "interconnect",
                "communication",
                "parallel",
                "pipeline",
                "tensor",
                "expert",
            )
            matched = {term for term in concept_terms if term in lowered or term in answer}
            return len(matched) >= 3
        return len(answer.strip()) >= 120

    def _enforce_grounding_policy(
        self,
        answer: str,
        question: str,
        decision: GroundingDecision,
        sources: list[SourceEvidence],
        low_relevance_issue: str | None = None,
        review_failed: bool = False,
    ) -> GroundingReport:
        actions: list[str] = []
        if decision.category == "foundational" and (
            review_failed or self._needs_foundational_policy_fallback(question, answer)
        ):
            actions.append("foundational_policy_fallback")
            return GroundingReport(
                answer=self._foundational_fallback_answer(question, low_relevance_issue),
                fallback_used=True,
                enforcement_actions=actions,
            )

        sanitized = self._remove_external_knowledge_sentences(answer)
        if sanitized != answer:
            actions.append("removed_external_knowledge_sentences")
        removed_claims: list[str] = []
        if decision.precise_numbers_require_sources:
            sanitized, removed_claims = self._sanitize_unsupported_precise_values(sanitized, sources)
            if removed_claims:
                actions.append("sanitized_unsupported_precise_values")
        if self._violates_evidence_required_policy(question, sanitized, decision, sources):
            issue = "Unsupported evidence-required answer replaced: missing required retrieval evidence."
            actions.append("evidence_required_gap_replacement")
            return GroundingReport(
                answer=self._evidence_required_gap_answer(decision, sources),
                unverified_claims=[*removed_claims, issue],
                fallback_used=True,
                enforcement_actions=actions,
            )
        cleaned = self._remove_malformed_partial_lines(sanitized)
        if cleaned != sanitized:
            actions.append("removed_malformed_partial_lines")
        return GroundingReport(answer=cleaned, unverified_claims=removed_claims, enforcement_actions=actions)

    def _sanitize_unsupported_precise_values(
        self,
        answer: str,
        sources: list[SourceEvidence],
    ) -> tuple[str, list[str]]:
        source_text = " ".join(f"{source.title} {source.snippet}" for source in sources)
        source_compact = self._compact_spec_text(source_text)
        spec_pattern = re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:GB|TB|MB)\s*(?:/s|ps)?"
            r"(?:\s*(?:HBM3E?|HBM2E?|HBM|统一内存|内存|显存|memory|bandwidth|带宽))?"
            r"|\b\d+(?:\.\d+)?\s*(?:tok/s|tokens/s|token/s)\b",
            flags=re.IGNORECASE,
        )
        example_estimate_pattern = re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?)\b"
            r"|\b\d+(?:\.\d+)?\s*毫秒\b"
            r"|\b\d+(?:\.\d+)?\s*B\b(?=[\s\w\u4e00-\u9fff-]{0,24}(?:FP\d+|模型|model|parameters?|参数))",
            flags=re.IGNORECASE,
        )
        removed: list[str] = []
        seen: set[str] = set()

        def remember(spec: str, label: str = "hardware spec number") -> None:
            normalized = re.sub(r"\s+", " ", spec).strip()
            if not normalized:
                return
            key = f"{label}:{normalized.lower()}"
            if key in seen:
                return
            seen.add(key)
            removed.append(f"Removed unsupported {label}: {normalized}")

        def supported_variants(spec: str) -> set[str]:
            canonical = self._canonical_hardware_spec_number(spec)
            variants = {canonical}
            if canonical and canonical.endswith("/s"):
                variants.add(canonical[:-2] + "ps")
            latency = re.search(r"(\d+(?:\.\d+)?)\s*(ms|milliseconds?|毫秒)", spec, flags=re.IGNORECASE)
            if latency:
                variants.add(f"{latency.group(1)}ms")
                variants.add(f"{latency.group(1)}毫秒")
            model_params = re.search(r"(\d+(?:\.\d+)?)\s*B\b", spec, flags=re.IGNORECASE)
            if model_params:
                variants.add(f"{model_params.group(1)}b")
            return {variant for variant in variants if variant}

        def is_supported(spec: str) -> bool:
            variants = supported_variants(spec)
            if not variants:
                return True
            return any(variant in source_compact for variant in variants)

        def unsupported_specs(text: str) -> list[str]:
            specs: list[str] = []
            for match in spec_pattern.finditer(text):
                spec = match.group(0).strip()
                if not is_supported(spec):
                    specs.append(spec)
            for match in example_estimate_pattern.finditer(text):
                spec = match.group(0).strip()
                if not is_supported(spec):
                    specs.append(spec)
            return specs

        def strip_parenthetical(match: re.Match[str]) -> str:
            segment = match.group(0)
            unsupported = unsupported_specs(segment)
            if not unsupported:
                return segment
            for spec in unsupported:
                label = "hardware example estimate" if example_estimate_pattern.fullmatch(spec) else "hardware spec number"
                remember(spec, label)
            return ""

        sanitized = re.sub(r"[（(][^（）()\n]{0,80}\d[^（）()\n]{0,80}[）)]", strip_parenthetical, answer)

        def replace_spec(match: re.Match[str]) -> str:
            spec = match.group(0).strip()
            if is_supported(spec):
                return match.group(0)
            remember(spec)
            return "具体规格数值（待来源核实）"

        sanitized = spec_pattern.sub(replace_spec, sanitized)

        def replace_example_estimate(match: re.Match[str]) -> str:
            spec = match.group(0).strip()
            if is_supported(spec):
                return match.group(0)
            remember(spec, "hardware example estimate")
            return "示例估算数值（待来源核实）"

        sanitized = example_estimate_pattern.sub(replace_example_estimate, sanitized)
        sanitized = re.sub(r"\s+([，。；：、,.!?])", r"\1", sanitized)
        sanitized = re.sub(r"（\s*）|\(\s*\)", "", sanitized)
        sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
        return sanitized, removed

    def _sanitize_unsupported_hardware_spec_numbers(
        self,
        answer: str,
        sources: list[SourceEvidence],
    ) -> tuple[str, list[str]]:
        # Legacy compatibility wrapper. The active workflow calls
        # _sanitize_unsupported_precise_values through Grounding Policy; do not
        # add new rules here. Remove once external callers have migrated.
        return self._sanitize_unsupported_precise_values(answer, sources)

    def _violates_evidence_required_policy(
        self,
        question: str,
        answer: str,
        decision: GroundingDecision,
        sources: list[SourceEvidence],
    ) -> bool:
        if decision.category not in {"evidence_required", "high_stakes"}:
            return False
        if not decision.require_retrieval_sources:
            return False
        if not sources:
            return self._contains_evidence_required_positive_claim(answer)
        if "deployment_feasibility" not in decision.required_evidence_families:
            return False
        if self._has_deployment_feasibility_evidence(question, decision, sources):
            return False
        return self._contains_evidence_required_positive_claim(answer)

    def _has_deployment_feasibility_evidence(
        self,
        question: str,
        decision: GroundingDecision,
        sources: list[SourceEvidence],
    ) -> bool:
        model_entity = self._requested_model_entity(question, decision)
        for source in sources:
            if self._source_is_query_echo_without_page_content(source):
                continue
            source_text = f"{source.title} {source.snippet} {source.url}"
            if model_entity and self._compact_spec_text(model_entity) not in self._compact_spec_text(source_text):
                continue
            if self._has_model_parameter_or_benchmark_evidence(source_text):
                return True
        return False

    def _requested_model_entity(self, question: str, decision: GroundingDecision) -> str | None:
        lowered_question = question.lower()
        for entity in decision.specific_entities:
            escaped = re.escape(entity)
            if re.search(
                rf"(?:model|模型|参数).{{0,40}}{escaped}|{escaped}.{{0,40}}(?:model|模型|参数)",
                question,
                flags=re.IGNORECASE,
            ):
                return entity
        if ("model" in lowered_question or "模型" in question) and decision.specific_entities:
            return decision.specific_entities[-1]
        return None

    def _has_model_parameter_or_benchmark_evidence(self, text: str) -> bool:
        lowered = text.lower()
        negative_patterns = (
            r"does not disclose.{0,80}(?:parameter|benchmark|throughput)",
            r"not disclose.{0,80}(?:parameter|benchmark|throughput)",
            r"no .{0,40}(?:parameter|benchmark|throughput|performance)",
            r"未.{0,20}(?:公开|确认|披露).{0,40}(?:参数|benchmark|基准|吞吐|性能)",
            r"没有.{0,40}(?:参数|benchmark|基准|吞吐|性能)",
        )
        if any(re.search(pattern, lowered) or re.search(pattern, text) for pattern in negative_patterns):
            return False
        if re.search(
            r"\b\d+(?:\.\d+)?\s*(?:b|t)\b.{0,80}(?:total parameters|parameters|params|activated|active parameters|experts)",
            lowered,
        ):
            return True
        if re.search(
            r"(?:total parameters|parameters|params|activated|active parameters|experts).{0,80}\b\d+(?:\.\d+)?\s*(?:b|t)\b",
            lowered,
        ):
            return True
        if re.search(r"(?:num_hidden_layers|hidden_size|num_attention_heads|num_experts|moe_intermediate_size)", lowered):
            return True
        return bool(
            re.search(r"\b\d+(?:\.\d+)?\s*(?:tok/s|tokens/s|token/s|ms)\b", lowered)
            or re.search(r"(?:benchmark|throughput|latency|performance).{0,80}\b\d+(?:\.\d+)?", lowered)
            or re.search(r"(?:基准|吞吐|延迟|性能).{0,40}\d+(?:\.\d+)?", text)
        )

    def _source_is_query_echo_without_page_content(self, source: SourceEvidence) -> bool:
        snippet = source.snippet.lower()
        return "direct source selected for query" in snippet and "page content fetch was unavailable" in snippet

    def _contains_evidence_required_positive_claim(self, answer: str) -> bool:
        positive_markers = (
            "can probably fit",
            "can fit",
            "could fit",
            "can deploy",
            "could deploy",
            "is deployable",
            "will run",
            "few tok/s",
            "tok/s",
            "tokens/s",
            "single-stream decode",
            "throughput",
            "is enough",
            "sufficient",
            "feasible",
            "can confirm",
            "confirmed",
            "有可能装下",
            "有可能在内存",
            "可能装下",
            "内存上可能",
            "权重肯定",
            "肯定放得下",
            "肯定装得下",
            "肯定能装下",
            "确实小于",
            "可以装下",
            "能装下",
            "可以部署",
            "能够部署",
            "能部署",
            "可以运行",
            "能够运行",
            "推理吞吐",
            "足够",
            "可行",
            "确认",
        )
        negated_claim_markers = (
            "not to claim",
            "without claiming",
            "do not claim",
            "don't claim",
            "not claim",
            "不要声称",
            "不能声称",
            "不应声称",
            "不要宣称",
            "不能宣称",
            "不应宣称",
            "cannot confirm",
            "cannot determine",
            "无法确认",
            "不能确认",
            "无法判断",
        )
        segments = [segment.strip() for segment in re.split(r"[\n。；;.!?]+", answer) if segment.strip()]
        for segment in segments:
            lowered_segment = segment.lower()
            if not any(marker in lowered_segment or marker in segment for marker in positive_markers):
                continue
            if any(marker in lowered_segment or marker in segment for marker in negated_claim_markers):
                continue
            return True
        return False

    def _evidence_required_gap_answer(
        self,
        decision: GroundingDecision,
        sources: list[SourceEvidence],
    ) -> str:
        relevant_sources = sources[:5]
        missing_items = self._missing_evidence_items(decision)
        lines = [
            "I cannot confirm the requested evidence-required conclusion from the retrieved evidence.",
            "",
            "What the search evidence supports:",
        ]
        if relevant_sources:
            for source in relevant_sources:
                lines.append(f"- {source.title}: {source.url}")
        else:
            lines.append("- No usable source was returned by the live search step.")
        lines.extend(
            [
                "",
                "What is still missing:",
                *missing_items,
                "",
                "Safe conclusion:",
                "- The answer should stay at low confidence until the missing source-backed facts are found.",
                "- It is fine to use retrieved facts as constraints, but not to state the requested conclusion more strongly than the evidence allows.",
            ]
        )
        if decision.specific_entities:
            lines.extend(["", f"Policy scope: {', '.join(decision.specific_entities)}"])
        return "\n".join(lines)

    def _missing_evidence_items(self, decision: GroundingDecision) -> list[str]:
        families = set(decision.required_evidence_families)
        items: list[str] = []
        if "current_fact" in families:
            items.append("- A current authoritative source for the requested fact.")
        if "exact_values" in families:
            items.append("- Source-backed exact values or specifications for the requested numeric claim.")
        if "vendor_or_version_claims" in families:
            items.append("- Vendor, project, release, model-card, or version documentation that directly covers the requested entity.")
        if "deployment_feasibility" in families:
            items.append("- The requested model parameters or model-card facts needed for the calculation.")
            items.append("- A supported deployment or parallelism plan for the requested system count.")
        if "model_parameters_or_benchmarks" in families:
            items.append("- A benchmark or reproducible serving configuration for throughput, latency, batch size, context length, and cache memory.")
        if "authoritative_sources" in families:
            items.append("- Authoritative sources sufficient for a high-risk conclusion.")
        if not items:
            items.append("- Retrieval sources that directly support the requested claim.")
        return items

    def _compact_spec_text(self, text: str) -> str:
        return re.sub(r"\s+", "", text).lower()

    def _canonical_hardware_spec_number(self, spec: str) -> str | None:
        memory_or_bandwidth = re.search(r"(\d+(?:\.\d+)?)\s*(GB|TB|MB)\s*(/s|ps)?", spec, flags=re.IGNORECASE)
        if memory_or_bandwidth:
            value = memory_or_bandwidth.group(1)
            unit = memory_or_bandwidth.group(2).lower()
            suffix = memory_or_bandwidth.group(3) or ""
            if suffix:
                suffix = "/s"
            return f"{value}{unit}{suffix}"
        token_rate = re.search(r"(\d+(?:\.\d+)?)\s*(tok/s|tokens/s|token/s)", spec, flags=re.IGNORECASE)
        if token_rate:
            return f"{token_rate.group(1)}tok/s"
        return None

    def _low_source_relevance_issue(
        self,
        question: str,
        classification: Classification,
        audit: SearchAudit,
        grounding_decision: GroundingDecision,
    ) -> str | None:
        if classification == "simple" or not audit.executed:
            return None
        question_terms = self._relevance_terms(question)
        if not question_terms:
            return None
        if not audit.sources:
            if grounding_decision.category == "foundational":
                return "搜索未返回可用结果：本轮联网搜索没有可用来源，只能给出低置信的基础概念解释。"
            if grounding_decision.require_retrieval_sources:
                return "搜索未返回可用结果：本轮联网搜索没有可用来源，不能给出确定事实、精确数字或部署结论。"
            return None

        best_overlap = 0
        semantic_aliases = self._semantic_relevance_aliases(question)
        for source in audit.sources:
            source_text = f"{source.title} {source.snippet} {source.url}"
            lowered_source_text = source_text.lower()
            if self._has_strong_entity_match(question_terms, lowered_source_text):
                return None
            if self._has_semantic_alias_match(semantic_aliases, lowered_source_text):
                return None
            source_terms = self._relevance_terms(source_text)
            best_overlap = max(best_overlap, len(question_terms & source_terms))
        if best_overlap >= 2:
            return None
        return "搜索结果相关性不足：当前返回来源没有覆盖问题中的核心主题，不能用模型记忆硬给确定结论。"

    def _allow_foundational_fallback(
        self,
        grounding_decision: GroundingDecision,
        low_relevance_issue: str | None,
    ) -> bool:
        if not low_relevance_issue:
            return False
        return grounding_decision.category == "foundational" and grounding_decision.allow_stable_model_knowledge

    def _has_strong_entity_match(self, question_terms: set[str], source_text: str) -> bool:
        for term in question_terms:
            for alias in self._strong_entity_aliases(term):
                if alias in source_text:
                    return True
        return False

    def _strong_entity_aliases(self, term: str) -> list[str]:
        normalized = term.lower()
        aliases = {
            "cxl": ["cxl", "compute express link"],
            "cx7": ["cx7", "connectx-7", "connectx 7"],
            "nvlink": ["nvlink"],
            "gr00t": ["gr00t"],
            "hbm": ["hbm", "hbm2e", "hbm3", "hbm3e", "high bandwidth memory"],
            "hbm3": ["hbm3", "high bandwidth memory"],
            "moe": ["moe", "mixture-of-experts", "mixture of experts"],
        }
        if normalized in aliases:
            return aliases[normalized]
        if re.fullmatch(r"[a-z]{1,8}\d[a-z0-9-]*", normalized):
            return [normalized]
        return []

    def _trusted_source_domains(self) -> tuple[str, ...]:
        return (
            "nvidia.com",
            "deepseek.com",
            "modelscope.cn",
            "huggingface.co",
            "github.com",
            "computeexpresslink.org",
            "deepmind.google",
            "arxiv.org",
        )

    def _semantic_relevance_aliases(self, question: str) -> list[str]:
        lowered = question.lower()
        aliases: list[str] = []
        if self._is_distributed_model_understanding_question(question):
            aliases.extend(
                [
                    "model parallelism",
                    "tensor parallelism",
                    "pipeline parallelism",
                    "expert parallelism",
                    "context parallelism",
                    "all-to-all",
                    "alltoall",
                    "multi-node",
                    "multiple nodes",
                    "moe",
                ]
            )
        if any(
            marker in lowered
            for marker in (
                "物理大模型",
                "具身",
                "world model",
                "physical ai",
                "embodied",
                "physics foundation",
            )
        ):
            aliases.extend(["physical ai", "world foundation", "world model", "embodied", "cosmos", "genie"])
        if self._is_accelerator_inference_bandwidth_question(lowered):
            aliases.extend(
                [
                    "hbm",
                    "hbm3",
                    "memory bandwidth",
                    "bandwidth",
                    "inference",
                    "llm",
                    "decode",
                    "kv cache",
                    "tensor core",
                    "transformer engine",
                    "tokens per second",
                ]
            )
        return self._unique_queries(aliases)

    def _has_semantic_alias_match(self, aliases: list[str], source_text: str) -> bool:
        return any(alias in source_text for alias in aliases)

    def _weak_relevance_terms(self) -> set[str]:
        return {
            "tok",
            "token",
            "tokens",
            "tps",
            "page",
            "pageview",
            "tracking",
            "script",
        }

    def _low_source_relevance_answer(self, issue: str) -> str:
        return (
            f"{issue}\n\n"
            "我已阻断本轮生成。建议重新搜索更具体的关键词、官方文档、论文、模型卡或厂商技术博客；"
            "在没有相关证据前，只能说明搜索失败，不能给出事实性综述或部署结论。"
        )

    def _relevance_terms(self, text: str) -> set[str]:
        lowered = text.lower()
        raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9+-]{2,}|[\u4e00-\u9fff]{2,}", lowered)
        stop_terms = {
            "2023",
            "2024",
            "2025",
            "2026",
            "what",
            "where",
            "when",
            "with",
            "from",
            "this",
            "that",
            "and",
            "the",
            "for",
            "are",
            "was",
            "were",
            "how",
            "why",
            "can",
            "source",
            "sources",
            "是否",
            "什么",
            "为什么",
            "解释",
            "说明",
            "请给出",
            "截至",
            "目前",
            "现在",
            "发展",
            "哪里",
            "典型",
            "公司",
            "路线",
            "局限",
            "未来",
            "判断",
            "哪里",
            "哪个",
            "就是",
            "这么",
        }
        cjk_signal_terms = (
            "分布式",
            "大模型",
            "模型",
            "节点",
            "互联",
            "通信",
            "并行",
            "协同",
            "带宽",
            "显存",
            "内存",
            "推理",
            "吞吐",
            "解码",
            "缓存",
            "算力",
            "输出",
            "加速",
        )
        terms: set[str] = set()
        for term in raw_terms:
            if term in stop_terms:
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]+", term):
                matched_terms = [signal for signal in cjk_signal_terms if signal in term]
                terms.update(matched_terms)
                if matched_terms and len(term) > 6:
                    continue
            if len(term) < 3 and not re.fullmatch(r"[A-Z0-9+-]{2,}", term.upper()):
                continue
            terms.add(term)
        return terms

    def _extract_answerable_question_from_feedback(self, text: str) -> str | None:
        segments = [segment.strip(" \t\r\n:：；;。") for segment in re.split(r"[\r\n]+", text) if segment.strip()]
        if len(segments) >= 2 and self._looks_like_answerable_question_segment(segments[0]):
            return segments[0]

        feedback_anchors = (
            r"(?:面对|对于|针对)(?:上面|前面|这个|这类)?问题",
            r"(?:面对|对于|针对)(?:上面|前面|这次|这个)?回答",
            r"(?:这个|这类)问题(?:他|你|agent|助手)",
            r"(?:这个|这类)回答(?:他|你|agent|助手)",
        )
        for pattern in feedback_anchors:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match or match.start() <= 0:
                continue
            candidate = text[: match.start()].strip(" \t\r\n:：，,；;。")
            if self._looks_like_answerable_question_segment(candidate):
                return candidate
        return None

    def _looks_like_answerable_question_segment(self, text: str) -> bool:
        candidate = text.strip()
        if len(candidate) < 6:
            return False
        if self._is_user_feedback(candidate):
            return False
        lowered = candidate.lower()
        question_markers = (
            "?",
            "？",
            "what",
            "why",
            "how",
            "can ",
            "could ",
            "is ",
            "does ",
            "是否",
            "能否",
            "能不能",
            "可不可以",
            "可以理解",
            "理解为",
            "是什么",
            "为什么",
            "怎么",
            "如何",
        )
        return any(marker in lowered or marker in candidate for marker in question_markers)

    def _is_user_feedback(self, text: str) -> bool:
        lowered = text.lower()
        correction_signals = (
            "wrong",
            "incorrect",
            "not right",
            "missed",
            "missing",
            "omitted",
            "mistake",
            "error",
            "bug",
            "\u4e0d\u5bf9",
            "\u9519",
            "\u6f0f",
            "\u9057\u6f0f",
            "\u6ca1\u63d0",
            "太死板",
            "很死板",
            "机械",
            "模板",
            "没有体现",
            "没体现",
            "不像",
            "答非所问",
            "摆烂",
            "阻断",
        )
        future_guidance_signals = (
            "next time",
            "in future",
            "from now on",
            "should",
            "do not",
            "don't",
            "avoid",
            "remember",
            "feedback",
            "correction",
            "\u4ee5\u540e",
            "\u4e0b\u6b21",
            "\u5e94\u8be5",
            "\u4e0d\u8981",
            "\u907f\u514d",
            "\u8bb0\u4f4f",
            "\u53cd\u9988",
            "\u7ea0\u6b63",
            "\u6539\u8fdb",
            "修正",
            "纠偏",
            "不应该",
            "不该",
            "必须",
        )
        assistant_context_signals = (
            "you",
            "your",
            "assistant",
            "bot",
            "agent",
            "助手",
            "搜索助手",
            "回答",
            "回复",
            "问题",
            "搜索",
            "联网",
            "证据",
            "基础信息",
            "基础概念",
            "他",
            "\u4f60",
        )
        has_correction = any(signal in lowered for signal in correction_signals)
        has_future_guidance = any(signal in lowered for signal in future_guidance_signals)
        has_assistant_context = any(signal in lowered for signal in assistant_context_signals)
        implied_correction = any(
            signal in lowered
            for signal in (
                "为什么没有",
                "为何没有",
                "你为什么没",
                "没有去",
                "没有查",
                "没去",
                "没查",
                "识别为问题",
                "识别为了问题",
                "还是不知道",
            )
        )
        return (
            has_correction
            or implied_correction
            or self._mentions_source_insufficiency_feedback(text)
            or (has_future_guidance and has_assistant_context)
        )

    def _is_skill_list_request(self, text: str) -> bool:
        lowered = text.lower().strip()
        if lowered in {"skill-list", "skill list", "/skill-list", "/skills", "skills"}:
            return True
        has_skill_target = "skill" in lowered or "技能" in text
        if not has_skill_target:
            return False
        list_signals = (
            "list",
            "show",
            "status",
            "review status",
            "有哪些",
            "列出",
            "查看",
            "展示",
            "状态",
            "列表",
        )
        return any(signal in lowered or signal in text for signal in list_signals)

    def _is_skill_promote_request(self, text: str) -> bool:
        lowered = text.lower().strip()
        if lowered.startswith(("skill-promote", "/skill-promote", "promote skill", "skill promote")):
            return True
        has_skill_target = "skill" in lowered or "技能" in text
        if not has_skill_target:
            return False
        promote_signals = (
            "promote",
            "activate",
            "enable",
            "启用",
            "激活",
            "启用为 active",
            "设为 active",
            "加入 active",
        )
        return any(signal in lowered or signal in text for signal in promote_signals)

    def _is_skill_generation_request(self, text: str) -> bool:
        lowered = text.lower()
        has_skill_target = "skill" in lowered or "技能" in text
        if not has_skill_target:
            return False
        action_signals = (
            "generate",
            "create",
            "draft",
            "make",
            "turn into",
            "promote into",
            "生成",
            "创建",
            "新建",
            "沉淀",
            "整理",
            "提炼",
            "固化",
            "做成",
            "转成",
        )
        return any(signal in lowered or signal in text for signal in action_signals)

    def _is_candidate_confirm_request(self, text: str) -> bool:
        """User confirms a pending knowledge candidate for release."""
        lowered = text.lower().strip()
        if self._is_candidate_pending_list_request(text):
            return False
        if lowered.startswith(("candidate-confirm", "/candidate-confirm", "confirm candidate", "confirm knowledge")):
            return True
        confirm_signals = ("确认知识", "同意入库", "确认入库", "通过候选", "确认沉淀", "确认候选")
        if not any(signal in text for signal in confirm_signals):
            return False
        # A bare "确认候选" without an id is ambiguous; treat it as a prompt to
        # list pending candidates instead of a confirm command.
        return bool(self._candidate_confirm_target(text))

    def _is_candidate_pending_list_request(self, text: str) -> bool:
        """User asks to list candidates waiting for confirmation."""
        lowered = text.lower().strip()
        if lowered.startswith(("candidate-pending", "/candidate-pending", "pending candidates", "list pending")):
            return True
        list_signals = ("待确认知识", "待确认候选", "待入库候选", "待审批候选", "查看待确认")
        return any(signal in text for signal in list_signals)

    def _candidate_confirm_target(self, text: str) -> str:
        """Extract the candidate id from a confirm request."""
        patterns = (
            r"(?:candidate-confirm|/candidate-confirm|confirm\s+(?:candidate|knowledge))\s*[:：]?\s*(.+)$",
            r"(?:确认知识候选|确认知识|同意入库|确认入库|通过候选|确认沉淀|确认候选)\s*[:：]?\s*(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            target = match.group(1).strip()
            # "确认知识候选：<id>" -> "候选：<id>": strip the trailing noun and
            # its colon only when the noun is followed by a separator; never
            # strip characters out of the id itself.
            target = re.sub(r"^(候选|知识|candidate|knowledge)\s*[:：]\s*", "", target, flags=re.IGNORECASE)
            if target and not target.lower().startswith(("list", "待确认")):
                return target
        return ""

    def _answer_candidate_confirm_command(self, message: IncomingMessage, question_id: str) -> AnswerPackage:
        """Confirm one pending knowledge candidate: pending -> validated."""
        from search_assistant.evolution.service import DomainKnowledgeCandidateService

        audit = SearchAudit(
            executed=False,
            queries=[],
            sources=[],
            engines=self._search_engine_names(),
            skipped_reason="candidate confirm command does not trigger web search",
        )
        target = self._candidate_confirm_target(message.text)
        service = DomainKnowledgeCandidateService(self.store)
        confirmed: dict[str, Any] | None = None
        error: str | None = None
        if not target:
            error = "没有识别到要确认的知识候选 id。可先发送“查看待确认候选”获取列表。"
        else:
            try:
                confirmed = service.confirm_candidate(target, reviewer=message.user_id, reason="user confirmed in chat")
            except (KeyError, ValueError) as exc:
                error = str(exc)

        if confirmed is not None and confirmed.get("already_validated"):
            final_answer = f"知识候选 {target} 已经处于已确认（validated）状态，无需重复确认。"
        elif confirmed is not None:
            final_answer = (
                f"已确认知识候选 {target}（{confirmed.get('topic', '')}）入库为 validated_knowledge。"
                "它将作为规划上下文使用，不视为当前事实来源；引用前仍须打开原始来源核实。"
            )
        else:
            final_answer = f"未能确认知识候选：{error or '未知错误'}"
        package = AnswerPackage(
            question_id=question_id,
            answer_text=self._append_search_record(final_answer, audit),
            classification="simple",
            confidence="high" if confirmed else "low",
            verified_claims=[],
            unverified_claims=[] if confirmed else [error or "candidate confirmation failed"],
            sources=[],
            calibration=None,
            review={
                "ran": False,
                "approved": bool(confirmed),
                "issues": [] if confirmed else [error or "candidate confirmation failed"],
                "revision": final_answer,
                "reason": "user confirmed knowledge candidate release from chat",
            },
            memory_updates=[],
            search_record=self._search_record_from_audit(audit),
        )
        self.store.record_answer(package)
        self.store.add_experience_item(
            title="User confirmed knowledge candidate",
            body=(
                f"user_request={message.text}\n"
                f"target={target or ''}\n"
                f"confirmed={bool(confirmed)}\n"
                f"error={error or ''}\n"
                "future_rule=Only user-confirmed knowledge candidates enter validated planning context."
            ),
            source_ids=[package.question_id],
            user_id=message.user_id,
            chat_id=message.chat_id,
        )
        return package

    def _answer_candidate_pending_list_command(self, message: IncomingMessage, question_id: str) -> AnswerPackage:
        """List candidates waiting for user confirmation."""
        from search_assistant.evolution.service import DomainKnowledgeCandidateService

        audit = SearchAudit(
            executed=False,
            queries=[],
            sources=[],
            engines=self._search_engine_names(),
            skipped_reason="candidate pending list command does not trigger web search",
        )
        service = DomainKnowledgeCandidateService(self.store)
        pending = [
            candidate
            for candidate in service.list_candidates(status="pending_user_confirm")
            if str(candidate.get("status")) == "pending_user_confirm"
        ]
        if not pending:
            final_answer = "当前没有等待确认的知识候选。"
        else:
            lines = ["以下知识候选已通过自动验证门，等待你确认入库：", ""]
            for item in pending:
                lines.append(
                    f"- id: {item['id']} | 主题: {item['topic']} | 置信: {item.get('confidence')} | "
                    f"证据: {item.get('evidence_url_count', 0)} 个来源"
                )
            lines.extend(
                [
                    "",
                    "回复“确认知识候选：<id>”即可将其入库为 validated_knowledge；",
                    "若内容有问题，可回复“废弃知识候选：<id>”。",
                ]
            )
            final_answer = "\n".join(lines)
        package = AnswerPackage(
            question_id=question_id,
            answer_text=self._append_search_record(final_answer, audit),
            classification="simple",
            confidence="high",
            verified_claims=[],
            unverified_claims=[],
            sources=[],
            calibration=None,
            review={
                "ran": False,
                "approved": True,
                "issues": [],
                "revision": final_answer,
                "reason": "user requested pending knowledge candidate list",
            },
            memory_updates=[],
            search_record=self._search_record_from_audit(audit),
        )
        self.store.record_answer(package)
        return package

    def _skill_promote_target(self, text: str) -> str:
        patterns = (
            r"(?:skill-promote|/skill-promote|promote\s+skill|skill\s+promote|activate\s+skill|enable\s+skill)\s*[:：]?\s*(.+)$",
            r"(?:启用|激活|设为\s*active|加入\s*active)\s*(?:skill|技能)\s*[:：]?\s*(.+)$",
            r"(?:skill|技能)\s*[:：]\s*(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            target = self._clean_skill_title(match.group(1))
            if target:
                return target
        return ""

    def _is_evolution_request(self, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        exact_commands = {
            "evolve",
            "/evolve",
            "run evolve",
            "start evolve",
            "run self-evolution",
            "start self-evolution",
            "self-evolution",
            "self evolution",
            "开始自进化",
            "运行自进化",
            "执行自进化",
            "启动自进化",
            "自进化",
            "开始自学习",
            "运行自学习",
        }
        if normalized in exact_commands or text.strip() in exact_commands:
            return True
        has_target = any(signal in normalized or signal in text for signal in ("evolve", "self-evolution", "自进化", "自学习"))
        has_action = any(
            signal in normalized or signal in text
            for signal in ("run", "start", "execute", "开始", "运行", "执行", "启动")
        )
        question_markers = ("what", "why", "how", "是什么", "为什么", "如何", "怎么", "?")
        return has_target and has_action and not any(marker in normalized or marker in text for marker in question_markers)

    def _is_learning_report_request(self, text: str) -> bool:
        lowered = text.lower().strip()
        has_report_target = any(
            signal in lowered or signal in text
            for signal in (
                "learning report",
                "study report",
                "progress report",
                "学习报告",
                "學習報告",
                "学习总结",
                "學習總結",
                "学习复盘",
                "學習復盤",
            )
        )
        if not has_report_target:
            return False
        action_signals = (
            "generate",
            "create",
            "write",
            "show",
            "summarize",
            "report",
            "生成",
            "创建",
            "输出",
            "查看",
            "整理",
            "总结",
            "复盘",
        )
        return any(signal in lowered or signal in text for signal in action_signals)

    def _skill_generation_title(self, text: str) -> str:
        patterns = (
            r"(?:skill|技能)\s*[:：]\s*(.+)$",
            r"(?:标题|名称|name)\s*[:：]\s*(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                title = self._clean_skill_title(match.group(1))
                if title:
                    return title
        topics = self.store.list_profile_snapshots()
        if topics:
            try:
                summary = json.loads(str(topics[-1]["summary_json"]))
            except (KeyError, json.JSONDecodeError):
                summary = {}
            recurring_topics = summary.get("recurring_topics", [])
            if recurring_topics and isinstance(recurring_topics[0], str):
                return f"{recurring_topics[0]} Practice"
        return "Conversation Requested Skill"

    def _clean_skill_title(self, title: str) -> str:
        cleaned = title.strip().strip("`\"'“”‘’")
        cleaned = re.sub(r"[。.!！?？]+$", "", cleaned).strip()
        return cleaned[:80]

    def _skill_source_ids_for_draft(self, user_id: str, chat_id: str) -> list[str]:
        experiences = self.store.list_experience_items(user_id, chat_id)
        if experiences:
            return [str(experiences[-1]["id"])]
        profiles = self.store.list_profile_snapshots()
        if profiles:
            return [str(profiles[-1]["id"])]
        answers = self.store.list_answers()
        if answers:
            return [answers[-1].question_id]
        return ["manual"]

    def _skill_generation_acknowledgement_answer(self, title: str, draft_path: str, source_ids: list[str]) -> str:
        return "\n".join(
            [
                f"已创建 skill 草稿: {title}",
                f"路径: {draft_path}",
                f"来源: {', '.join(source_ids)}",
                "",
                "状态: review-only draft，尚未启用。",
                "下一步: 审核 SKILL.md 后运行 skill-promote，才会进入后续问答上下文。",
            ]
        )

    def _skill_list_answer(self, result: dict[str, object]) -> str:
        counts = result.get("counts", {})
        if not isinstance(counts, dict):
            counts = {}
        skills = result.get("skills", [])
        if not isinstance(skills, list):
            skills = []
        lines = [
            "skill 列表:",
            (
                f"total={counts.get('total', 0)} "
                f"draft={counts.get('draft', 0)} "
                f"promoted={counts.get('promoted', 0)} "
                f"active={counts.get('active_files', 0)} "
                f"missing={counts.get('missing_files', 0)}"
            ),
        ]
        if not skills:
            lines.append("- 暂无 skill 草稿。可以先运行 自进化 或 请生成skill：<名称>。")
            return "\n".join(lines)
        for item in skills:
            if not isinstance(item, dict):
                continue
            status = str(item.get("review_status", "unknown"))
            slug = str(item.get("slug", ""))
            name = str(item.get("name", ""))
            path = str(item.get("path", ""))
            active = " active" if item.get("active") else ""
            lines.append(f"- [{status}{active}] {slug} | {name} | {path}")
        lines.extend(
            [
                "",
                "启用方式: 审核草稿内容后发送 `启用skill：<名称或slug>`。",
            ]
        )
        return "\n".join(lines)

    def _skill_promote_answer(self, target: str, promoted_path: str | None, error: str | None) -> str:
        if promoted_path:
            slug = Path(promoted_path).parent.name
            return "\n".join(
                [
                    f"已启用 skill: {slug}",
                    f"active path: {promoted_path}",
                    "",
                    "后续普通问答会把这个 SKILL.md 作为已审核的过程指导传给搜索规划、生成、校准和审查阶段。",
                    "注意: skill 不是事实来源，事实仍必须由联网搜索结果支持。",
                ]
            )
        return "\n".join(
            [
                f"未能启用 skill: {target or '(未识别)'}",
                f"原因: {error or 'unknown error'}",
                "",
                "可先发送 `查看skill列表` 确认名称或 slug，再发送 `启用skill：<名称或slug>`。",
            ]
        )

    def _learning_report_acknowledgement_answer(self, markdown: str, report_path: Path) -> str:
        excerpt_lines = [line for line in markdown.splitlines() if line.strip()]
        excerpt = "\n".join(excerpt_lines[:12])
        return "\n".join(
            [
                "已生成学习报告。",
                f"路径: {report_path}",
                "",
                "报告摘录:",
                excerpt,
            ]
        )

    def _evolution_acknowledgement_answer(
        self,
        report_path: Path,
        markdown: str,
        skill_paths: list[str],
        refreshed_skill_paths: list[str],
    ) -> str:
        lines = [
            "已运行自进化。",
            f"学习报告: {report_path}",
            f"报告长度: {len(markdown)} 字符",
            f"新增 skill 草稿: {len(skill_paths)}",
            f"刷新 skill 草稿: {len(refreshed_skill_paths)}",
        ]
        if skill_paths:
            lines.extend(["", "review-only draft:"])
            for path in skill_paths:
                title = Path(path).parent.name.replace("-", " ").title()
                lines.append(f"- {title}: {path}")
        if refreshed_skill_paths:
            lines.extend(["", "refreshed review-only draft:"])
            for path in refreshed_skill_paths:
                title = Path(path).parent.name.replace("-", " ").title()
                lines.append(f"- {title}: {path}")
        if skill_paths or refreshed_skill_paths:
            lines.extend(
                [
                    "",
                    "下一步: 审核这些 SKILL.md 后运行 skill-promote，才会进入后续问答上下文。",
                ]
            )
        else:
            lines.append("本轮没有新增或刷新 skill 草稿，通常是因为没有新的画像主题/经验，或同名草稿已经是最新模板。")
        return "\n".join(lines)

    def _feedback_acknowledgement_answer(self, text: str) -> str:
        lines = [
            "已记录这条修正，我会把它作为后续搜索和回答的经验规则。",
            "",
            "后续处理规则:",
        ]
        if self._mentions_deepseek_model_community_feedback(text):
            lines.extend(
                [
                    "- DeepSeek 模型参数、模型卡、部署核验类问题，优先查开源模型社区和官方模型卡。",
                    "- 必查来源族: ModelScope/魔搭/魔塔、Hugging Face、GitHub；涉及硬件部署时再补 NVIDIA 官方页。",
                    "- 如果这些来源互相不一致，要明确列出差异，不用模型记忆硬补参数。",
                ]
            )
        else:
            lines.append("- 以后遇到同类问题，先按这条反馈调整搜索规划，再生成回答。")
        if self._mentions_answer_style_feedback(text):
            lines.append("- 回答风格要更像技术伙伴：先给判断，再说明证据、假设、推理路径和不确定点，避免模板化清单。")
        if self._mentions_source_insufficiency_feedback(text):
            lines.append("- 基础概念/理解校正类问题不能因为搜索证据弱就直接阻断；要给低置信的基础解释，并明确说明联网证据不足。")
        lines.append("- 这类修正输入不再按普通问答处理。")
        return "\n".join(lines)

    def _feedback_experience_body(
        self,
        message: IncomingMessage,
        package: AnswerPackage,
        audit: SearchAudit,
    ) -> str:
        source_urls = [source.url for source in package.sources]
        return (
            f"user_feedback={message.text}\n"
            f"source={message.source}\n"
            f"classification={package.classification}\n"
            f"confidence={package.confidence}\n"
            f"search_executed={audit.executed}\n"
            f"search_engines={json.dumps(audit.engines, ensure_ascii=False)}\n"
            f"search_queries={json.dumps(audit.queries, ensure_ascii=False)}\n"
            f"source_urls={json.dumps(source_urls, ensure_ascii=False)}\n"
            f"{self._feedback_source_family_body(message.text)}"
            f"{self._feedback_answer_style_body(message.text)}"
            f"{self._feedback_source_relevance_body(message.text)}"
            "future_rule=Use this as planning and calibration guidance: infer the user's intent first, "
            "derive fresh search queries from the question, cover missed entities/source families, "
            "and avoid repeating the cited mistake."
        )

    def _feedback_source_family_body(self, text: str) -> str:
        if not self._mentions_deepseek_model_community_feedback(text):
            return ""
        return (
            "source_family=deepseek_model_communities\n"
            "canonical_sources=ModelScope/魔搭/魔塔, Hugging Face, GitHub, NVIDIA official pages when hardware is involved\n"
            "future_rule_detail=For DeepSeek model parameter/model-card/deployment questions, search open model communities before answering.\n"
        )

    def _feedback_answer_style_body(self, text: str) -> str:
        if not self._mentions_answer_style_feedback(text):
            return ""
        return (
            "answer_style=natural_technical_partner\n"
            "avoid_rigid_template=true\n"
            "future_rule_detail=For technical, discussion, or uncertain answers, show a compact user-visible reasoning path: judgment, evidence, assumptions, tradeoffs, uncertainty, and next verification.\n"
        )

    def _feedback_source_relevance_body(self, text: str) -> str:
        if not self._mentions_source_insufficiency_feedback(text):
            return ""
        return (
            "source_relevance_policy=foundational_fallback\n"
            "avoid_blocking_foundational_answers=true\n"
            "future_rule_detail=For foundational concept or understanding-check questions, weak live-search relevance should lower confidence and be disclosed, not block the basic explanation.\n"
        )

    def _mentions_source_insufficiency_feedback(self, text: str) -> bool:
        lowered = text.lower()
        evidence_terms = (
            "搜索内容不足",
            "搜索结果不足",
            "搜索证据不足",
            "搜索不足",
            "证据不足",
            "source insufficiency",
            "weak search",
            "weak evidence",
        )
        blocking_terms = (
            "阻断",
            "拒答",
            "不回答",
            "不能回答",
            "摆烂",
            "block",
            "refuse",
            "refusal",
        )
        foundational_terms = (
            "基础信息",
            "基础的信息",
            "基础概念",
            "基本信息",
            "基本解释",
            "概念题",
            "foundational",
            "basic explanation",
        )
        return (
            any(term in lowered or term in text for term in evidence_terms)
            and any(term in lowered or term in text for term in blocking_terms)
        ) or (
            any(term in lowered or term in text for term in evidence_terms)
            and any(term in lowered or term in text for term in foundational_terms)
        )

    def _mentions_answer_style_feedback(self, text: str) -> bool:
        lowered = text.lower()
        style_terms = (
            "太死板",
            "很死板",
            "机械",
            "模板",
            "template",
            "canned",
            "rigid",
            "stiff",
            "没有体现",
            "没体现",
            "思考",
            "reasoning",
            "technical partner",
            "技术伙伴",
            "判断过程",
            "推理路径",
        )
        answer_context = ("回答", "回复", "answer", "response", "assistant", "bot", "agent", "助手", "搜索助手")
        return any(term in lowered or term in text for term in style_terms) and any(
            term in lowered or term in text for term in answer_context
        )

    def _mentions_deepseek_model_community_feedback(self, text: str) -> bool:
        lowered = text.lower()
        has_deepseek = "deepseek" in lowered
        has_model_fact = any(term in lowered for term in ("模型参数", "参数", "模型卡", "model parameter", "model card", "部署"))
        has_community = any(
            term in lowered
            for term in (
                "modelscope",
                "魔搭",
                "魔塔",
                "hugging face",
                "huggingface",
                "github",
                "开源模型社区",
                "模型社区",
            )
        )
        return has_deepseek and has_model_fact and has_community

    def _search_record_from_audit(self, audit: SearchAudit) -> SearchRecord:
        return SearchRecord(
            executed=audit.executed,
            queries=list(audit.queries),
            sources=list(audit.sources),
            engines=list(audit.engines),
            skipped_reason=audit.skipped_reason,
        )

    def _append_search_record(self, answer_text: str, audit: SearchAudit) -> str:
        search_status = "已执行" if audit.executed else f"未执行（{audit.skipped_reason or '未配置搜索客户端'}）"
        lines = [
            answer_text.rstrip(),
            "",
            "搜索记录:",
            f"- 联网搜索: {search_status}",
        ]
        if audit.engines:
            lines.append(f"- 搜索引擎: {', '.join(audit.engines)}")
        if audit.queries:
            lines.append("- 搜索词:")
            for index, query in enumerate(audit.queries, start=1):
                lines.append(f"  {index}. {query}")
        else:
            lines.append("- 搜索词: 无")

        if audit.sources:
            lines.append("- 搜索结果:")
            for index, source in enumerate(audit.sources, start=1):
                lines.append(f"  {index}. [{source.provider}] {source.title} - {source.url}")
        else:
            lines.append("- 搜索结果: 未返回可用结果")
        return "\n".join(lines)

    def _search_sources(
        self,
        text: str,
        classification: Classification,
        planning_context: dict[str, object],
    ) -> SearchAudit:
        if self.search_client is None:
            return SearchAudit(executed=False, queries=[], sources=[])
        queries = self._planned_search_queries(text, planning_context)
        engines = self._search_engine_names()
        per_query_limit = 2 if len(queries) > 1 else 5
        results: list[SourceEvidence] = []
        seen: set[str] = set()
        attempted_queries: list[str] = []
        deadline = time.monotonic() + self.search_budget_seconds
        for index, query in enumerate(queries):
            if index > 0 and time.monotonic() >= deadline:
                break
            attempted_queries.append(query)
            try:
                query_results = self.search_client.search(query, limit=per_query_limit)
            except Exception:
                continue
            for source in query_results:
                key = self._source_key(source)
                if key in seen:
                    continue
                seen.add(key)
                results.append(source)
        ranked_results = self._rank_sources(text, results)
        return SearchAudit(executed=True, queries=attempted_queries, sources=ranked_results[:5], engines=engines)

    def _rank_sources(self, question: str, sources: list[SourceEvidence]) -> list[SourceEvidence]:
        question_terms = self._relevance_terms(question)
        semantic_aliases = self._semantic_relevance_aliases(question)
        weak_terms = self._weak_relevance_terms()

        def source_score(indexed_source: tuple[int, SourceEvidence]) -> int:
            index, source = indexed_source
            source_text = f"{source.title} {source.snippet} {source.url}".lower()
            source_terms = self._relevance_terms(source_text)
            overlap = question_terms & source_terms
            significant_overlap = overlap - weak_terms
            strong_entity_match = self._has_strong_entity_match(question_terms, source_text)
            semantic_alias_match = self._has_semantic_alias_match(semantic_aliases, source_text)
            score = 10 * len(significant_overlap)
            if str(source.provider).startswith("direct-"):
                score += 50
            if strong_entity_match:
                score += 30
            if semantic_alias_match:
                score += 30
            if any(domain in source.url for domain in self._trusted_source_domains()):
                score += 10
            if not significant_overlap and not strong_entity_match and not semantic_alias_match:
                score -= 20
            return score

        scored_sources = [(source_score(indexed_source), indexed_source[0], indexed_source[1]) for indexed_source in enumerate(sources)]
        ordered = sorted(scored_sources, key=lambda item: (item[0], -item[1]), reverse=True)
        relevant_sources = [source for score, _, source in ordered if score >= 0]
        if relevant_sources:
            return relevant_sources
        return [source for _, _, source in ordered]

    def _search_engine_names(self) -> list[str]:
        names = getattr(self.search_client, "engine_names", [])
        if not isinstance(names, list):
            return []
        return [str(name) for name in names if str(name).strip()]

    def _planned_search_queries(self, text: str, planning_context: dict[str, object]) -> list[str]:
        try:
            planned = self.runtime.plan_search_queries(text, planning_context)
        except Exception:
            planned = []
        queries = self._unique_queries([str(query) for query in planned if str(query).strip()])
        queries = self._unique_queries(queries + self._feedback_constraint_queries(text, planning_context, queries))
        if queries:
            return queries[:6]
        return self._unique_queries([self._search_query(text)])

    def _feedback_constraint_queries(
        self,
        text: str,
        planning_context: dict[str, object],
        planned_queries: list[str],
    ) -> list[str]:
        experience = planning_context.get("experience", [])
        if not isinstance(experience, list):
            return []

        question_terms = f"{text} {' '.join(planned_queries)}".lower()
        constraints: list[str] = []
        for item in experience:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", ""))
            if "User feedback" not in title and "用户反馈" not in title:
                continue
            body = str(item.get("body", ""))
            if not self._feedback_applies_to_question(body, question_terms):
                continue
            constraints.extend(self._extract_feedback_entities(body))

        base_query = self._search_query(text)
        return [f"{base_query} {constraint}" for constraint in self._unique_queries(constraints)]

    def _feedback_applies_to_question(self, feedback_body: str, question_terms: str) -> bool:
        feedback_terms = feedback_body.lower()
        anchors = re.findall(r"[A-Za-z][A-Za-z0-9+-]{2,}", feedback_terms)
        anchors.extend(re.findall(r"[\u4e00-\u9fff]{2,}", feedback_body))
        stop_anchors = {
            "user",
            "feedback",
            "future",
            "rule",
            "cover",
            "missed",
            "entities",
            "source",
            "families",
            "以后回答",
            "不要漏掉",
            "请记录",
            "说明后续",
        }
        for anchor in anchors:
            normalized = anchor.lower()
            if normalized in stop_anchors:
                continue
            if normalized in question_terms:
                return True
        return False

    def _extract_feedback_entities(self, feedback_body: str) -> list[str]:
        entities: list[str] = []
        if "source_family=deepseek_model_communities" in feedback_body or self._mentions_deepseek_model_community_feedback(feedback_body):
            entities.extend(
                [
                    "ModelScope 魔搭 魔塔 DeepSeek 模型参数 模型卡",
                    "Hugging Face deepseek-ai model card parameters",
                    "GitHub deepseek-ai model repository parameters",
                ]
            )
        patterns = (
            r"(?:不要漏掉|别漏掉|漏掉了|遗漏了|include|cover|missing|missed)\s*([^。\n；;]+)",
            r"(?:没有去|没去|没有查|没查)\s*([^。\n；;]+?)(?:去查找|查找|搜索|检索|找)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, feedback_body, flags=re.IGNORECASE):
                entities.extend(self._split_feedback_entity_phrase(match.group(1)))
        return self._unique_queries(entities)

    def _split_feedback_entity_phrase(self, phrase: str) -> list[str]:
        phrase = re.sub(r"(?:网卡|社区|平台|source|sources|entities|families)\b", "", phrase, flags=re.IGNORECASE)
        parts = re.split(r"(?:和|及|与|、|，|,|/|\\|\band\b|\bor\b)", phrase)
        entities: list[str] = []
        for part in parts:
            candidate = re.sub(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff+-]+$", "", part).strip()
            candidate = re.sub(r"\s+", " ", candidate)
            if len(candidate) < 2:
                continue
            if candidate.lower() in {"user_feedback", "future_rule", "不要漏掉", "include", "cover"}:
                continue
            entities.append(candidate)
        return entities

    def _unique_queries(self, queries: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = re.sub(r"\s+", " ", query).strip()
            if not normalized or normalized.lower() in seen:
                continue
            unique.append(normalized)
            seen.add(normalized.lower())
        return unique

    def _source_key(self, source: SourceEvidence) -> str:
        return re.sub(r"#.*$", "", source.url).rstrip("/").lower()

    def _search_query(self, text: str) -> str:
        hardware_query = self._hardware_inference_search_query(text)
        if hardware_query:
            return hardware_query
        query = text.strip()
        query = re.split(r"[？?。]\s*(?:请|请问|麻烦)?", query, maxsplit=1)[0]
        query = re.split(r"\s+(?:请|请问|麻烦)", query, maxsplit=1)[0]
        query = re.sub(r"^\s*(?:请|请问|麻烦|帮我)?(?:搜索|查询|检索|查找|搜一下)(?:并)?(?:计算|解释|总结)?[:：]?\s*", "", query)
        query = re.sub(r"[（(]([^）)]*)[）)]", r" \1 ", query)
        query = self._remove_search_noise(query)
        query = self._compact_english_search_query(query)
        query = re.sub(r"[：:，,；;。？?、/]+", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        return query or text.strip()

    def _hardware_inference_search_query(self, text: str) -> str | None:
        lowered = text.lower()
        accelerator = self._accelerator_like_entity(text)
        if not accelerator or not self._is_accelerator_inference_bandwidth_question(lowered, accelerator):
            return None

        query_terms = [
            accelerator,
            "tok/s",
            "LLM inference",
            "decode throughput",
            "HBM3 memory bandwidth",
            "KV cache",
            "Tensor Core",
        ]
        return " ".join(query_terms)

    def _accelerator_like_entity(self, text: str) -> str | None:
        match = re.search(r"(?<![A-Za-z0-9])[A-Z]{1,8}\d{2,}[A-Za-z0-9-]*(?![A-Za-z0-9])", text)
        if match:
            return match.group(0).upper()
        return None

    def _is_accelerator_inference_bandwidth_question(self, lowered_text: str, accelerator: str | None = None) -> bool:
        has_accelerator = bool(accelerator) or bool(
            re.search(r"(?<![a-z0-9])[a-z]{1,8}\d{2,}[a-z0-9-]*(?![a-z0-9])", lowered_text)
        )
        has_throughput_signal = any(
            term in lowered_text
            for term in (
                "tok/s",
                "tokens/s",
                "token/s",
                "tokens per second",
                "tps",
                "throughput",
                "decode",
                "inference",
                "推理",
                "吞吐",
                "输出",
            )
        )
        has_bandwidth_signal = any(
            term in lowered_text
            for term in (
                "bandwidth",
                "memory bandwidth",
                "hbm",
                "hbm3",
                "hbm3e",
                "kv cache",
                "带宽",
                "显存",
                "内存",
                "缓存",
            )
        )
        return has_accelerator and (has_throughput_signal or has_bandwidth_signal)

    def _remove_search_noise(self, query: str) -> str:
        year_match = re.search(r"(20\d{2})\s*年", query)
        year = year_match.group(1) if year_match else ""
        cleaned = query
        noise_patterns = (
            r"^\s*截至\s*20\d{2}\s*年\s*",
            r"^\s*\d+\s*台\s*并联\s*",
            r"能否部署满血",
            r"发展到哪里了",
            r"目前",
            r"现在",
        )
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, " ", cleaned)
        if year and "AI 物理大模型" in cleaned and year not in cleaned:
            cleaned = f"{cleaned} {year}"
        return cleaned

    def _compact_english_search_query(self, query: str) -> str:
        if not re.search(r"[A-Za-z]", query):
            return query
        if re.search(r"[\u4e00-\u9fff]", query):
            return query

        lowered_query = query.lower()
        if not any(phrase in lowered_query for phrase in ("help me", "what is", "as of", "where are", "how does")):
            return query

        cleaned = query
        cleaned = re.sub(r"[,.;:?!()\[\]{}\"']", " ", cleaned)
        raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|\d{4}|\d+", cleaned)

        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "be",
            "can",
            "calculate",
            "does",
            "deploy",
            "full",
            "help",
            "how",
            "include",
            "includes",
            "including",
            "is",
            "it",
            "me",
            "of",
            "or",
            "parallel",
            "relate",
            "should",
            "system",
            "systems",
            "the",
            "to",
            "what",
            "where",
            "whether",
            "with",
        }
        tokens: list[str] = []
        years: list[str] = []
        for raw_token in raw_tokens:
            token = self._canonical_search_token(raw_token)
            lower = token.lower()
            if re.fullmatch(r"20\d{2}", token):
                if token not in years:
                    years.append(token)
                continue
            if token.isdigit():
                continue
            if lower in stopwords or lower == "class":
                continue
            tokens.append(token)

        compact = " ".join(tokens + years)
        return compact or query

    def _canonical_search_token(self, token: str) -> str:
        lower = token.lower()
        if lower == "nvidia":
            return "NVIDIA"
        if lower == "ai":
            return "AI"
        if lower == "cxl":
            return "CXL"
        if lower == "gpus":
            return "GPUs"
        return token
