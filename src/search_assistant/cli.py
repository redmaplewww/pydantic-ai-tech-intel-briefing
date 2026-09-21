from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from search_assistant.contracts import IncomingMessage
from search_assistant.briefing.service import DailyBriefingService
from search_assistant.diagnostics.readiness import run_readiness_diagnostics
from search_assistant.evaluation.service import EvaluationService
from search_assistant.evolution.service import (
    KNOWLEDGE_LAYER_REJECTED,
    KNOWLEDGE_LAYER_UNREVIEWED,
    KNOWLEDGE_LAYER_VALIDATED,
    KNOWLEDGE_LAYER_WEAK_SIGNAL,
    DomainKnowledgeCandidateService,
)
from search_assistant.feishu.client import FakeFeishuClient, FeishuHttpClient
from search_assistant.feishu.events import parse_feishu_event
from search_assistant.knowledge_graph.embedding import embedding_provider_from_settings
from search_assistant.knowledge_graph.service import DomainKnowledgeGraphService
from search_assistant.memory.store import MemoryStore
from search_assistant.profile.service import ProfileService
from search_assistant.reports.service import ReportService
from search_assistant.search.provider import search_client_from_settings
from search_assistant.search.source_registry import (
    normalize_source_recipe,
    source_contracts_as_dicts,
    source_recipe_summary,
)
from search_assistant.skills.service import SkillDraftService
from search_assistant.config import Settings
from search_assistant.verification.backfill import VerificationBackfillService
from search_assistant.runtime.pydantic_ai import runtime_from_settings
from search_assistant.workflow.service import SearchAssistantWorkflow


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(prog="search-assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    _add_data_dir(ask_parser)

    topic_add_parser = subparsers.add_parser("topic-add")
    topic_add_parser.add_argument("topic")
    topic_add_parser.add_argument("--user-id", default="local-user")
    topic_add_parser.add_argument("--chat-id", default="local-cli")
    _add_data_dir(topic_add_parser)

    topic_list_parser = subparsers.add_parser("topic-list")
    topic_list_parser.add_argument("--user-id", default="local-user")
    topic_list_parser.add_argument("--chat-id", default="local-cli")
    _add_data_dir(topic_list_parser)

    source_contracts_parser = subparsers.add_parser("source-contracts")
    _add_data_dir(source_contracts_parser)

    topic_recipe_show_parser = subparsers.add_parser("topic-recipe-show")
    topic_recipe_show_parser.add_argument("topic")
    topic_recipe_show_parser.add_argument("--user-id", default="local-user")
    topic_recipe_show_parser.add_argument("--chat-id", default="local-cli")
    _add_data_dir(topic_recipe_show_parser)

    topic_recipe_set_parser = subparsers.add_parser("topic-recipe-set")
    topic_recipe_set_parser.add_argument("topic")
    topic_recipe_set_parser.add_argument("recipe_json")
    topic_recipe_set_parser.add_argument("--user-id", default="local-user")
    topic_recipe_set_parser.add_argument("--chat-id", default="local-cli")
    _add_data_dir(topic_recipe_set_parser)

    briefing_parser = subparsers.add_parser("brief-run")
    briefing_parser.add_argument("topic")
    briefing_parser.add_argument("--user-id", default="local-user")
    briefing_parser.add_argument("--chat-id", default="local-cli")
    _add_data_dir(briefing_parser)

    briefing_loop_parser = subparsers.add_parser("brief-loop")
    briefing_loop_parser.add_argument("topic")
    briefing_loop_parser.add_argument("--user-id", default="local-user")
    briefing_loop_parser.add_argument("--chat-id", default="local-cli")
    briefing_loop_parser.add_argument("--interval-seconds", type=float, default=86400.0)
    briefing_loop_parser.add_argument("--max-runs", type=int, default=None)
    briefing_loop_parser.add_argument(
        "--topics",
        default=None,
        help="Comma-separated topic pool to rotate each run, e.g. '工业智能体,GraphRAG,医学影像'. "
        "Repeated tracking of a fixed pool lets cross-trajectory knowledge candidates accumulate.",
    )
    briefing_loop_parser.add_argument(
        "--topics-file",
        default=None,
        help="Path to a UTF-8 file with one topic per line; rotates together with --topics.",
    )
    _add_data_dir(briefing_loop_parser)

    feedback_parser = subparsers.add_parser("brief-feedback")
    feedback_parser.add_argument("topic")
    feedback_parser.add_argument("body")
    feedback_parser.add_argument("--url", default=None)
    feedback_parser.add_argument("--user-id", default="local-user")
    feedback_parser.add_argument("--chat-id", default="local-cli")
    _add_data_dir(feedback_parser)

    report_parser = subparsers.add_parser("report")
    _add_data_dir(report_parser)

    candidate_list_parser = subparsers.add_parser("candidate-list")
    candidate_list_parser.add_argument("--topic-id", default=None)
    candidate_list_parser.add_argument("--status", default=None)
    candidate_list_parser.add_argument("--limit", type=_positive_int, default=50)
    _add_data_dir(candidate_list_parser)

    provider_trace_list_parser = subparsers.add_parser("provider-trace-list")
    provider_trace_list_parser.add_argument("--topic-id", default=None)
    provider_trace_list_parser.add_argument("--run-id", default=None)
    provider_trace_list_parser.add_argument("--limit", type=_positive_int, default=50)
    _add_data_dir(provider_trace_list_parser)

    memory_layer_list_parser = subparsers.add_parser("memory-layer-list")
    memory_layer_list_parser.add_argument("--layer", default=None)
    memory_layer_list_parser.add_argument("--user-id", default=None)
    memory_layer_list_parser.add_argument("--chat-id", default=None)
    memory_layer_list_parser.add_argument("--limit", type=_positive_int, default=50)
    _add_data_dir(memory_layer_list_parser)

    evolve_parser = subparsers.add_parser("evolve")
    _add_data_dir(evolve_parser)

    evolve_loop_parser = subparsers.add_parser("evolve-loop")
    _add_data_dir(evolve_loop_parser)
    evolve_loop_parser.add_argument("--interval-seconds", type=float, default=86400.0)
    evolve_loop_parser.add_argument("--max-runs", type=int, default=None)

    offline_evolution_parser = subparsers.add_parser(
        "evolution-offline-run",
        help="Verify immutable trajectories with the three-layer verifier, distill knowledge candidates, and monitor release gates.",
    )
    offline_evolution_parser.add_argument(
        "--judge",
        choices=["rule", "llm"],
        default="rule",
        help="Quality-layer judge: deterministic offline rules (default) or an LLM rubric judge.",
    )
    offline_evolution_parser.add_argument(
        "--max-trajectories",
        type=_positive_int,
        default=None,
        help="Only verify the newest N trajectories that do not yet have an evaluation.",
    )
    _add_data_dir(offline_evolution_parser)

    judge_calibration_parser = subparsers.add_parser(
        "evolution-judge-calibrate",
        help="Measure a quality judge's agreement against the expert-labeled calibration set.",
    )
    judge_calibration_parser.add_argument(
        "--judge",
        choices=["rule", "llm"],
        default="rule",
        help="Quality judge to calibrate: deterministic rule judge (default) or the LLM rubric judge.",
    )
    _add_data_dir(judge_calibration_parser)

    eval_parser = subparsers.add_parser("eval-suite")
    eval_parser.add_argument("--questions", default=None)
    eval_parser.add_argument("--max-questions", type=_positive_int, default=None)
    eval_parser.add_argument(
        "--judge",
        choices=["rule", "llm"],
        default="rule",
        help="Quality-layer judge for answer evaluation: deterministic rule judge (default) or an LLM rubric judge.",
    )
    eval_parser.add_argument(
        "--pool",
        choices=["mixed", "simple", "research", "hard", "high_stakes"],
        default=None,
        help="Use the layered evaluation question pool (default: all 12 questions) or a specific tier. "
        "mixed samples 2 questions per tier.",
    )
    _add_data_dir(eval_parser)

    eval_replay_parser = subparsers.add_parser("eval-replay")
    eval_replay_parser.add_argument("--report", default=None)
    eval_replay_parser.add_argument("--max-items", type=_positive_int, default=None)
    eval_replay_parser.add_argument(
        "--judge",
        choices=["rule", "llm"],
        default="rule",
        help="Quality judge for previous-vs-current answer deltas (default: deterministic rule judge).",
    )
    _add_data_dir(eval_replay_parser)

    evidence_backfill_parser = subparsers.add_parser("evidence-backfill")
    _add_data_dir(evidence_backfill_parser)

    kg_seed_parser = subparsers.add_parser("knowledge-graph-seed")
    kg_seed_parser.add_argument("--domain", action="append", default=None)
    _add_data_dir(kg_seed_parser)

    kg_list_parser = subparsers.add_parser("knowledge-graph-list")
    _add_data_dir(kg_list_parser)

    kg_query_parser = subparsers.add_parser("knowledge-graph-query")
    kg_query_parser.add_argument("query")
    kg_query_parser.add_argument("--domain", default=None)
    kg_query_parser.add_argument("--limit", type=_positive_int, default=10)
    _add_data_dir(kg_query_parser)

    kg_export_parser = subparsers.add_parser("knowledge-graph-export")
    kg_export_parser.add_argument("domain")
    kg_export_parser.add_argument("--output", default=None)
    _add_data_dir(kg_export_parser)

    candidate_list_parser = subparsers.add_parser("knowledge-candidate-list")
    candidate_list_parser.add_argument(
        "--status",
        choices=["candidate", "validated", "deprecated", "pending_user_confirm"],
        default=None,
    )
    candidate_list_parser.add_argument(
        "--layer",
        choices=[
            KNOWLEDGE_LAYER_VALIDATED,
            KNOWLEDGE_LAYER_WEAK_SIGNAL,
            KNOWLEDGE_LAYER_REJECTED,
            KNOWLEDGE_LAYER_UNREVIEWED,
        ],
        default=None,
    )
    _add_data_dir(candidate_list_parser)

    candidate_validate_parser = subparsers.add_parser("knowledge-candidate-validate")
    candidate_validate_parser.add_argument("candidate_id")
    _add_data_dir(candidate_validate_parser)

    candidate_confirm_parser = subparsers.add_parser(
        "knowledge-candidate-confirm",
        help="Promote a pending_user_confirm candidate to validated after explicit user confirmation.",
    )
    candidate_confirm_parser.add_argument("candidate_id")
    candidate_confirm_parser.add_argument("--reviewer", default="local-user")
    candidate_confirm_parser.add_argument("--reason", default="user confirmed after review")
    _add_data_dir(candidate_confirm_parser)

    candidate_approve_parser = subparsers.add_parser("knowledge-candidate-approve")
    candidate_approve_parser.add_argument("candidate_id")
    candidate_approve_parser.add_argument("--reviewer", default="local-reviewer")
    candidate_approve_parser.add_argument("--reason", default="approved after human review")
    candidate_approve_parser.add_argument("--waive-eval-gate", action="store_true")
    _add_data_dir(candidate_approve_parser)

    candidate_deprecate_parser = subparsers.add_parser("knowledge-candidate-deprecate")
    candidate_deprecate_parser.add_argument("candidate_id")
    candidate_deprecate_parser.add_argument("--reason", required=True)
    _add_data_dir(candidate_deprecate_parser)

    ledger_record_parser = subparsers.add_parser("ledger-record")
    ledger_record_parser.add_argument("--type", default="manual")
    ledger_record_parser.add_argument("--subject", required=True)
    ledger_record_parser.add_argument("--status", default="recorded")
    ledger_record_parser.add_argument("--summary", required=True)
    ledger_record_parser.add_argument("--evidence-ref", action="append", default=[])
    ledger_record_parser.add_argument("--risk", default="")
    ledger_record_parser.add_argument("--rollback", default="")
    ledger_record_parser.add_argument("--metadata-json", default="{}")
    _add_data_dir(ledger_record_parser)

    ledger_list_parser = subparsers.add_parser("ledger-list")
    ledger_list_parser.add_argument("--type", default=None)
    ledger_list_parser.add_argument("--limit", type=_positive_int, default=20)
    _add_data_dir(ledger_list_parser)

    ledger_state_parser = subparsers.add_parser("ledger-state")
    ledger_state_parser.add_argument("--project-id", default="pydantic-ai-tech-intel-briefing")
    ledger_state_parser.add_argument("--limit", type=_positive_int, default=5)
    _add_data_dir(ledger_state_parser)

    gate_list_parser = subparsers.add_parser("gate-list")
    gate_list_parser.add_argument("--type", default=None)
    gate_list_parser.add_argument("--limit", type=_positive_int, default=50)
    _add_data_dir(gate_list_parser)

    trace_list_parser = subparsers.add_parser("trace-list")
    trace_list_parser.add_argument("--run-id", default=None)
    trace_list_parser.add_argument("--limit", type=_positive_int, default=50)
    _add_data_dir(trace_list_parser)

    checkpoint_list_parser = subparsers.add_parser("checkpoint-list")
    checkpoint_list_parser.add_argument("--run-id", default=None)
    checkpoint_list_parser.add_argument("--limit", type=_positive_int, default=50)
    _add_data_dir(checkpoint_list_parser)

    provider_health_parser = subparsers.add_parser("provider-health")
    provider_health_parser.add_argument("--run-id", default=None)
    provider_health_parser.add_argument("--limit", type=_positive_int, default=100)
    _add_data_dir(provider_health_parser)

    agentops_report_parser = subparsers.add_parser("agentops-report")
    agentops_report_parser.add_argument("--limit", type=_positive_int, default=20)
    _add_data_dir(agentops_report_parser)

    skill_parser = subparsers.add_parser("skill-draft")
    skill_parser.add_argument("title")
    _add_data_dir(skill_parser)

    promote_parser = subparsers.add_parser("skill-promote")
    promote_parser.add_argument("name_or_slug")
    _add_data_dir(promote_parser)

    skill_list_parser = subparsers.add_parser("skill-list")
    _add_data_dir(skill_list_parser)

    learning_loop_parser = subparsers.add_parser(
        "skill-learning-loop",
        help="Full learning loop: bad case -> staging draft -> eval-replay regression -> release gate -> audit.",
    )
    learning_loop_parser.add_argument(
        "--bad-case",
        default=None,
        help="Question id of the bad case to learn from; resolved against the evaluation report items.",
    )
    learning_loop_parser.add_argument(
        "--bad-case-json",
        default=None,
        help="Inline JSON bad-case record (id/question/quality_flags/root_causes/attribution_layer/...).",
    )
    learning_loop_parser.add_argument("--fix", default="", help="Description of the fix applied to the bad case.")
    learning_loop_parser.add_argument("--max-items", type=_positive_int, default=None)
    learning_loop_parser.add_argument("--judge", choices=["rule", "llm"], default="rule")
    learning_loop_parser.add_argument("--report", default=None, help="Path to evaluation-report.json (default: data-dir).")
    learning_loop_parser.add_argument("--audit-dir", default=None)
    _add_data_dir(learning_loop_parser)

    library_list_parser = subparsers.add_parser("skill-library-list", help="List active/staging/archive versioned skills.")
    _add_data_dir(library_list_parser)

    library_archive_parser = subparsers.add_parser("skill-library-archive", help="Deprecate an active skill into .archive/.")
    library_archive_parser.add_argument("name_or_slug")
    library_archive_parser.add_argument("--reason", default="")
    _add_data_dir(library_archive_parser)

    fixture_parser = subparsers.add_parser("feishu-fixture")
    fixture_parser.add_argument("path")
    _add_data_dir(fixture_parser)

    long_connection_parser = subparsers.add_parser("feishu-long-connection")
    _add_data_dir(long_connection_parser)

    status_parser = subparsers.add_parser("feishu-status")
    _add_data_dir(status_parser)

    reply_probe_parser = subparsers.add_parser("feishu-reply-probe")
    reply_probe_target = reply_probe_parser.add_mutually_exclusive_group(required=True)
    reply_probe_target.add_argument("--message-id")
    reply_probe_target.add_argument("--latest", action="store_true")
    reply_probe_parser.add_argument("--text", default="Search assistant Feishu reply probe.")
    _add_data_dir(reply_probe_parser)

    wait_reply_parser = subparsers.add_parser("feishu-wait-reply")
    wait_reply_parser.add_argument("--channel", default="feishu-long-connection")
    wait_reply_parser.add_argument("--timeout-seconds", type=float, default=120.0)
    wait_reply_parser.add_argument("--poll-seconds", type=float, default=2.0)
    _add_data_dir(wait_reply_parser)

    poll_once_parser = subparsers.add_parser("feishu-poll-once")
    poll_once_target = poll_once_parser.add_mutually_exclusive_group(required=True)
    poll_once_target.add_argument("--chat-id")
    poll_once_target.add_argument("--latest-chat", action="store_true")
    poll_once_parser.add_argument("--lookback-seconds", type=int, default=3600)
    _add_data_dir(poll_once_parser)

    poll_loop_parser = subparsers.add_parser("feishu-poll-loop")
    poll_loop_target = poll_loop_parser.add_mutually_exclusive_group(required=True)
    poll_loop_target.add_argument("--chat-id")
    poll_loop_target.add_argument("--latest-chat", action="store_true")
    poll_loop_parser.add_argument("--lookback-seconds", type=int, default=3600)
    poll_loop_parser.add_argument("--interval-seconds", type=float, default=5.0)
    poll_loop_parser.add_argument("--max-runs", type=int, default=None)
    _add_data_dir(poll_loop_parser)

    doctor_parser = subparsers.add_parser("doctor")
    _add_data_dir(doctor_parser)

    subparsers.add_parser("feishu-doctor")

    args = parser.parse_args(argv)
    if args.command == "feishu-doctor":
        report = run_feishu_diagnostics()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1

    data_dir = Path(args.data_dir) if args.data_dir else Settings.from_env().data_dir
    if args.command == "feishu-long-connection":
        run_long_connection(data_dir=data_dir)
        return 0

    store = _store(data_dir)

    if args.command == "ask":
        package = _answer_cli_question(store, args.question)
        print(json.dumps(package.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    if args.command == "topic-add":
        subscription = store.upsert_topic(args.user_id, args.chat_id, args.topic)
        print(json.dumps(subscription.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    if args.command == "topic-list":
        topics = store.list_topics(args.user_id, args.chat_id)
        print(json.dumps([topic.model_dump(mode="json") for topic in topics], ensure_ascii=False, indent=2))
        return 0
    if args.command == "source-contracts":
        print(json.dumps(source_contracts_as_dicts(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "topic-recipe-show":
        subscription = store.upsert_topic(args.user_id, args.chat_id, args.topic)
        print(
            json.dumps(
                {
                    "topic": subscription.topic,
                    "topic_id": subscription.id,
                    "source_recipe": source_recipe_summary(subscription.source_recipe),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "topic-recipe-set":
        subscription = store.upsert_topic(args.user_id, args.chat_id, args.topic)
        recipe = normalize_source_recipe(_parse_source_recipe(args.recipe_json))
        store.set_topic_source_recipe(subscription.id, recipe)
        refreshed = store.upsert_topic(args.user_id, args.chat_id, args.topic)
        print(
            json.dumps(
                {
                    "topic": refreshed.topic,
                    "topic_id": refreshed.id,
                    "source_recipe": source_recipe_summary(refreshed.source_recipe),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "brief-feedback":
        feedback_id = _briefing_service(store).add_feedback(
            args.topic,
            args.user_id,
            args.chat_id,
            args.body,
            args.url,
        )
        print(json.dumps({"feedback_id": feedback_id, "topic": args.topic}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "brief-run":
        briefing, path, artifact_path = _run_briefing(store, args.topic, args.user_id, args.chat_id)
        print(
            json.dumps(
                {
                    "path": str(path),
                    "artifact_path": str(artifact_path),
                    "sources": len(briefing.sources),
                    "source_candidates": len(briefing.source_candidates),
                    "provider_events": len(briefing.provider_events),
                    "topic": briefing.topic,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "brief-loop":
        result = _run_briefing_loop(
            store,
            args.topic,
            args.user_id,
            args.chat_id,
            args.interval_seconds,
            args.max_runs,
            topics=args.topics,
            topics_file=args.topics_file,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "feishu-status":
        print(json.dumps(_feishu_status(store), ensure_ascii=False, indent=2))
        return 0
    if args.command == "feishu-reply-probe":
        result = _run_feishu_reply_probe(
            store,
            message_id=args.message_id,
            use_latest=args.latest,
            text=args.text,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1
    if args.command == "feishu-wait-reply":
        result = _wait_for_reply_attempt(
            store,
            channel=args.channel,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1
    if args.command == "feishu-poll-once":
        result = _run_feishu_poll_once(
            store,
            chat_id=_resolve_poll_chat_id(store, args.chat_id, args.latest_chat),
            lookback_seconds=args.lookback_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("failed", 0) == 0 else 1
    if args.command == "feishu-poll-loop":
        result = _run_feishu_poll_loop(
            store,
            chat_id=_resolve_poll_chat_id(store, args.chat_id, args.latest_chat),
            lookback_seconds=args.lookback_seconds,
            interval_seconds=args.interval_seconds,
            max_runs=args.max_runs,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "doctor":
        report = run_readiness_diagnostics(store, data_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1
    if args.command == "report":
        markdown = ReportService(store, output_dir=data_dir / "reports").generate_markdown()
        path = data_dir / "reports" / "learning-report.md"
        print(str(path))
        if not markdown:
            return 1
        return 0
    if args.command == "candidate-list":
        candidates = store.list_source_candidates(topic_id=args.topic_id, status=args.status, limit=args.limit)
        print(json.dumps([candidate.model_dump(mode="json") for candidate in candidates], ensure_ascii=False, indent=2))
        return 0
    if args.command == "provider-trace-list":
        events = store.list_provider_trace_events(topic_id=args.topic_id, run_id=args.run_id, limit=args.limit)
        print(json.dumps([event.model_dump(mode="json") for event in events], ensure_ascii=False, indent=2))
        return 0
    if args.command == "memory-layer-list":
        items = store.list_layered_memory_items(
            layer=args.layer,
            user_id=args.user_id,
            chat_id=args.chat_id,
            limit=args.limit,
        )
        print(json.dumps([item.model_dump(mode="json") for item in items], ensure_ascii=False, indent=2))
        return 0
    if args.command == "evolve":
        result = _run_evolution(store, data_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evolve-loop":
        result = _run_evolution_loop(
            store,
            data_dir,
            interval_seconds=args.interval_seconds,
            max_runs=args.max_runs,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evolution-offline-run":
        result = _run_offline_evolution(
            store,
            data_dir,
            judge_name=args.judge,
            max_trajectories=args.max_trajectories,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1
    if args.command == "evolution-judge-calibrate":
        result = _run_judge_calibration(store, data_dir, judge_name=args.judge)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "eval-suite":
        result = _run_evaluation(
            store,
            data_dir,
            questions_source=args.questions,
            max_questions=args.max_questions,
            judge_name=args.judge,
            pool=args.pool,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "eval-replay":
        result = _run_evaluation_replay(
            store,
            data_dir,
            report_path=Path(args.report) if args.report else None,
            max_items=args.max_items,
            judge_name=args.judge,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1
    if args.command == "evidence-backfill":
        result = VerificationBackfillService(store).run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "knowledge-graph-seed":
        result = _graph_service(store).seed_default_graphs(args.domain)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "knowledge-graph-list":
        result = _graph_service(store).list_graphs()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "knowledge-graph-query":
        hits = _graph_service(store).query(args.query, domain_id=args.domain, limit=args.limit)
        print(json.dumps([hit.model_dump(mode="json") for hit in hits], ensure_ascii=False, indent=2))
        return 0
    if args.command == "knowledge-graph-export":
        markdown = _graph_service(store).export_markdown(args.domain)
        output_path = Path(args.output) if args.output else data_dir / "knowledge-graphs" / f"{args.domain}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(
            json.dumps(
                {
                    "domain": args.domain,
                    "path": str(output_path),
                    "characters": len(markdown),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "knowledge-candidate-list":
        result = _candidate_service(store).list_candidates(status=args.status, layer=args.layer)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "knowledge-candidate-validate":
        result = _candidate_service(store).validate(args.candidate_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["validated"] else 1
    if args.command == "knowledge-candidate-confirm":
        result = _candidate_service(store).confirm_candidate(
            args.candidate_id,
            reviewer=args.reviewer,
            reason=args.reason,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("confirmed") else 1
    if args.command == "knowledge-candidate-approve":
        result = _candidate_service(store).approve(
            args.candidate_id,
            reviewer=args.reviewer,
            reason=args.reason,
            eval_gate=_waived_evaluation_gate(data_dir) if args.waive_eval_gate else _evaluation_gate_from_report(data_dir),
            require_eval_gate=not args.waive_eval_gate,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["approved"] else 1
    if args.command == "knowledge-candidate-deprecate":
        _candidate_service(store).deprecate(args.candidate_id, args.reason)
        print(json.dumps({"deprecated": True, "candidate_id": args.candidate_id}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "ledger-record":
        ledger_id = store.add_project_ledger_entry(
            entry_type=args.type,
            subject=args.subject,
            status=args.status,
            summary=args.summary,
            evidence_refs=list(args.evidence_ref),
            risk=args.risk,
            rollback=args.rollback,
            metadata=_parse_metadata_json(args.metadata_json),
        )
        print(json.dumps({"ledger_id": ledger_id}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "ledger-list":
        print(
            json.dumps(
                store.list_project_ledger_entries(entry_type=args.type, limit=args.limit),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "ledger-state":
        print(
            json.dumps(
                {
                    "latest": store.latest_project_ledger_snapshot(args.project_id),
                    "snapshots": store.list_project_ledger_snapshots(args.project_id, limit=args.limit),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "gate-list":
        print(json.dumps(store.list_gate_records(gate_type=args.type, limit=args.limit), ensure_ascii=False, indent=2))
        return 0
    if args.command == "trace-list":
        print(json.dumps(store.list_trace_events(run_id=args.run_id, limit=args.limit), ensure_ascii=False, indent=2))
        return 0
    if args.command == "checkpoint-list":
        print(
            json.dumps(
                store.list_run_checkpoints(run_id=args.run_id, limit=args.limit),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "provider-health":
        print(
            json.dumps(
                {
                    "summary": store.search_provider_health_summary(limit=args.limit),
                    "records": store.list_search_provider_health(run_id=args.run_id, limit=args.limit),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "agentops-report":
        print(json.dumps(_agentops_report(store, limit=args.limit), ensure_ascii=False, indent=2))
        return 0
    if args.command == "skill-draft":
        path = SkillDraftService(store, drafts_dir=data_dir / "skills" / "drafts").create_from_experience(
            args.title,
            source_ids=_skill_source_ids(store),
        )
        print(path)
        return 0
    if args.command == "skill-promote":
        service = SkillDraftService(
            store,
            drafts_dir=data_dir / "skills" / "drafts",
            active_dir=data_dir / "skills" / "active",
        )
        path = service.promote(args.name_or_slug)
        draft = _skill_draft_by_path(store, path)
        print(
            json.dumps(
                {
                    "status": "promoted",
                    "name": draft["name"] if draft else args.name_or_slug,
                    "path": path,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "skill-list":
        result = SkillDraftService(
            store,
            drafts_dir=data_dir / "skills" / "drafts",
            active_dir=data_dir / "skills" / "active",
        ).list_review_status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "skill-learning-loop":
        result = _run_skill_learning_loop(store, data_dir, args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if (result.get("release") or {}).get("decision", {}).get("approved") else 1
    if args.command == "skill-library-list":
        print(json.dumps(_skill_library_inventory(data_dir), ensure_ascii=False, indent=2))
        return 0
    if args.command == "skill-library-archive":
        record = _skill_library_archive(data_dir, args.name_or_slug, reason=args.reason)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    if args.command == "feishu-fixture":
        reply = _run_feishu_fixture(store, Path(args.path))
        print(json.dumps(reply, ensure_ascii=False, indent=2))
        return 0
    return 2


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=None)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _store(data_dir: Path) -> MemoryStore:
    store = MemoryStore(data_dir / "assistant.sqlite3")
    store.initialize()
    return store


def _candidate_service(store: MemoryStore) -> DomainKnowledgeCandidateService:
    settings = Settings.from_env()
    return DomainKnowledgeCandidateService(
        store,
        embedding_provider=embedding_provider_from_settings(settings),
        llm_runner=_build_semantic_llm_runner(),
    )


def _build_semantic_llm_runner():
    """LLM runner for semantic candidate classification, mirroring the judge
    runner convention so it can be mocked in tests."""
    runtime = runtime_from_settings(Settings.from_env())

    def runner(instructions: str, payload: dict[str, Any]) -> str:
        return runtime.run_text_task(
            instructions,
            payload,
            temperature=0.0,
            max_tokens=400,
        )

    return runner


def _graph_service(store: MemoryStore) -> DomainKnowledgeGraphService:
    settings = Settings.from_env()
    return DomainKnowledgeGraphService(
        store,
        embedding_provider=embedding_provider_from_settings(settings),
    )


def _briefing_service(store: MemoryStore) -> DailyBriefingService:
    settings = Settings.from_env()
    return DailyBriefingService(
        store=store,
        search_client=search_client_from_settings(settings),
        runtime=runtime_from_settings(settings),
        max_queries=settings.briefing_max_queries,
        results_per_query=settings.briefing_results_per_query,
        max_sources=settings.briefing_max_sources,
        model_max_sources=settings.briefing_model_max_sources,
        search_budget_seconds=settings.briefing_search_budget_seconds,
        timezone_name=settings.briefing_timezone,
        embedding_provider=embedding_provider_from_settings(settings),
    )


def _run_briefing(store: MemoryStore, topic: str, user_id: str, chat_id: str):
    briefing = _briefing_service(store).run(topic, user_id, chat_id)
    output_dir = _store_data_dir(store) / "briefs"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{briefing.run_date.isoformat()}-{_safe_filename(topic)}.md"
    artifact_path = output_dir / f"{briefing.run_date.isoformat()}-{_safe_filename(topic)}.json"
    path.write_text(briefing.markdown, encoding="utf-8")
    artifact_path.write_text(
        json.dumps(briefing.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return briefing, path, artifact_path


def _topic_pool(primary: str, topics: str | None, topics_file: str | None) -> list[str]:
    """Resolve the briefing-loop topic pool.

    ``primary`` is always kept; ``--topics`` adds comma-separated topics and
    ``--topics-file`` adds one topic per line.  Duplicates are removed while
    preserving order.
    """
    pool: list[str] = [primary]
    if topics:
        pool.extend(part.strip() for part in topics.split(",") if part.strip())
    if topics_file:
        path = Path(topics_file)
        pool.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    seen: set[str] = set()
    unique: list[str] = []
    for topic in pool:
        if topic not in seen:
            seen.add(topic)
            unique.append(topic)
    return unique


def _run_briefing_loop(
    store: MemoryStore,
    topic: str,
    user_id: str,
    chat_id: str,
    interval_seconds: float,
    max_runs: int | None,
    topics: str | None = None,
    topics_file: str | None = None,
) -> dict[str, object]:
    pool = _topic_pool(topic, topics, topics_file)
    metadata: dict[str, object] = {
        "topic": topic,
        "topic_pool": pool,
        "user_id": user_id,
        "chat_id": chat_id,
        "interval_seconds": interval_seconds,
        "max_runs": max_runs,
    }
    session_id = store.record_runtime_session("brief-loop", _current_process_id(), "started", metadata)
    results: list[dict[str, object]] = []
    runs = 0
    try:
        while max_runs is None or runs < max_runs:
            current_topic = pool[runs % len(pool)]
            briefing, path, artifact_path = _run_briefing(store, current_topic, user_id, chat_id)
            results.append(
                {
                    "topic": current_topic,
                    "path": str(path),
                    "artifact_path": str(artifact_path),
                    "sources": len(briefing.sources),
                    "source_candidates": len(briefing.source_candidates),
                    "provider_events": len(briefing.provider_events),
                    "run_date": briefing.run_date.isoformat(),
                }
            )
            runs += 1
            if max_runs is not None and runs >= max_runs:
                break
            time.sleep(max(1.0, interval_seconds))
    finally:
        metadata["runs"] = runs
        store.update_runtime_session_status(session_id, "completed", metadata)
    return {"runtime_session_id": session_id, "runs": runs, "topic_pool": pool, "results": results}


def _safe_filename(value: str) -> str:
    slug = "-".join(part for part in "".join(char if char.isalnum() else " " for char in value).split())
    return slug[:80] or "briefing"


def _answer_cli_question(store: MemoryStore, question: str):
    settings = Settings.from_env()
    data_dir = _store_data_dir(store)
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime_from_settings(settings),
        search_client=search_client_from_settings(settings),
        search_budget_seconds=settings.workflow_search_budget_seconds,
        skill_drafts_dir=data_dir / "skills" / "drafts",
        report_output_dir=data_dir / "reports",
        admin_user_ids=set(settings.admin_user_ids),
    )
    message = IncomingMessage(
        message_id=f"cli_{uuid.uuid4().hex}",
        event_id=None,
        user_id="local-user",
        chat_id="local-cli",
        text=question,
        source="cli",
    )
    package = workflow.answer(message)
    ProfileService(store).update_from_answer(package)
    return package


def _run_feishu_fixture(store: MemoryStore, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    client = FakeFeishuClient()
    settings = Settings.from_env()
    data_dir = _store_data_dir(store)
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime_from_settings(settings),
        search_client=search_client_from_settings(settings),
        search_budget_seconds=settings.workflow_search_budget_seconds,
        skill_drafts_dir=data_dir / "skills" / "drafts",
        report_output_dir=data_dir / "reports",
        admin_user_ids=set(settings.admin_user_ids),
    )
    message = parse_feishu_event(payload)
    package = workflow.answer(message)
    ProfileService(store).update_from_answer(package)
    return client.reply_text(message.message_id, package.answer_text)


def _build_quality_judge(judge_name: str):
    """Build a quality judge (rule or LLM rubric) for evaluation commands."""
    from search_assistant.evolution.verification import LLMRubricJudge, RuleQualityJudge

    if judge_name == "rule":
        return RuleQualityJudge()
    runtime = runtime_from_settings(Settings.from_env())

    def _llm_judge_runner(instructions: str, payload: dict[str, Any]) -> str:
        return runtime.run_text_task(
            instructions,
            payload,
            temperature=0.0,
            max_tokens=1400,
        )

    return LLMRubricJudge(judge_runner=_llm_judge_runner)


def _run_evaluation(
    store: MemoryStore,
    data_dir: Path,
    questions_source: str | None = None,
    max_questions: int | None = None,
    judge_name: str = "rule",
    pool: str | None = None,
) -> dict[str, Any]:
    settings = Settings.from_env()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime_from_settings(settings),
        search_client=search_client_from_settings(settings),
        search_budget_seconds=settings.workflow_search_budget_seconds,
        skill_drafts_dir=data_dir / "skills" / "drafts",
        report_output_dir=data_dir / "reports",
        admin_user_ids=set(settings.admin_user_ids),
    )
    service = EvaluationService(
        store=store,
        workflow=workflow,
        output_dir=data_dir / "evaluations",
        report_output_dir=data_dir / "reports",
        judge=_build_quality_judge(judge_name),
    )
    return service.run(
        _load_evaluation_questions(questions_source) if questions_source else None,
        max_questions=max_questions,
        tier=pool,
    )


def _verdict_rank(verdict: str) -> int:
    """Rank rubric verdicts: pass > uncertain > fail."""
    return {"pass": 2, "uncertain": 1, "fail": 0}.get(str(verdict).lower(), 1)


def _judge_verdicts(judge: Any, question: str, answer: str, source_urls: list[str], unverified: list[str]) -> dict[str, str]:
    """Run a quality judge over one answer and return per-dimension verdicts."""
    payload = {
        "question": question,
        "classification": "research",
        "final_answer": answer,
        "review": {"ran": True, "approved": True, "issues": [], "revision": answer},
        "unverified_claims": unverified or [],
    }
    sources = [{"title": "", "url": url, "snippet": ""} for url in source_urls if str(url).startswith("http")]
    judged = judge.judge(payload, sources)
    dimensions = judged.get("dimensions") or {}
    return {str(dim): str(value.get("verdict", "uncertain")) for dim, value in dimensions.items()}


def _answer_body(text: str) -> str:
    """Strip the appended search-record section before judging an answer."""
    marker = "\n搜索记录:"
    if marker in text:
        return text.split(marker, maxsplit=1)[0].strip()
    return text.strip()


def _mcnemar_p_value(improved: int, regressed: int) -> float:
    """Two-sided exact McNemar test on the discordant pair (improved, regressed).

    With no discordant pairs the p-value is 1.0; otherwise it is twice the
    lower tail of the binomial(n, 0.5) distribution, clamped to 1.0.
    """
    n = improved + regressed
    if n == 0:
        return 1.0
    k = min(improved, regressed)
    p = 2.0 * sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, p)


def _run_evaluation_replay(
    store: MemoryStore,
    data_dir: Path,
    report_path: Path | None = None,
    max_items: int | None = None,
    judge_name: str = "rule",
) -> dict[str, Any]:
    source_path = report_path or data_dir / "evaluations" / "evaluation-report.json"
    if not source_path.exists():
        return {
            "ok": False,
            "reason": "evaluation report not found",
            "report_path": str(source_path),
            "hint": "Run eval-suite first, then rerun eval-replay.",
        }
    report = json.loads(source_path.read_text(encoding="utf-8-sig"))
    items = report.get("items", [])
    if not isinstance(items, list) or not items:
        return {
            "ok": False,
            "reason": "evaluation report has no replayable items",
            "report_path": str(source_path),
        }
    selected_items = items[:max_items] if max_items is not None else items
    settings = Settings.from_env()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime_from_settings(settings),
        search_client=search_client_from_settings(settings),
        search_budget_seconds=settings.workflow_search_budget_seconds,
        skill_drafts_dir=data_dir / "skills" / "drafts",
        report_output_dir=data_dir / "reports",
        admin_user_ids=set(settings.admin_user_ids),
    )
    judge = _build_quality_judge(judge_name)

    replay_items: list[dict[str, Any]] = []
    failures = 0
    changed_answers = 0
    changed_source_sets = 0
    improved_questions = 0
    regressed_questions = 0
    stable_questions = 0
    for index, previous in enumerate(selected_items, start=1):
        question = str(previous.get("question") or "").strip()
        if not question:
            failures += 1
            replay_items.append(
                {
                    "index": index,
                    "ok": False,
                    "reason": "missing question",
                    "previous_question_id": previous.get("question_id"),
                }
            )
            continue
        try:
            message = IncomingMessage(
                message_id=f"eval_replay_{uuid.uuid4().hex}",
                event_id=None,
                user_id="evaluation-replay-user",
                chat_id="evaluation-replay",
                text=question,
                source="evaluation-replay",
            )
            package = workflow.answer(message)
            ProfileService(store).update_from_answer(package)
        except Exception as exc:
            failures += 1
            replay_items.append(
                {
                    "index": index,
                    "ok": False,
                    "question": question,
                    "previous_question_id": previous.get("question_id"),
                    "error": _safe_cli_error(exc),
                }
            )
            continue

        previous_answer = str(previous.get("answer_excerpt") or "").strip()
        current_answer = _preview_text(package.answer_text, max_chars=1400)
        previous_source_urls = sorted(str(url) for url in previous.get("source_urls", []) if str(url).strip())
        current_source_urls = sorted(source.url for source in package.sources)
        answer_changed = previous_answer != current_answer
        source_set_changed = previous_source_urls != current_source_urls
        changed_answers += 1 if answer_changed else 0
        changed_source_sets += 1 if source_set_changed else 0

        # Quality delta: judge the previous and current answers on the same
        # rubric and compare verdicts per dimension.
        previous_verdicts = _judge_verdicts(
            judge,
            question,
            _answer_body(previous_answer),
            previous_source_urls,
            [str(claim) for claim in previous.get("unverified_claim_samples", [])],
        )
        current_verdicts = _judge_verdicts(
            judge,
            question,
            _answer_body(_preview_text(package.answer_text, max_chars=4000)),
            current_source_urls,
            [str(claim) for claim in package.unverified_claims],
        )
        improved = [
            dim
            for dim in current_verdicts
            if _verdict_rank(current_verdicts[dim]) > _verdict_rank(previous_verdicts.get(dim, "uncertain"))
        ]
        regressed = [
            dim
            for dim in current_verdicts
            if _verdict_rank(current_verdicts[dim]) < _verdict_rank(previous_verdicts.get(dim, "uncertain"))
        ]
        stable = [dim for dim in current_verdicts if dim not in improved and dim not in regressed]
        quality_delta = {
            "improved": improved,
            "regressed": regressed,
            "stable": stable,
            "previous_verdicts": previous_verdicts,
            "current_verdicts": current_verdicts,
        }
        if improved and not regressed:
            improved_questions += 1
        elif regressed and not improved:
            regressed_questions += 1
        else:
            stable_questions += 1
        replay_items.append(
            {
                "index": index,
                "ok": True,
                "question": question,
                "previous_question_id": previous.get("question_id"),
                "current_question_id": package.question_id,
                "answer_changed": answer_changed,
                "source_set_changed": source_set_changed,
                "previous_source_count": len(previous_source_urls),
                "current_source_count": len(current_source_urls),
                "previous_quality_flags": previous.get("quality_flags", []),
                "current_confidence": package.confidence,
                "quality_delta": quality_delta,
            }
        )

    p_value = _mcnemar_p_value(improved_questions, regressed_questions)
    significance = {
        "improved_questions": improved_questions,
        "regressed_questions": regressed_questions,
        "stable_questions": stable_questions,
        "mcnemar_p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }
    output_dir = data_dir / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "evaluation-replay-report.json"
    result = {
        "ok": failures == 0,
        "judge": judge_name,
        "source_report_path": str(source_path),
        "replay_report_path": str(replay_path),
        "requested_items": len(selected_items),
        "replayed": sum(1 for item in replay_items if item.get("ok")),
        "failed": failures,
        "answer_changed": changed_answers,
        "source_set_changed": changed_source_sets,
        "quality_delta_summary": significance,
        "items": replay_items,
    }
    replay_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    gate_id = store.add_gate_record(
        gate_type="evaluation_replay",
        subject_type="evaluation_report",
        subject_id=str(source_path),
        result="passed" if result["ok"] else "failed",
        reason="replayed all recorded evaluation questions" if result["ok"] else "one or more replay items failed",
        evidence_refs=[str(source_path), str(replay_path)],
        metadata={
            "requested_items": result["requested_items"],
            "replayed": result["replayed"],
            "answer_changed": result["answer_changed"],
            "source_set_changed": result["source_set_changed"],
            "quality_delta": significance,
            "judge": judge_name,
        },
    )
    store.add_project_ledger_entry(
        entry_type="evaluation_replay",
        subject=str(source_path),
        status="completed" if result["ok"] else "failed",
        summary=f"Replayed {result['replayed']} of {result['requested_items']} evaluation questions with the {judge_name} judge.",
        evidence_refs=[gate_id, str(replay_path)],
        risk="Replay quality deltas come from the configured judge; human review is still needed for semantic drift.",
        rollback="Rerun eval-suite and eval-replay in an isolated data directory after fixing failures.",
        metadata={
            "answer_changed": result["answer_changed"],
            "source_set_changed": result["source_set_changed"],
            "quality_delta": significance,
            "failed": result["failed"],
            "judge": judge_name,
        },
    )
    return result


def _evaluation_gate_from_report(data_dir: Path) -> dict[str, Any]:
    report_path = data_dir / "evaluations" / "evaluation-report.json"
    if not report_path.exists():
        return {
            "passed": False,
            "reason": "evaluation report is missing; run eval-suite before approving knowledge candidates",
            "report_path": str(report_path),
        }
    try:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "reason": f"evaluation report is not valid JSON: {exc.msg}",
            "report_path": str(report_path),
        }
    items = report.get("items", [])
    summary = report.get("summary", {})
    if not isinstance(items, list) or not items:
        return {
            "passed": False,
            "reason": "evaluation report has no completed items",
            "report_path": str(report_path),
        }
    flagged = int(summary.get("flagged_answers", 0) or 0)
    rejected = int(summary.get("review_rejected_answers", 0) or 0)
    result_failed = int(summary.get("result_failed_answers", 0) or 0)
    process_flagged = int(summary.get("process_flagged_answers", 0) or 0)
    passed = flagged == 0 and rejected == 0 and result_failed == 0 and process_flagged == 0
    blocking = {
        "flagged_answers": flagged,
        "review_rejected_answers": rejected,
        "result_failed_answers": result_failed,
        "process_flagged_answers": process_flagged,
    }
    return {
        "passed": passed,
        "reason": "evaluation report passed release gate"
        if passed
        else "evaluation report has blocking quality/process findings",
        "report_path": str(report_path),
        "summary": blocking,
        "total_questions": report.get("total_questions", len(items)),
    }


def _waived_evaluation_gate(data_dir: Path) -> dict[str, Any]:
    report_path = data_dir / "evaluations" / "evaluation-report.json"
    return {
        "passed": False,
        "waived": True,
        "reason": "operator explicitly waived evaluation gate for a local experiment",
        "report_path": str(report_path),
    }


def _load_evaluation_questions(source: str | Path) -> list[str]:
    source_text = str(source).strip()
    if source_text.startswith("["):
        data = json.loads(source_text)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("Evaluation questions JSON must be a list of strings")
        return [item.strip() for item in data if item.strip()]

    path = Path(source)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("Evaluation questions JSON must be a list of strings")
        return [item.strip() for item in data if item.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_source_recipe(recipe_json: str) -> dict[str, float]:
    raw = json.loads(recipe_json)
    if not isinstance(raw, dict):
        raise ValueError("source recipe must be a JSON object, for example: {\"general-web\": 3, \"bilibili\": 1}")
    parsed: dict[str, float] = {}
    for key, value in raw.items():
        try:
            parsed[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"source recipe weight for {key!r} must be a number") from exc
    return parsed


def _feishu_status(store: MemoryStore) -> dict[str, Any]:
    return {
        "counts": store.diagnostic_counts(),
        "latest_interactions": [_safe_interaction(row) for row in store.list_interactions(limit=5)],
        "latest_reply_attempts": [_safe_reply_attempt(row) for row in store.list_reply_attempts(limit=5)],
        "latest_runtime_sessions": store.list_runtime_sessions(limit=5),
    }


def _safe_interaction(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("text") or "")
    return {
        "question_id": row.get("id"),
        "message_id": row.get("message_id"),
        "event_id": row.get("event_id"),
        "source": row.get("source"),
        "chat_id": row.get("chat_id"),
        "answer_id": row.get("answer_id"),
        "created_at": row.get("created_at"),
        "text_preview": _preview_text(text),
    }


def _preview_text(text: str, max_chars: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _parse_metadata_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("--metadata-json must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--metadata-json must be a JSON object")
    return parsed


def _agentops_report(store: MemoryStore, limit: int = 20) -> dict[str, Any]:
    gates = store.list_gate_records(limit=limit)
    traces = store.list_trace_events(limit=limit)
    checkpoints = store.list_run_checkpoints(limit=limit)
    ledger_entries = store.list_project_ledger_entries(limit=limit)
    provider_health = store.search_provider_health_summary(limit=limit * 5)
    return {
        "counts": store.diagnostic_counts(),
        "dependency_lock": _dependency_lock_status(Path.cwd()),
        "latest_project_state": store.latest_project_ledger_snapshot("pydantic-ai-tech-intel-briefing"),
        "latest_project_ledger": ledger_entries,
        "gate_summary": {
            "total": len(gates),
            "passed": sum(1 for item in gates if item["result"] == "passed"),
            "failed": sum(1 for item in gates if item["result"] == "failed"),
            "waived": sum(1 for item in gates if item["result"] == "waived"),
            "latest": gates,
        },
        "trace_summary": {
            "total_sampled": len(traces),
            "failed": sum(1 for item in traces if item["status"] == "failed"),
            "latest": traces,
        },
        "checkpoint_summary": {
            "total_sampled": len(checkpoints),
            "failed": sum(1 for item in checkpoints if item["status"] == "failed"),
            "latest": checkpoints,
        },
        "provider_health": provider_health,
    }


def _dependency_lock_status(root: Path) -> dict[str, Any]:
    constraints_path = root / "constraints-dev.txt"
    pyproject_path = root / "pyproject.toml"
    status: dict[str, Any] = {
        "constraints_path": str(constraints_path),
        "constraints_present": constraints_path.exists(),
        "pyproject_present": pyproject_path.exists(),
    }
    if constraints_path.exists():
        pinned = [
            line.strip()
            for line in constraints_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        status["pinned_packages"] = len(pinned)
        status["has_pydantic_ai_slim_pin"] = any(line.lower().startswith("pydantic-ai-slim==") for line in pinned)
    else:
        status["pinned_packages"] = 0
        status["has_pydantic_ai_slim_pin"] = False
    status["ok"] = bool(status["constraints_present"] and status["has_pydantic_ai_slim_pin"])
    return status


def _safe_reply_attempt(row: dict[str, Any]) -> dict[str, Any]:
    response_json = row.get("response_json") or "{}"
    return {
        "id": row.get("id"),
        "question_id": row.get("question_id"),
        "message_id": row.get("message_id"),
        "channel": row.get("channel"),
        "status": row.get("status"),
        "error": row.get("error"),
        "created_at": row.get("created_at"),
        "response_recorded": response_json not in {"", "{}"},
    }


def _run_feishu_reply_probe(
    store: MemoryStore,
    message_id: str | None,
    use_latest: bool,
    text: str,
) -> dict[str, Any]:
    question_id = "manual-probe"
    resolved_message_id = message_id
    if use_latest:
        interactions = store.list_interactions(limit=1)
        if not interactions:
            return {
                "ok": False,
                "detail": "no interaction exists for --latest",
            }
        latest = interactions[0]
        question_id = latest["id"]
        resolved_message_id = latest["message_id"]
    if not resolved_message_id:
        return {"ok": False, "detail": "message id is required"}

    try:
        response = feishu_client_from_settings(Settings.from_env()).reply_text(resolved_message_id, text)
    except Exception as exc:
        store.record_reply_attempt(
            question_id=question_id,
            message_id=resolved_message_id,
            channel="feishu-probe",
            status="failure",
            error=_safe_cli_error(exc),
        )
        return {
            "ok": False,
            "question_id": question_id,
            "message_id": resolved_message_id,
            "channel": "feishu-probe",
            "error": _safe_cli_error(exc),
        }

    store.record_reply_attempt(
        question_id=question_id,
        message_id=resolved_message_id,
        channel="feishu-probe",
        status="success",
        response=response,
    )
    return {
        "ok": True,
        "question_id": question_id,
        "message_id": resolved_message_id,
        "channel": "feishu-probe",
    }


def _wait_for_reply_attempt(
    store: MemoryStore,
    channel: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    not_before = _latest_started_runtime_created_at(store, channel)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        matching = [
            attempt
            for attempt in store.list_reply_attempts()
            if attempt["channel"] == channel and attempt["status"] == "success"
            if not_before is None or str(attempt.get("created_at", "")) > not_before
        ]
        if matching:
            return {
                "ok": True,
                "channel": channel,
                "not_before": not_before,
                "attempt": _safe_reply_attempt(matching[0]),
            }

        latest_attempts = store.list_reply_attempts(limit=1)
        if time.monotonic() >= deadline:
            return {
                "ok": False,
                "channel": channel,
                "detail": f"timed out waiting for successful {channel} reply attempt",
                "not_before": not_before,
                "latest_attempt": _safe_reply_attempt(latest_attempts[0]) if latest_attempts else None,
            }
        time.sleep(max(0.1, poll_seconds))


def _latest_started_runtime_created_at(store: MemoryStore, kind: str) -> str | None:
    for session in store.list_runtime_sessions():
        if session.get("kind") == kind and session.get("status") == "started":
            return str(session.get("created_at") or "")
    return None


def _resolve_poll_chat_id(store: MemoryStore, chat_id: str | None, latest_chat: bool) -> str:
    if chat_id:
        return chat_id
    if latest_chat:
        interactions = [
            row
            for row in store.list_interactions()
            if str(row.get("source", "")).startswith("feishu-") and str(row.get("chat_id", "")).startswith("oc_")
        ]
        if not interactions:
            raise RuntimeError("No stored interaction exists for --latest-chat")
        return str(interactions[0]["chat_id"])
    raise RuntimeError("chat id is required")


def _run_feishu_poll_once(store: MemoryStore, chat_id: str, lookback_seconds: int) -> dict[str, Any]:
    settings = Settings.from_env()
    from search_assistant.feishu.polling import FeishuPollingService

    service = FeishuPollingService(
        store=store,
        runtime=runtime_from_settings(settings),
        search_client=search_client_from_settings(settings),
        feishu_client=feishu_client_from_settings(settings),
        search_budget_seconds=settings.workflow_search_budget_seconds,
        admin_user_ids=set(settings.admin_user_ids),
    )
    return service.poll_once(chat_id=chat_id, lookback_seconds=lookback_seconds)


def _run_feishu_poll_loop(
    store: MemoryStore,
    chat_id: str,
    lookback_seconds: int,
    interval_seconds: float,
    max_runs: int | None,
) -> dict[str, Any]:
    runs = 0
    results: list[dict[str, Any]] = []
    while max_runs is None or runs < max_runs:
        results.append(_run_feishu_poll_once(store, chat_id=chat_id, lookback_seconds=lookback_seconds))
        runs += 1
        if max_runs is not None and runs >= max_runs:
            break
        time.sleep(interval_seconds)
    return {"runs": runs, "chat_id": chat_id, "results": results}


def feishu_client_from_settings(settings: Settings):
    return FeishuHttpClient(settings.feishu_app_id, settings.feishu_app_secret)


def _safe_cli_error(exc: Exception) -> str:
    text = str(exc)
    for marker in ("tenant_access_token", "Authorization", "app_secret"):
        text = text.replace(marker, "[redacted]")
    return text


def _store_data_dir(store: MemoryStore) -> Path:
    return Path(store.database_path).parent


def _skill_source_ids(store: MemoryStore) -> list[str]:
    profiles = store.list_profile_snapshots()
    if profiles:
        return [profiles[-1]["id"]]
    answers = store.list_answers()
    if answers:
        return [answers[-1].question_id]
    return ["manual"]


def _skill_draft_by_path(store: MemoryStore, active_path: str) -> dict[str, Any] | None:
    active_slug = Path(active_path).parent.name
    for draft in store.list_skill_drafts():
        if Path(str(draft["path"])).parent.name == active_slug:
            return draft
    return None


def _run_skill_learning_loop(store: MemoryStore, data_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Run the Path-B learning loop: bad case -> staging -> regression -> gate -> audit."""
    from search_assistant.skills.audit import SkillLearningAudit
    from search_assistant.skills.learning import SkillLearningTrigger
    from search_assistant.skills.library import VersionedSkillLibrary
    from search_assistant.skills.loop import SkillLearningLoop
    from search_assistant.skills.regression import SkillRegressionHook
    from search_assistant.skills.release import SkillReleaseGate

    report_path = Path(args.report) if args.report else data_dir / "evaluations" / "evaluation-report.json"
    previous_items = _evaluation_report_items(report_path)
    bad_case = _resolve_learning_bad_case(args, previous_items)

    skills_root = data_dir / "skills"
    library = VersionedSkillLibrary(skills_root)
    audit_dir = Path(args.audit_dir) if args.audit_dir else skills_root / "audits"
    trigger = SkillLearningTrigger(library, llm_runner=_build_learning_llm_runner())
    regression_hook = SkillRegressionHook(judge=_build_quality_judge(args.judge))
    loop = SkillLearningLoop(
        store=store,
        library=library,
        audit=SkillLearningAudit(audit_dir),
        trigger=trigger,
        regression_hook=regression_hook,
        release_gate=SkillReleaseGate(store, library),
    )
    return loop.run_cycle(
        bad_case=bad_case,
        previous_items=previous_items,
        fix_note=args.fix,
        max_items=args.max_items,
        notes=(
            f"Triggered by CLI skill-learning-loop with judge={args.judge}; "
            f"bad_case={bad_case.get('id') or bad_case.get('question_id') or 'unknown'}."
        ),
    )


def _build_learning_llm_runner():
    """LLM runner for skill distillation, mirroring the judge runner convention."""
    runtime = runtime_from_settings(Settings.from_env())

    def runner(instructions: str, payload: dict[str, Any]) -> str:
        return runtime.run_text_task(
            instructions,
            payload,
            temperature=0.2,
            max_tokens=1400,
        )

    return runner


def _evaluation_report_items(report_path: Path) -> list[dict[str, Any]]:
    if not report_path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {report_path} (run eval-suite first)")
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    items = report.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Evaluation report has no items list: {report_path}")
    return [item for item in items if isinstance(item, dict) and str(item.get("question") or "").strip()]


def _resolve_learning_bad_case(args: argparse.Namespace, previous_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the bad case: explicit --bad-case-json wins, then --bad-case (question id),
    then the first item with quality flags."""
    if args.bad_case_json:
        raw = json.loads(args.bad_case_json)
        if not isinstance(raw, dict):
            raise ValueError("--bad-case-json must be a JSON object")
        return raw
    if args.bad_case:
        question_id = str(args.bad_case)
        for item in previous_items:
            if str(item.get("question_id") or "") == question_id or str(item.get("index") or "") == question_id:
                return _bad_case_from_item(item)
        raise FileNotFoundError(f"No evaluation item matches bad case {question_id!r}")
    for item in previous_items:
        if item.get("quality_flags"):
            return _bad_case_from_item(item)
    if previous_items:
        return _bad_case_from_item(previous_items[0])
    raise FileNotFoundError("No evaluation items available to derive a bad case; run eval-suite first")


def _bad_case_from_item(item: dict[str, Any]) -> dict[str, Any]:
    diagnosis = item.get("diagnosis") or {}
    return {
        "id": str(item.get("question_id") or "unknown"),
        "question": str(item.get("question") or ""),
        "tier": str(item.get("tier") or item.get("classification") or ""),
        "quality_flags": [str(flag) for flag in (item.get("quality_flags") or [])],
        "root_causes": [str(cause) for cause in (diagnosis.get("root_causes") or [])],
        "attribution_layer": "prompt/skill",
        "answer_excerpt": str(item.get("answer_excerpt") or ""),
    }


def _skill_library_inventory(data_dir: Path) -> dict[str, Any]:
    from search_assistant.skills.library import VersionedSkillLibrary

    library = VersionedSkillLibrary(data_dir / "skills")
    return {
        "active": [record.to_dict() for record in library.list_active()],
        "staging": [record.to_dict() for record in library.list_staging()],
        "archive": [record.to_dict() for record in library.list_archive()],
    }


def _skill_library_archive(data_dir: Path, name_or_slug: str, reason: str = "") -> dict[str, Any]:
    from search_assistant.skills.library import VersionedSkillLibrary

    library = VersionedSkillLibrary(data_dir / "skills")
    record = library.archive(name_or_slug, reason=reason)
    return {"action": "archived", "skill": record.to_dict()}


def _run_evolution(store: MemoryStore, data_dir: Path) -> dict[str, object]:
    markdown = ReportService(store, output_dir=data_dir / "reports").generate_markdown()
    report_path = data_dir / "reports" / "learning-report.md"
    link_result = _candidate_service(store).link_candidates_to_knowledge_graphs()
    candidates = store.list_domain_knowledge_candidates()
    candidate_status_counts = {
        status: sum(1 for item in candidates if item["status"] == status)
        for status in ("candidate", "validated", "deprecated")
    }
    skill_service = SkillDraftService(store, drafts_dir=data_dir / "skills" / "drafts")
    refreshed_skill_paths = skill_service.refresh_reviewable_drafts()
    skill_paths = skill_service.auto_create_from_experience(refresh_existing=False)
    store.add_project_ledger_entry(
        entry_type="evolution_run",
        subject="local-self-evolution",
        status="completed",
        summary="Generated learning report, refreshed reviewable skill drafts, and linked knowledge candidates.",
        evidence_refs=[
            f"report:{report_path}",
            *[f"skill:{path}" for path in refreshed_skill_paths[:5]],
            *[f"skill:{path}" for path in skill_paths[:5]],
        ],
        risk="Generated drafts and candidates remain reviewable artifacts; they are not automatically deployed as trusted policy.",
        rollback="Deprecate incorrect candidates and remove unapproved draft files from the isolated data directory.",
        metadata={
            "skill_paths": skill_paths,
            "refreshed_skill_paths": refreshed_skill_paths,
            "candidate_status_counts": candidate_status_counts,
            "created_candidate_graph_links": link_result["created_links"],
        },
    )
    return {
        "report_path": str(report_path),
        "report_written": bool(markdown),
        "skill_paths": skill_paths,
        "refreshed_skill_paths": refreshed_skill_paths,
        "immutable_trajectories": len(store.list_trajectory_logs()),
        "structured_evaluations": len(store.list_trajectory_evaluations()),
        "domain_knowledge_candidates": candidate_status_counts,
        "domain_knowledge_candidate_graph_links": {
            "created": link_result["created_links"],
            "total": len(store.list_domain_candidate_graph_links()),
        },
    }


def _run_evolution_loop(
    store: MemoryStore,
    data_dir: Path,
    interval_seconds: float,
    max_runs: int | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "interval_seconds": interval_seconds,
        "max_runs": max_runs,
    }
    session_id = store.record_runtime_session(
        "evolve-loop",
        process_id=_current_process_id(),
        status="started",
        metadata=metadata,
    )
    runs = 0
    results: list[dict[str, object]] = []
    while max_runs is None or runs < max_runs:
        results.append(_run_evolution(store, data_dir))
        runs += 1
        if max_runs is not None and runs >= max_runs:
            break
        time.sleep(interval_seconds)
    metadata["runs"] = runs
    store.update_runtime_session_status(session_id, "completed", metadata=metadata)
    return {
        "runs": runs,
        "interval_seconds": interval_seconds,
        "results": results,
        "runtime_session_id": session_id,
    }


def _run_offline_evolution(
    store: MemoryStore,
    data_dir: Path,
    judge_name: str = "rule",
    max_trajectories: int | None = None,
) -> dict[str, Any]:
    """Run the offline evolution loop: three-layer trajectory verification,
    knowledge-candidate distillation, and release/rollback monitoring."""
    from search_assistant.evolution.offline_loop import OfflineEvolutionLoop
    from search_assistant.evolution.verification import LLMRubricJudge

    judge = None
    if judge_name == "llm":
        runtime = runtime_from_settings(Settings.from_env())

        def _llm_judge_runner(instructions: str, payload: dict[str, Any]) -> str:
            return runtime.run_text_task(
                instructions,
                payload,
                temperature=0.0,
                max_tokens=1400,
            )

        judge = LLMRubricJudge(judge_runner=_llm_judge_runner)
    loop = OfflineEvolutionLoop(store, data_dir=data_dir, judge=judge)
    return loop.run(max_trajectories=max_trajectories)


def _run_judge_calibration(
    store: MemoryStore,
    data_dir: Path,
    judge_name: str = "rule",
) -> dict[str, Any]:
    """Measure the quality judge against the expert-labeled calibration set."""
    from search_assistant.evolution.verification import (
        DEFAULT_CALIBRATION_EXAMPLES,
        LLMRubricJudge,
        RuleQualityJudge,
        calibrate_quality_judge,
    )

    judge = RuleQualityJudge()
    if judge_name == "llm":
        runtime = runtime_from_settings(Settings.from_env())

        def _llm_judge_runner(instructions: str, payload: dict[str, Any]) -> str:
            return runtime.run_text_task(
                instructions,
                payload,
                temperature=0.0,
                max_tokens=1400,
            )

        judge = LLMRubricJudge(judge_runner=_llm_judge_runner)
    report = calibrate_quality_judge(judge, DEFAULT_CALIBRATION_EXAMPLES)
    result = {
        "judge": judge_name,
        "calibration_examples": len(DEFAULT_CALIBRATION_EXAMPLES),
        "dimensions": report,
        "report_path": str(data_dir / "evolution" / "judge-calibration.json"),
    }
    output_path = data_dir / "evolution"
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "judge-calibration.json"
    report_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    store.add_gate_record(
        gate_type="evolution_judge_calibration",
        subject_type="quality_judge",
        subject_id=judge_name,
        result="passed",
        reason=f"calibrated {judge_name} judge against {len(DEFAULT_CALIBRATION_EXAMPLES)} expert-labeled examples",
        evidence_refs=[str(report_file)],
        metadata={"judge": judge_name, "dimensions": report},
    )
    return result


def run_long_connection(data_dir: Path | None = None) -> None:
    from search_assistant.feishu.long_connection import run_long_connection as run

    run(data_dir=data_dir)


def run_feishu_diagnostics(settings=None) -> dict[str, Any]:
    from search_assistant.feishu.diagnostics import run_feishu_diagnostics as run

    return run(settings=settings)


def _current_process_id() -> int:
    import os

    return os.getpid()


if __name__ == "__main__":
    raise SystemExit(main())
