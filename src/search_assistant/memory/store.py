from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from search_assistant.contracts import (
    AnswerPackage,
    CollectedSource,
    DailyBriefing,
    DomainKnowledgeCandidate,
    DomainKnowledgeEntity,
    DomainKnowledgeGraph,
    DomainKnowledgeRelation,
    IncomingMessage,
    LayeredMemoryItem,
    ProviderTraceEvent,
    SourceCandidate,
    TopicFeedbackSignal,
    TopicSubscription,
    VerifiedClaim,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class MemoryStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    message_id TEXT NOT NULL,
                    event_id TEXT,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    raw_event_json TEXT NOT NULL,
                    answer_id TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS answers (
                    question_id TEXT PRIMARY KEY,
                    answer_text TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    package_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY,
                    question_id TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    source TEXT,
                    checked_at TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    supersedes_id TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experience_items (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    source_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS profile_snapshots (
                    id TEXT PRIMARY KEY,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS learning_reports (
                    id TEXT PRIMARY KEY,
                    markdown_body TEXT NOT NULL,
                    path TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS skill_drafts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    source_ids_json TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reply_attempts (
                    id TEXT PRIMARY KEY,
                    question_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runtime_sessions (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    process_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS topic_subscriptions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, chat_id, topic)
                );

                CREATE TABLE IF NOT EXISTS topic_feedback (
                    id TEXT PRIMARY KEY,
                    topic_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    body TEXT NOT NULL,
                    source_url TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS collected_sources (
                    id TEXT PRIMARY KEY,
                    topic_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    snippet TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    relevance_score REAL NOT NULL,
                    importance_score REAL NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(topic_id, url)
                );

                CREATE TABLE IF NOT EXISTS daily_briefings (
                    id TEXT PRIMARY KEY,
                    topic_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    run_date TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    package_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(topic_id, run_date)
                );

                CREATE TABLE IF NOT EXISTS trajectory_logs (
                    id TEXT PRIMARY KEY,
                    question_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trajectory_evaluations (
                    id TEXT PRIMARY KEY,
                    trajectory_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    process_json TEXT NOT NULL,
                    quality_json TEXT NOT NULL,
                    diagnosis_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS domain_knowledge_candidates (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    applies_when TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    contradictions_json TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('candidate', 'validated', 'deprecated', 'pending_user_confirm')),
                    source_ids_json TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS domain_knowledge_candidate_events (
                    id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS domain_knowledge_graphs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    overview TEXT NOT NULL,
                    source TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS domain_knowledge_entities (
                    id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (graph_id, id)
                );

                CREATE TABLE IF NOT EXISTS domain_knowledge_relations (
                    id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    source_entity_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    weight REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (graph_id, id)
                );

                CREATE TABLE IF NOT EXISTS domain_knowledge_candidate_graph_links (
                    id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    entity_id TEXT,
                    relation_id TEXT,
                    link_type TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS project_ledger_entries (
                    id TEXT PRIMARY KEY,
                    entry_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    rollback TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS project_ledger_snapshots (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL,
                    next_decision TEXT NOT NULL,
                    open_blockers_json TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    decisions_json TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agentops_gate_records (
                    id TEXT PRIMARY KEY,
                    gate_type TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    result TEXT NOT NULL CHECK(result IN ('passed', 'failed', 'waived')),
                    reason TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agentops_trace_events (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    parent_id TEXT,
                    event_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agentops_run_checkpoints (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    workflow TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    step TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS search_provider_health (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    requested_platform TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    attempted INTEGER NOT NULL,
                    ok INTEGER NOT NULL,
                    result_count INTEGER NOT NULL,
                    duration_ms REAL NOT NULL,
                    error TEXT,
                    checked_at TEXT NOT NULL
                );

                CREATE TRIGGER IF NOT EXISTS trajectory_logs_no_update
                BEFORE UPDATE ON trajectory_logs
                BEGIN
                    SELECT RAISE(ABORT, 'trajectory logs are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS trajectory_logs_no_delete
                BEFORE DELETE ON trajectory_logs
                BEGIN
                    SELECT RAISE(ABORT, 'trajectory logs are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS trajectory_evaluations_no_update
                BEFORE UPDATE ON trajectory_evaluations
                BEGIN
                    SELECT RAISE(ABORT, 'trajectory evaluations are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS trajectory_evaluations_no_delete
                BEFORE DELETE ON trajectory_evaluations
                BEGIN
                    SELECT RAISE(ABORT, 'trajectory evaluations are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS domain_knowledge_candidate_events_no_update
                BEFORE UPDATE ON domain_knowledge_candidate_events
                BEGIN
                    SELECT RAISE(ABORT, 'domain knowledge candidate events are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS domain_knowledge_candidate_events_no_delete
                BEFORE DELETE ON domain_knowledge_candidate_events
                BEGIN
                    SELECT RAISE(ABORT, 'domain knowledge candidate events are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS domain_knowledge_candidate_graph_links_no_update
                BEFORE UPDATE ON domain_knowledge_candidate_graph_links
                BEGIN
                    SELECT RAISE(ABORT, 'domain knowledge candidate graph links are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS domain_knowledge_candidate_graph_links_no_delete
                BEFORE DELETE ON domain_knowledge_candidate_graph_links
                BEGIN
                    SELECT RAISE(ABORT, 'domain knowledge candidate graph links are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS project_ledger_entries_no_update
                BEFORE UPDATE ON project_ledger_entries
                BEGIN
                    SELECT RAISE(ABORT, 'project ledger entries are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS project_ledger_entries_no_delete
                BEFORE DELETE ON project_ledger_entries
                BEGIN
                    SELECT RAISE(ABORT, 'project ledger entries are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS project_ledger_snapshots_no_update
                BEFORE UPDATE ON project_ledger_snapshots
                BEGIN
                    SELECT RAISE(ABORT, 'project ledger snapshots are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS project_ledger_snapshots_no_delete
                BEFORE DELETE ON project_ledger_snapshots
                BEGIN
                    SELECT RAISE(ABORT, 'project ledger snapshots are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS agentops_gate_records_no_update
                BEFORE UPDATE ON agentops_gate_records
                BEGIN
                    SELECT RAISE(ABORT, 'agentops gate records are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS agentops_gate_records_no_delete
                BEFORE DELETE ON agentops_gate_records
                BEGIN
                    SELECT RAISE(ABORT, 'agentops gate records are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS agentops_trace_events_no_update
                BEFORE UPDATE ON agentops_trace_events
                BEGIN
                    SELECT RAISE(ABORT, 'agentops trace events are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS agentops_trace_events_no_delete
                BEFORE DELETE ON agentops_trace_events
                BEGIN
                    SELECT RAISE(ABORT, 'agentops trace events are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS agentops_run_checkpoints_no_update
                BEFORE UPDATE ON agentops_run_checkpoints
                BEGIN
                    SELECT RAISE(ABORT, 'agentops run checkpoints are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS agentops_run_checkpoints_no_delete
                BEFORE DELETE ON agentops_run_checkpoints
                BEGIN
                    SELECT RAISE(ABORT, 'agentops run checkpoints are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS search_provider_health_no_update
                BEFORE UPDATE ON search_provider_health
                BEGIN
                    SELECT RAISE(ABORT, 'search provider health records are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS search_provider_health_no_delete
                BEFORE DELETE ON search_provider_health
                BEGIN
                    SELECT RAISE(ABORT, 'search provider health records are immutable');
                END;
                CREATE TABLE IF NOT EXISTS source_candidates (
                    id TEXT PRIMARY KEY,
                    topic_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    snippet TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    relevance_score REAL NOT NULL,
                    importance_score REAL NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS provider_trace_events (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    topic_id TEXT,
                    provider TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_count INTEGER NOT NULL,
                    reason TEXT,
                    error TEXT,
                    elapsed_ms REAL,
                    tier TEXT,
                    budget_share REAL,
                    checked_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS layered_memory_items (
                    id TEXT PRIMARY KEY,
                    layer TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    status TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS entity_embeddings (
                    content_hash TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS topic_feedback_signals (
                    id TEXT PRIMARY KEY,
                    topic_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    body TEXT NOT NULL,
                    source_url TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(connection, "memory_items", "user_id", "TEXT NOT NULL DEFAULT 'legacy'")
            self._ensure_column(connection, "memory_items", "chat_id", "TEXT NOT NULL DEFAULT 'legacy'")
            self._ensure_column(connection, "experience_items", "user_id", "TEXT NOT NULL DEFAULT 'legacy'")
            self._ensure_column(connection, "experience_items", "chat_id", "TEXT NOT NULL DEFAULT 'legacy'")
            self._ensure_column(connection, "skill_drafts", "user_id", "TEXT NOT NULL DEFAULT 'legacy'")
            self._ensure_column(connection, "skill_drafts", "chat_id", "TEXT NOT NULL DEFAULT 'legacy'")
            self._ensure_column(connection, "topic_subscriptions", "source_recipe_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column(connection, "provider_trace_events", "tier", "TEXT")
            self._ensure_column(connection, "provider_trace_events", "budget_share", "REAL")
            self._migrate_candidate_status_constraint(connection)

    @staticmethod
    def _migrate_candidate_status_constraint(connection: sqlite3.Connection) -> None:
        """Rebuild domain_knowledge_candidates when its status CHECK constraint
        predates ``pending_user_confirm`` (older schemas only allowed
        candidate/validated/deprecated).  Data is preserved."""
        sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='domain_knowledge_candidates'"
        ).fetchone()
        if sql is None or "pending_user_confirm" in str(sql[0]):
            return
        connection.execute("ALTER TABLE domain_knowledge_candidates RENAME TO domain_knowledge_candidates_old")
        connection.execute(
            """
            CREATE TABLE domain_knowledge_candidates (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                claim TEXT NOT NULL,
                applies_when TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                contradictions_json TEXT NOT NULL,
                confidence TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('candidate', 'validated', 'deprecated', 'pending_user_confirm')),
                source_ids_json TEXT NOT NULL,
                fingerprint TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO domain_knowledge_candidates
                (id, topic, claim, applies_when, evidence_json, contradictions_json,
                 confidence, status, source_ids_json, fingerprint, created_at, updated_at)
            SELECT id, topic, claim, applies_when, evidence_json, contradictions_json,
                   confidence, status, source_ids_json, fingerprint, created_at, updated_at
            FROM domain_knowledge_candidates_old
            """
        )
        connection.execute("DROP TABLE domain_knowledge_candidates_old")

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def record_interaction(self, message: IncomingMessage) -> str:
        question_id = _new_id("q")
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO interactions (
                        id, dedupe_key, message_id, event_id, user_id, chat_id, text,
                        source, raw_event_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        question_id,
                        message.dedupe_key,
                        message.message_id,
                        message.event_id,
                        message.user_id,
                        message.chat_id,
                        message.text,
                        message.source,
                        json.dumps(message.raw_event, ensure_ascii=False),
                        _now_iso(),
                    ),
                )
            except sqlite3.IntegrityError:
                row = connection.execute(
                    """
                    SELECT id FROM interactions
                    WHERE dedupe_key = ? OR message_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (message.dedupe_key, message.message_id),
                ).fetchone()
                if row is None:
                    raise
                return str(row["id"])
        return question_id

    def record_answer(self, package: AnswerPackage) -> None:
        payload = package.model_dump(mode="json")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO answers (
                    question_id, answer_text, classification, confidence, package_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    package.question_id,
                    package.answer_text,
                    package.classification,
                    package.confidence,
                    json.dumps(payload, ensure_ascii=False),
                    _now_iso(),
                ),
            )
            connection.execute(
                "UPDATE interactions SET answer_id = ? WHERE id = ?",
                (package.question_id, package.question_id),
            )
            self._insert_missing_verified_claims(connection, package.question_id, package.verified_claims)
            self._insert_trajectory_log(connection, package)

    def has_interaction(self, dedupe_key: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM interactions WHERE dedupe_key = ? OR message_id = ? LIMIT 1",
                (dedupe_key, dedupe_key),
            ).fetchone()
        return row is not None

    def list_interactions(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM interactions ORDER BY created_at DESC, id DESC"
        parameters: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            parameters = (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def latest_answer_for_dedupe_key(self, dedupe_key: str) -> AnswerPackage | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT a.package_json
                FROM interactions i
                JOIN answers a ON a.question_id = i.answer_id
                WHERE i.dedupe_key = ?
                ORDER BY a.created_at DESC
                LIMIT 1
                """,
                (dedupe_key,),
            ).fetchone()
        if row is None:
            return None
        return AnswerPackage.model_validate_json(row["package_json"])

    def latest_answer_for_message_id(self, message_id: str) -> AnswerPackage | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT a.package_json
                FROM interactions i
                JOIN answers a ON a.question_id = i.answer_id
                WHERE i.message_id = ?
                ORDER BY a.created_at DESC
                LIMIT 1
                """,
                (message_id,),
            ).fetchone()
        if row is None:
            return None
        return AnswerPackage.model_validate_json(row["package_json"])

    def add_memory_item(
        self,
        kind: str,
        content: str,
        source_id: str,
        supersedes_id: str | None = None,
        user_id: str = "system",
        chat_id: str = "system",
    ) -> str:
        item_id = _new_id("mem")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_items (
                    id, kind, content, source_id, supersedes_id, user_id, chat_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (item_id, kind, content, source_id, supersedes_id, user_id, chat_id, _now_iso()),
            )
        return item_id

    def list_memory_items(self, user_id: str | None = None, chat_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM memory_items"
        parameters: tuple[str, ...] = ()
        if user_id is not None and chat_id is not None:
            query += " WHERE (user_id = ? AND chat_id = ?) OR (user_id = 'system' AND chat_id = 'system')"
            parameters = (user_id, chat_id)
        query += " ORDER BY created_at ASC, id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def list_answers(self) -> list[AnswerPackage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT package_json FROM answers ORDER BY created_at ASC"
            ).fetchall()
        return [AnswerPackage.model_validate_json(row["package_json"]) for row in rows]

    def list_evidence(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evidence ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_trajectory_logs(
        self,
        user_id: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM trajectory_logs"
        parameters: tuple[str, ...] = ()
        if user_id is not None and chat_id is not None:
            query += " WHERE user_id = ? AND chat_id = ?"
            parameters = (user_id, chat_id)
        query += " ORDER BY created_at ASC, id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._trajectory_row(row) for row in rows]

    def latest_trajectory_for_question(self, question_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM trajectory_logs WHERE question_id = ? ORDER BY created_at DESC LIMIT 1",
                (question_id,),
            ).fetchone()
        return self._trajectory_row(row) if row is not None else None

    def add_trajectory_evaluation(
        self,
        trajectory_id: str,
        question_id: str,
        result_verification: dict[str, Any],
        process_verification: dict[str, Any],
        quality_verification: dict[str, Any],
        diagnosis: dict[str, Any],
    ) -> str:
        evaluation_id = _new_id("traj_eval")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO trajectory_evaluations (
                    id, trajectory_id, question_id, result_json, process_json,
                    quality_json, diagnosis_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    trajectory_id,
                    question_id,
                    json.dumps(result_verification, ensure_ascii=False),
                    json.dumps(process_verification, ensure_ascii=False),
                    json.dumps(quality_verification, ensure_ascii=False),
                    json.dumps(diagnosis, ensure_ascii=False),
                    _now_iso(),
                ),
            )
        return evaluation_id

    def list_trajectory_evaluations(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM trajectory_evaluations ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [
            {
                **dict(row),
                "result_verification": json.loads(row["result_json"]),
                "process_verification": json.loads(row["process_json"]),
                "quality_verification": json.loads(row["quality_json"]),
                "diagnosis": json.loads(row["diagnosis_json"]),
            }
            for row in rows
        ]

    def add_domain_knowledge_candidate(self, candidate: DomainKnowledgeCandidate) -> str:
        return self.upsert_domain_knowledge_candidate(candidate)[0]

    def upsert_domain_knowledge_candidate(self, candidate: DomainKnowledgeCandidate) -> tuple[str, bool]:
        """Insert a candidate or merge new evidence into the existing row.

        Deduplication key is ``fingerprint`` (topic + claim).  When the
        fingerprint already exists, evidence is unioned by URL, ``source_ids``
        and ``contradictions`` are unioned, confidence never decreases, and a
        candidate event records the merge.  This is what lets a claim that
        reappears in an independent briefing accumulate cross-trajectory
        support instead of being silently dropped by ``INSERT OR IGNORE``.

        Returns ``(candidate_id, created)``.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO domain_knowledge_candidates (
                    id, topic, claim, applies_when, evidence_json, contradictions_json,
                    confidence, status, source_ids_json, fingerprint, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.id,
                    candidate.topic,
                    candidate.claim,
                    candidate.applies_when,
                    json.dumps([item.model_dump(mode="json") for item in candidate.evidence], ensure_ascii=False),
                    json.dumps(candidate.contradictions, ensure_ascii=False),
                    candidate.confidence,
                    candidate.status,
                    json.dumps(candidate.source_ids, ensure_ascii=False),
                    candidate.fingerprint,
                    candidate.created_at,
                    candidate.updated_at,
                ),
            )
            if cursor.rowcount:
                self._insert_candidate_event(
                    connection,
                    candidate.id,
                    from_status=None,
                    to_status="candidate",
                    reason="created from traceable search evidence",
                )
                return candidate.id, True
            row = connection.execute(
                "SELECT * FROM domain_knowledge_candidates WHERE fingerprint = ?",
                (candidate.fingerprint,),
            ).fetchone()
        if row is None:
            raise RuntimeError("candidate insert was ignored without a matching fingerprint")
        existing = self._domain_candidate_row(row)
        return self._merge_domain_knowledge_candidate(existing, candidate), False

    def _merge_domain_knowledge_candidate(
        self,
        existing: dict[str, Any],
        candidate: DomainKnowledgeCandidate,
    ) -> str:
        evidence_by_url = {
            str(item.get("url", "")): item
            for item in existing["evidence"]
            if str(item.get("url", ""))
        }
        for item in candidate.evidence:
            url = str(item.url or "")
            if url and url not in evidence_by_url:
                evidence_by_url[url] = item.model_dump(mode="json")
        merged_evidence = list(evidence_by_url.values())
        merged_source_ids = list(dict.fromkeys([*existing["source_ids"], *candidate.source_ids]))
        merged_contradictions = list(dict.fromkeys([*existing["contradictions"], *candidate.contradictions]))
        confidence_rank = {"high": 3, "medium": 2, "low": 1}
        merged_confidence = max(
            [str(existing["confidence"]), str(candidate.confidence)],
            key=lambda value: confidence_rank.get(value, 0),
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE domain_knowledge_candidates
                SET evidence_json = ?, source_ids_json = ?, contradictions_json = ?,
                    confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(merged_evidence, ensure_ascii=False),
                    json.dumps(merged_source_ids, ensure_ascii=False),
                    json.dumps(merged_contradictions, ensure_ascii=False),
                    merged_confidence,
                    _now_iso(),
                    existing["id"],
                ),
            )
            self._insert_candidate_event(
                connection,
                str(existing["id"]),
                from_status="candidate",
                to_status="candidate",
                reason="merged evidence from an independent trajectory",
            )
        return str(existing["id"])

    def list_domain_knowledge_candidates(self, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM domain_knowledge_candidates"
        parameters: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY created_at ASC, id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._domain_candidate_row(row) for row in rows]

    def get_domain_knowledge_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM domain_knowledge_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
        return self._domain_candidate_row(row) if row is not None else None

    def update_domain_knowledge_candidate_status(
        self,
        candidate_id: str,
        status: str,
        reason: str,
    ) -> None:
        allowed_transitions = {
            "candidate": {"validated", "deprecated", "pending_user_confirm"},
            "pending_user_confirm": {"validated", "deprecated"},
            "validated": {"deprecated"},
            "deprecated": set(),
        }
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM domain_knowledge_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"domain knowledge candidate not found: {candidate_id}")
            current_status = str(row["status"])
            if status == current_status:
                return
            if status not in allowed_transitions.get(current_status, set()):
                raise ValueError(f"invalid candidate status transition: {current_status} -> {status}")
            now = _now_iso()
            connection.execute(
                "UPDATE domain_knowledge_candidates SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, candidate_id),
            )
            self._insert_candidate_event(connection, candidate_id, current_status, status, reason)

    def list_domain_knowledge_candidate_events(self, candidate_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM domain_knowledge_candidate_events"
        parameters: tuple[str, ...] = ()
        if candidate_id is not None:
            query += " WHERE candidate_id = ?"
            parameters = (candidate_id,)
        query += " ORDER BY created_at ASC, id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def link_domain_candidate_to_graph(
        self,
        candidate_id: str,
        graph_id: str,
        entity_id: str | None = None,
        relation_id: str | None = None,
        link_type: str = "supports",
        note: str = "",
    ) -> str:
        link_id = _new_id("knowledge_link")
        with self._connect() as connection:
            candidate = connection.execute(
                "SELECT 1 FROM domain_knowledge_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if candidate is None:
                raise KeyError(f"domain knowledge candidate not found: {candidate_id}")
            graph = connection.execute(
                "SELECT 1 FROM domain_knowledge_graphs WHERE id = ?",
                (graph_id,),
            ).fetchone()
            if graph is None:
                raise KeyError(f"domain knowledge graph not found: {graph_id}")
            if entity_id is not None:
                entity = connection.execute(
                    "SELECT 1 FROM domain_knowledge_entities WHERE graph_id = ? AND id = ?",
                    (graph_id, entity_id),
                ).fetchone()
                if entity is None:
                    raise KeyError(f"domain knowledge entity not found: {graph_id}/{entity_id}")
            if relation_id is not None:
                relation = connection.execute(
                    "SELECT 1 FROM domain_knowledge_relations WHERE graph_id = ? AND id = ?",
                    (graph_id, relation_id),
                ).fetchone()
                if relation is None:
                    raise KeyError(f"domain knowledge relation not found: {graph_id}/{relation_id}")
            connection.execute(
                """
                INSERT INTO domain_knowledge_candidate_graph_links (
                    id, candidate_id, graph_id, entity_id, relation_id, link_type, note, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (link_id, candidate_id, graph_id, entity_id, relation_id, link_type, note, _now_iso()),
            )
        return link_id

    def list_domain_candidate_graph_links(self, candidate_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM domain_knowledge_candidate_graph_links"
        parameters: tuple[str, ...] = ()
        if candidate_id is not None:
            query += " WHERE candidate_id = ?"
            parameters = (candidate_id,)
        query += " ORDER BY created_at ASC, id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def add_project_ledger_entry(
        self,
        entry_type: str,
        subject: str,
        status: str,
        summary: str,
        evidence_refs: list[str] | None = None,
        risk: str = "",
        rollback: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        ledger_id = _new_id("ledger")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO project_ledger_entries (
                    id, entry_type, subject, status, summary, evidence_refs_json,
                    risk, rollback, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ledger_id,
                    entry_type.strip() or "operation",
                    subject.strip() or "unspecified",
                    status.strip() or "recorded",
                    summary.strip(),
                    json.dumps(evidence_refs or [], ensure_ascii=False),
                    risk.strip(),
                    rollback.strip(),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    _now_iso(),
                ),
            )
        return ledger_id

    def list_project_ledger_entries(
        self,
        entry_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM project_ledger_entries"
        parameters: list[Any] = []
        if entry_type is not None:
            query += " WHERE entry_type = ?"
            parameters.append(entry_type)
        query += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [self._json_row(row, ("evidence_refs", "metadata")) for row in rows]

    def add_project_ledger_snapshot(
        self,
        project_id: str,
        objective: str,
        phase: str,
        status: str,
        next_decision: str,
        open_blockers: list[str] | None = None,
        constraints: dict[str, Any] | None = None,
        decisions: list[dict[str, Any]] | None = None,
        evidence_refs: list[str] | None = None,
        version: str = "1",
    ) -> str:
        snapshot_id = _new_id("project_state")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO project_ledger_snapshots (
                    id, project_id, objective, phase, status, next_decision,
                    open_blockers_json, constraints_json, decisions_json,
                    evidence_refs_json, version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    project_id.strip() or "default",
                    objective.strip(),
                    phase.strip() or "unspecified",
                    status.strip() or "active",
                    next_decision.strip(),
                    json.dumps(open_blockers or [], ensure_ascii=False),
                    json.dumps(constraints or {}, ensure_ascii=False),
                    json.dumps(decisions or [], ensure_ascii=False),
                    json.dumps(evidence_refs or [], ensure_ascii=False),
                    version.strip() or "1",
                    _now_iso(),
                ),
            )
        return snapshot_id

    def ensure_project_ledger_snapshot(
        self,
        project_id: str,
        objective: str,
        phase: str,
        status: str,
        next_decision: str,
        open_blockers: list[str] | None = None,
        constraints: dict[str, Any] | None = None,
        decisions: list[dict[str, Any]] | None = None,
        evidence_refs: list[str] | None = None,
        version: str = "1",
    ) -> str:
        existing = self.latest_project_ledger_snapshot(project_id)
        if existing is not None:
            return str(existing["id"])
        return self.add_project_ledger_snapshot(
            project_id=project_id,
            objective=objective,
            phase=phase,
            status=status,
            next_decision=next_decision,
            open_blockers=open_blockers,
            constraints=constraints,
            decisions=decisions,
            evidence_refs=evidence_refs,
            version=version,
        )

    def list_project_ledger_snapshots(
        self,
        project_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM project_ledger_snapshots"
        parameters: list[Any] = []
        if project_id is not None:
            query += " WHERE project_id = ?"
            parameters.append(project_id)
        query += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [
            self._json_row(row, ("open_blockers", "constraints", "decisions", "evidence_refs"))
            for row in rows
        ]

    def latest_project_ledger_snapshot(self, project_id: str = "default") -> dict[str, Any] | None:
        rows = self.list_project_ledger_snapshots(project_id=project_id, limit=1)
        return rows[0] if rows else None

    def add_gate_record(
        self,
        gate_type: str,
        subject_type: str,
        subject_id: str,
        result: str,
        reason: str,
        evidence_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if result not in {"passed", "failed", "waived"}:
            raise ValueError("gate result must be passed, failed, or waived")
        gate_id = _new_id("gate")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agentops_gate_records (
                    id, gate_type, subject_type, subject_id, result, reason,
                    evidence_refs_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gate_id,
                    gate_type.strip() or "generic_gate",
                    subject_type.strip() or "unknown",
                    subject_id.strip(),
                    result,
                    reason.strip(),
                    json.dumps(evidence_refs or [], ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    _now_iso(),
                ),
            )
        return gate_id

    def list_gate_records(
        self,
        gate_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM agentops_gate_records"
        parameters: list[Any] = []
        if gate_type is not None:
            query += " WHERE gate_type = ?"
            parameters.append(gate_type)
        query += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [self._json_row(row, ("evidence_refs", "metadata")) for row in rows]

    def add_trace_event(
        self,
        run_id: str,
        event_type: str,
        name: str,
        status: str,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
        parent_id: str | None = None,
    ) -> str:
        trace_id = _new_id("trace")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agentops_trace_events (
                    id, run_id, parent_id, event_type, name, status,
                    duration_ms, metadata_json, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    run_id,
                    parent_id,
                    event_type.strip() or "event",
                    name.strip() or "unnamed",
                    status.strip() or "recorded",
                    float(duration_ms),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    error,
                    _now_iso(),
                ),
            )
        return trace_id

    def list_trace_events(
        self,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM agentops_trace_events"
        parameters: list[Any] = []
        if run_id is not None:
            query += " WHERE run_id = ?"
            parameters.append(run_id)
        query += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [self._json_row(row, ("metadata",)) for row in rows]

    def add_run_checkpoint(
        self,
        run_id: str,
        workflow: str,
        subject: str,
        step: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        checkpoint_id = _new_id("checkpoint")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agentops_run_checkpoints (
                    id, run_id, workflow, subject, step, status, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    run_id.strip(),
                    workflow.strip() or "workflow",
                    subject.strip() or "unspecified",
                    step.strip() or "step",
                    status.strip() or "recorded",
                    json.dumps(payload or {}, ensure_ascii=False),
                    _now_iso(),
                ),
            )
        return checkpoint_id

    def list_run_checkpoints(
        self,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM agentops_run_checkpoints"
        parameters: list[Any] = []
        if run_id is not None:
            query += " WHERE run_id = ?"
            parameters.append(run_id)
        query += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [self._json_row(row, ("payload",)) for row in rows]

    def record_search_provider_health(
        self,
        run_id: str,
        requested_platform: str,
        provider: str,
        query: str,
        ok: bool,
        result_count: int,
        duration_ms: float = 0.0,
        error: str | None = None,
        attempted: bool = True,
    ) -> str:
        health_id = _new_id("provider_health")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO search_provider_health (
                    id, run_id, requested_platform, provider, query_text,
                    attempted, ok, result_count, duration_ms, error, checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    health_id,
                    run_id,
                    requested_platform,
                    provider,
                    query,
                    int(attempted),
                    int(ok),
                    int(result_count),
                    float(duration_ms),
                    error,
                    _now_iso(),
                ),
            )
        return health_id

    def list_search_provider_health(
        self,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM search_provider_health"
        parameters: list[Any] = []
        if run_id is not None:
            query += " WHERE run_id = ?"
            parameters.append(run_id)
        query += " ORDER BY checked_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [
            {
                **dict(row),
                "attempted": bool(row["attempted"]),
                "ok": bool(row["ok"]),
            }
            for row in rows
        ]

    def search_provider_health_summary(self, limit: int | None = None) -> dict[str, Any]:
        rows = self.list_search_provider_health(limit=limit)
        providers: dict[str, dict[str, Any]] = {}
        platforms: dict[str, dict[str, Any]] = {}
        for row in rows:
            provider = str(row["provider"])
            platform = str(row["requested_platform"])
            provider_item = providers.setdefault(
                provider,
                {"provider": provider, "attempts": 0, "successes": 0, "results": 0, "errors": 0, "empty_results": 0},
            )
            platform_item = platforms.setdefault(
                platform,
                {
                    "requested_platform": platform,
                    "attempts": 0,
                    "successes": 0,
                    "results": 0,
                    "errors": 0,
                    "empty_results": 0,
                },
            )
            for item in (provider_item, platform_item):
                item["attempts"] += 1
                item["successes"] += 1 if row["ok"] else 0
                item["results"] += int(row["result_count"])
                item["errors"] += 1 if row.get("error") else 0
                if row["attempted"] and row["ok"] and int(row["result_count"]) == 0:
                    item["empty_results"] += 1
        return {
            "records": len(rows),
            "providers": sorted(providers.values(), key=lambda item: str(item["provider"])),
            "platforms": sorted(platforms.values(), key=lambda item: str(item["requested_platform"])),
        }

    def update_answer_verification(
        self,
        question_id: str,
        verified_claims: list[VerifiedClaim],
        unverified_claims: list[str],
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT package_json FROM answers WHERE question_id = ?",
                (question_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"answer not found: {question_id}")
            package = AnswerPackage.model_validate_json(row["package_json"])
            package.verified_claims = verified_claims
            package.unverified_claims = unverified_claims
            payload = package.model_dump(mode="json")
            connection.execute(
                """
                UPDATE answers
                SET package_json = ?
                WHERE question_id = ?
                """,
                (json.dumps(payload, ensure_ascii=False), question_id),
            )
            self._delete_stale_verified_claims(connection, question_id, verified_claims)
            return self._insert_missing_verified_claims(connection, question_id, verified_claims)

    def add_experience_item(
        self,
        title: str,
        body: str,
        source_ids: list[str],
        user_id: str = "system",
        chat_id: str = "system",
    ) -> str:
        experience_id = _new_id("exp")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experience_items (
                    id, title, body, source_ids_json, user_id, chat_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experience_id,
                    title,
                    body,
                    json.dumps(source_ids, ensure_ascii=False),
                    user_id,
                    chat_id,
                    _now_iso(),
                ),
            )
        return experience_id

    def list_experience_items(self, user_id: str | None = None, chat_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM experience_items"
        parameters: tuple[str, ...] = ()
        if user_id is not None and chat_id is not None:
            query += " WHERE (user_id = ? AND chat_id = ?) OR (user_id = 'system' AND chat_id = 'system')"
            parameters = (user_id, chat_id)
        query += " ORDER BY created_at ASC, id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def record_reply_attempt(
        self,
        question_id: str,
        message_id: str,
        channel: str,
        status: str,
        response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> str:
        attempt_id = _new_id("reply")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO reply_attempts (
                    id, question_id, message_id, channel, status, response_json, error, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    question_id,
                    message_id,
                    channel,
                    status,
                    json.dumps(response or {}, ensure_ascii=False),
                    error,
                    _now_iso(),
                ),
            )
        return attempt_id

    def list_reply_attempts(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM reply_attempts ORDER BY created_at DESC, id DESC"
        parameters: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            parameters = (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def has_successful_reply(self, message_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM reply_attempts WHERE message_id = ? AND status = 'success' LIMIT 1",
                (message_id,),
            ).fetchone()
        return row is not None

    def record_runtime_session(
        self,
        kind: str,
        process_id: int,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        session_id = _new_id("runtime")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runtime_sessions (
                    id, kind, process_id, status, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    kind,
                    process_id,
                    status,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    _now_iso(),
                ),
            )
        return session_id

    def update_runtime_session_status(
        self,
        session_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            if metadata is None:
                connection.execute(
                    "UPDATE runtime_sessions SET status = ? WHERE id = ?",
                    (status, session_id),
                )
            else:
                connection.execute(
                    "UPDATE runtime_sessions SET status = ?, metadata_json = ? WHERE id = ?",
                    (status, json.dumps(metadata, ensure_ascii=False), session_id),
                )

    def list_runtime_sessions(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM runtime_sessions ORDER BY created_at DESC, id DESC"
        parameters: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            parameters = (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def diagnostic_counts(self) -> dict[str, int]:
        tables = (
            "interactions",
            "answers",
            "evidence",
            "memory_items",
            "experience_items",
            "profile_snapshots",
            "learning_reports",
            "skill_drafts",
            "reply_attempts",
            "runtime_sessions",
            "topic_subscriptions",
            "topic_feedback",
            "collected_sources",
            "daily_briefings",
            "domain_knowledge_graphs",
            "domain_knowledge_entities",
            "domain_knowledge_relations",
            "trajectory_logs",
            "trajectory_evaluations",
            "domain_knowledge_candidates",
            "domain_knowledge_candidate_events",
            "domain_knowledge_candidate_graph_links",
            "project_ledger_entries",
            "project_ledger_snapshots",
            "agentops_gate_records",
            "agentops_trace_events",
            "agentops_run_checkpoints",
            "search_provider_health",
            "source_candidates",
            "provider_trace_events",
            "layered_memory_items",
            "topic_feedback_signals",
        )
        with self._connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in tables
            }

    def add_profile_snapshot(self, summary: dict[str, Any]) -> str:
        snapshot_id = _new_id("profile")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO profile_snapshots (id, summary_json, created_at) VALUES (?, ?, ?)",
                (snapshot_id, json.dumps(summary, ensure_ascii=False), _now_iso()),
            )
        return snapshot_id

    def list_profile_snapshots(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM profile_snapshots ORDER BY created_at ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def add_learning_report(self, markdown_body: str, path: str | None = None) -> str:
        report_id = _new_id("report")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO learning_reports (id, markdown_body, path, created_at) VALUES (?, ?, ?, ?)",
                (report_id, markdown_body, path, _now_iso()),
            )
        return report_id

    def list_learning_reports(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM learning_reports ORDER BY created_at ASC").fetchall()
        return [dict(row) for row in rows]

    def add_skill_draft(
        self,
        name: str,
        path: str,
        source_ids: list[str],
        user_id: str = "system",
        chat_id: str = "system",
    ) -> str:
        draft_id = _new_id("skill")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO skill_drafts (
                    id, name, path, source_ids_json, review_status, user_id, chat_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (draft_id, name, path, json.dumps(source_ids), "draft", user_id, chat_id, _now_iso()),
            )
        return draft_id

    def update_skill_draft_status(self, draft_id: str, review_status: str, path: str | None = None) -> None:
        with self._connect() as connection:
            if path is None:
                connection.execute(
                    "UPDATE skill_drafts SET review_status = ? WHERE id = ?",
                    (review_status, draft_id),
                )
            else:
                connection.execute(
                    "UPDATE skill_drafts SET review_status = ?, path = ? WHERE id = ?",
                    (review_status, path, draft_id),
                )

    def list_skill_drafts(self, user_id: str | None = None, chat_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM skill_drafts"
        parameters: tuple[str, ...] = ()
        if user_id is not None and chat_id is not None:
            query += " WHERE (user_id = ? AND chat_id = ?) OR (user_id = 'system' AND chat_id = 'system')"
            parameters = (user_id, chat_id)
        query += " ORDER BY created_at ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def upsert_topic(self, user_id: str, chat_id: str, topic: str, enabled: bool = True) -> TopicSubscription:
        normalized = " ".join(topic.split())
        if not normalized:
            raise ValueError("topic must not be empty")
        now = _now_iso()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM topic_subscriptions WHERE user_id = ? AND chat_id = ? AND topic = ?",
                (user_id, chat_id, normalized),
            ).fetchone()
            if row is None:
                topic_id = _new_id("topic")
                connection.execute(
                    """
                    INSERT INTO topic_subscriptions (id, user_id, chat_id, topic, enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (topic_id, user_id, chat_id, normalized, int(enabled), now, now),
                )
                row = connection.execute("SELECT * FROM topic_subscriptions WHERE id = ?", (topic_id,)).fetchone()
            else:
                connection.execute(
                    "UPDATE topic_subscriptions SET enabled = ?, updated_at = ? WHERE id = ?",
                    (int(enabled), now, row["id"]),
                )
                row = connection.execute("SELECT * FROM topic_subscriptions WHERE id = ?", (row["id"],)).fetchone()
        return self._topic_from_row(row)

    def list_topics(self, user_id: str, chat_id: str, enabled_only: bool = False) -> list[TopicSubscription]:
        query = "SELECT * FROM topic_subscriptions WHERE user_id = ? AND chat_id = ?"
        parameters: tuple[object, ...] = (user_id, chat_id)
        if enabled_only:
            query += " AND enabled = 1"
        query += " ORDER BY updated_at DESC, id DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._topic_from_row(row) for row in rows]

    def add_topic_feedback(
        self,
        topic_id: str,
        user_id: str,
        chat_id: str,
        body: str,
        source_url: str | None = None,
    ) -> str:
        feedback_id = _new_id("feedback")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO topic_feedback (id, topic_id, user_id, chat_id, body, source_url, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (feedback_id, topic_id, user_id, chat_id, body, source_url, _now_iso()),
            )
        return feedback_id

    def list_topic_feedback(self, topic_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM topic_feedback WHERE topic_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (topic_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_topic_source_recipe(self, topic_id: str, source_recipe: dict[str, float]) -> None:
        payload = {
            str(key): float(value)
            for key, value in source_recipe.items()
            if _is_number(value) and float(value) > 0
        }
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE topic_subscriptions
                SET source_recipe_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(payload, ensure_ascii=False, sort_keys=True), _now_iso(), topic_id),
            )

    def add_topic_feedback_signal(
        self,
        topic_id: str,
        user_id: str,
        chat_id: str,
        signal_type: str,
        scope: str,
        body: str,
        source_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        signal_id = _new_id("signal")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO topic_feedback_signals (
                    id, topic_id, user_id, chat_id, signal_type, scope, body, source_url, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    topic_id,
                    user_id,
                    chat_id,
                    signal_type,
                    scope,
                    body,
                    source_url,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    _now_iso(),
                ),
            )
        return signal_id

    def list_topic_feedback_signals(self, topic_id: str | None = None, limit: int = 100) -> list[TopicFeedbackSignal]:
        query = "SELECT * FROM topic_feedback_signals"
        parameters: tuple[object, ...] = ()
        if topic_id is not None:
            query += " WHERE topic_id = ?"
            parameters = (topic_id,)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        parameters = (*parameters, limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            TopicFeedbackSignal(
                id=str(row["id"]),
                topic_id=str(row["topic_id"]),
                user_id=str(row["user_id"]),
                chat_id=str(row["chat_id"]),
                signal_type=str(row["signal_type"]),  # type: ignore[arg-type]
                scope=str(row["scope"]),  # type: ignore[arg-type]
                body=str(row["body"]),
                source_url=row["source_url"],
                metadata=_json_object(row["metadata_json"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def upsert_collected_source(self, source: CollectedSource) -> CollectedSource:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM collected_sources WHERE topic_id = ? AND url = ?",
                (source.topic_id, source.url),
            ).fetchone()
            source_id = str(existing["id"]) if existing else source.id
            connection.execute(
                """
                INSERT INTO collected_sources (
                    id, topic_id, user_id, title, url, snippet, platform, provider, query_text,
                    relevance_score, importance_score, retrieved_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic_id, url) DO UPDATE SET
                    title = excluded.title,
                    snippet = excluded.snippet,
                    platform = excluded.platform,
                    provider = excluded.provider,
                    query_text = excluded.query_text,
                    relevance_score = excluded.relevance_score,
                    importance_score = excluded.importance_score,
                    retrieved_at = excluded.retrieved_at
                """,
                (
                    source_id,
                    source.topic_id,
                    source.user_id,
                    source.title,
                    source.url,
                    source.snippet,
                    source.platform,
                    source.provider,
                    source.query,
                    source.relevance_score,
                    source.importance_score,
                    source.retrieved_at,
                    _now_iso(),
                ),
            )
        return source.model_copy(update={"id": source_id})

    def record_source_candidate(self, candidate: SourceCandidate) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO source_candidates (
                    id, topic_id, user_id, title, url, snippet, platform, provider, query_text,
                    status, reason, relevance_score, importance_score, retrieved_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.id,
                    candidate.topic_id,
                    candidate.user_id,
                    candidate.title,
                    candidate.url,
                    candidate.snippet,
                    candidate.platform,
                    candidate.provider,
                    candidate.query,
                    candidate.status,
                    candidate.reason,
                    candidate.relevance_score,
                    candidate.importance_score,
                    candidate.retrieved_at,
                    candidate.created_at,
                ),
            )

    def list_source_candidates(
        self,
        topic_id: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[SourceCandidate]:
        query = "SELECT * FROM source_candidates"
        conditions: list[str] = []
        parameters: list[object] = []
        if topic_id is not None:
            conditions.append("topic_id = ?")
            parameters.append(topic_id)
        if status is not None:
            conditions.append("status = ?")
            parameters.append(status)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [
            SourceCandidate(
                id=str(row["id"]),
                topic_id=str(row["topic_id"]),
                user_id=str(row["user_id"]),
                title=str(row["title"]),
                url=str(row["url"]),
                snippet=str(row["snippet"]),
                platform=str(row["platform"]),
                provider=str(row["provider"]),
                query=str(row["query_text"]),
                status=str(row["status"]),  # type: ignore[arg-type]
                reason=str(row["reason"]),
                relevance_score=float(row["relevance_score"]),
                importance_score=float(row["importance_score"]),
                retrieved_at=str(row["retrieved_at"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def list_collected_sources(self, topic_id: str, limit: int = 100) -> list[CollectedSource]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM collected_sources WHERE topic_id = ?
                ORDER BY importance_score DESC, retrieved_at DESC, id DESC LIMIT ?
                """,
                (topic_id, limit),
            ).fetchall()
        return [
            CollectedSource(
                id=str(row["id"]),
                topic_id=str(row["topic_id"]),
                user_id=str(row["user_id"]),
                title=str(row["title"]),
                url=str(row["url"]),
                snippet=str(row["snippet"]),
                platform=str(row["platform"]),
                provider=str(row["provider"]),
                query=str(row["query_text"]),
                relevance_score=float(row["relevance_score"]),
                importance_score=float(row["importance_score"]),
                retrieved_at=str(row["retrieved_at"]),
            )
            for row in rows
        ]

    def record_daily_briefing(self, briefing: DailyBriefing) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO daily_briefings (
                    id, topic_id, user_id, chat_id, run_date, markdown, package_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic_id, run_date) DO UPDATE SET
                    markdown = excluded.markdown,
                    package_json = excluded.package_json,
                    created_at = excluded.created_at
                """,
                (
                    briefing.id,
                    briefing.topic_id,
                    briefing.user_id,
                    briefing.chat_id,
                    briefing.run_date.isoformat(),
                    briefing.markdown,
                    json.dumps(briefing.model_dump(mode="json"), ensure_ascii=False),
                    briefing.created_at,
                ),
            )

    def record_provider_trace_event(
        self,
        event: ProviderTraceEvent,
        run_id: str | None = None,
        topic_id: str | None = None,
    ) -> str:
        return self.record_provider_trace_events([event], run_id=run_id, topic_id=topic_id)[0]

    def record_provider_trace_events(
        self,
        events: list[ProviderTraceEvent],
        run_id: str | None = None,
        topic_id: str | None = None,
    ) -> list[str]:
        if not events:
            return []
        event_ids = [_new_id("provider") for _event in events]
        created_at = _now_iso()
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO provider_trace_events (
                    id, run_id, topic_id, provider, query_text, status, result_count,
                    reason, error, elapsed_ms, tier, budget_share, checked_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event_id,
                        run_id,
                        topic_id,
                        event.provider,
                        event.query,
                        event.status,
                        event.result_count,
                        event.reason,
                        event.error,
                        event.elapsed_ms,
                        event.tier,
                        event.budget_share,
                        event.checked_at,
                        created_at,
                    )
                    for event_id, event in zip(event_ids, events, strict=True)
                ],
            )
        return event_ids

    def list_provider_trace_events(
        self,
        topic_id: str | None = None,
        run_id: str | None = None,
        limit: int = 200,
    ) -> list[ProviderTraceEvent]:
        query = "SELECT * FROM provider_trace_events"
        conditions: list[str] = []
        parameters: list[object] = []
        if topic_id is not None:
            conditions.append("topic_id = ?")
            parameters.append(topic_id)
        if run_id is not None:
            conditions.append("run_id = ?")
            parameters.append(run_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [
            ProviderTraceEvent(
                provider=str(row["provider"]),
                query=str(row["query_text"]),
                status=str(row["status"]),  # type: ignore[arg-type]
                result_count=int(row["result_count"]),
                reason=row["reason"],
                error=row["error"],
                elapsed_ms=float(row["elapsed_ms"]) if row["elapsed_ms"] is not None else None,
                tier=row["tier"],
                budget_share=float(row["budget_share"]) if row["budget_share"] is not None else None,
                checked_at=str(row["checked_at"]),
            )
            for row in rows
        ]

    def add_layered_memory_item(
        self,
        layer: str,
        kind: str,
        content: str,
        source_id: str,
        confidence: str = "medium",
        status: str = "active",
        user_id: str = "system",
        chat_id: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        item_id = _new_id("layer")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO layered_memory_items (
                    id, layer, kind, content, source_id, confidence, status,
                    user_id, chat_id, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    layer,
                    kind,
                    content,
                    source_id,
                    confidence,
                    status,
                    user_id,
                    chat_id,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    _now_iso(),
                ),
            )
        return item_id

    def list_layered_memory_items(
        self,
        layer: str | None = None,
        user_id: str | None = None,
        chat_id: str | None = None,
        limit: int = 200,
    ) -> list[LayeredMemoryItem]:
        query = "SELECT * FROM layered_memory_items"
        conditions: list[str] = []
        parameters: list[object] = []
        if layer is not None:
            conditions.append("layer = ?")
            parameters.append(layer)
        if user_id is not None and chat_id is not None:
            conditions.append("((user_id = ? AND chat_id = ?) OR (user_id = 'system' AND chat_id = 'system'))")
            parameters.extend([user_id, chat_id])
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at ASC, id ASC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [
            LayeredMemoryItem(
                id=str(row["id"]),
                layer=str(row["layer"]),  # type: ignore[arg-type]
                kind=str(row["kind"]),
                content=str(row["content"]),
                source_id=str(row["source_id"]),
                confidence=str(row["confidence"]),  # type: ignore[arg-type]
                status=str(row["status"]),  # type: ignore[arg-type]
                user_id=str(row["user_id"]),
                chat_id=str(row["chat_id"]),
                metadata=_json_object(row["metadata_json"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def latest_daily_briefing(self, topic_id: str) -> DailyBriefing | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT package_json FROM daily_briefings WHERE topic_id = ? ORDER BY run_date DESC LIMIT 1",
                (topic_id,),
            ).fetchone()
        return DailyBriefing.model_validate_json(row["package_json"]) if row is not None else None

    def get_entity_embedding(self, content_hash: str) -> list[float] | None:
        """Return a cached embedding vector by content hash, or None if not cached."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT embedding_json FROM entity_embeddings WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["embedding_json"])

    def upsert_entity_embedding(
        self,
        content_hash: str,
        provider: str,
        model: str,
        embedding: list[float],
    ) -> None:
        """Cache an embedding vector for later reuse."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO entity_embeddings (content_hash, provider, model, embedding_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    provider = excluded.provider,
                    model = excluded.model,
                    embedding_json = excluded.embedding_json,
                    created_at = datetime('now')
                """,
                (
                    content_hash,
                    provider,
                    model,
                    json.dumps(embedding),
                ),
            )

    def upsert_domain_knowledge_graph(self, graph: DomainKnowledgeGraph) -> dict[str, object]:
        now = _now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO domain_knowledge_graphs (
                    id, name, description, overview, source, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    overview = excluded.overview,
                    source = excluded.source,
                    version = excluded.version,
                    updated_at = excluded.updated_at
                """,
                (
                    graph.id,
                    graph.name,
                    graph.description,
                    graph.overview,
                    graph.source,
                    graph.version,
                    now,
                    now,
                ),
            )
            keep_entity_ids = {entity.id for entity in graph.entities}
            keep_relation_ids = {relation.id for relation in graph.relations}
            if keep_entity_ids:
                connection.execute(
                    "DELETE FROM domain_knowledge_entities WHERE graph_id = ? AND id NOT IN ({})".format(
                        ",".join("?" for _ in keep_entity_ids)
                    ),
                    (graph.id, *sorted(keep_entity_ids)),
                )
            else:
                connection.execute("DELETE FROM domain_knowledge_entities WHERE graph_id = ?", (graph.id,))
            if keep_relation_ids:
                connection.execute(
                    "DELETE FROM domain_knowledge_relations WHERE graph_id = ? AND id NOT IN ({})".format(
                        ",".join("?" for _ in keep_relation_ids)
                    ),
                    (graph.id, *sorted(keep_relation_ids)),
                )
            else:
                connection.execute("DELETE FROM domain_knowledge_relations WHERE graph_id = ?", (graph.id,))
            for entity in graph.entities:
                connection.execute(
                    """
                    INSERT INTO domain_knowledge_entities (
                        id, graph_id, name, entity_type, aliases_json, summary,
                        evidence_refs_json, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(graph_id, id) DO UPDATE SET
                        name = excluded.name,
                        entity_type = excluded.entity_type,
                        aliases_json = excluded.aliases_json,
                        summary = excluded.summary,
                        evidence_refs_json = excluded.evidence_refs_json,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        entity.id,
                        graph.id,
                        entity.name,
                        entity.entity_type,
                        json.dumps(entity.aliases, ensure_ascii=False),
                        entity.summary,
                        json.dumps(entity.evidence_refs, ensure_ascii=False),
                        json.dumps(entity.metadata, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            for relation in graph.relations:
                connection.execute(
                    """
                    INSERT INTO domain_knowledge_relations (
                        id, graph_id, source_entity_id, relation_type, target_entity_id,
                        description, evidence_refs_json, weight, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(graph_id, id) DO UPDATE SET
                        source_entity_id = excluded.source_entity_id,
                        relation_type = excluded.relation_type,
                        target_entity_id = excluded.target_entity_id,
                        description = excluded.description,
                        evidence_refs_json = excluded.evidence_refs_json,
                        weight = excluded.weight,
                        updated_at = excluded.updated_at
                    """,
                    (
                        relation.id,
                        graph.id,
                        relation.source_entity_id,
                        relation.relation_type,
                        relation.target_entity_id,
                        relation.description,
                        json.dumps(relation.evidence_refs, ensure_ascii=False),
                        relation.weight,
                        now,
                        now,
                    ),
                )
        return {
            "id": graph.id,
            "name": graph.name,
            "entities": len(graph.entities),
            "relations": len(graph.relations),
        }

    def list_domain_knowledge_graphs(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    g.*,
                    COUNT(DISTINCT e.id) AS entity_count,
                    COUNT(DISTINCT r.id) AS relation_count
                FROM domain_knowledge_graphs g
                LEFT JOIN domain_knowledge_entities e ON e.graph_id = g.id
                LEFT JOIN domain_knowledge_relations r ON r.graph_id = g.id
                GROUP BY g.id
                ORDER BY g.id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_domain_knowledge_graph(self, graph_id: str) -> DomainKnowledgeGraph | None:
        with self._connect() as connection:
            graph_row = connection.execute(
                "SELECT * FROM domain_knowledge_graphs WHERE id = ?",
                (graph_id,),
            ).fetchone()
            if graph_row is None:
                return None
            entity_rows = connection.execute(
                """
                SELECT * FROM domain_knowledge_entities
                WHERE graph_id = ?
                ORDER BY id ASC
                """,
                (graph_id,),
            ).fetchall()
            relation_rows = connection.execute(
                """
                SELECT * FROM domain_knowledge_relations
                WHERE graph_id = ?
                ORDER BY id ASC
                """,
                (graph_id,),
            ).fetchall()
        return DomainKnowledgeGraph(
            id=str(graph_row["id"]),
            name=str(graph_row["name"]),
            description=str(graph_row["description"]),
            overview=str(graph_row["overview"]),
            source=str(graph_row["source"]),
            version=str(graph_row["version"]),
            entities=[
                DomainKnowledgeEntity(
                    id=str(row["id"]),
                    name=str(row["name"]),
                    entity_type=str(row["entity_type"]),
                    aliases=json.loads(str(row["aliases_json"])),
                    summary=str(row["summary"]),
                    evidence_refs=json.loads(str(row["evidence_refs_json"])),
                    metadata=json.loads(str(row["metadata_json"])),
                )
                for row in entity_rows
            ],
            relations=[
                DomainKnowledgeRelation(
                    id=str(row["id"]),
                    source_entity_id=str(row["source_entity_id"]),
                    relation_type=str(row["relation_type"]),
                    target_entity_id=str(row["target_entity_id"]),
                    description=str(row["description"]),
                    evidence_refs=json.loads(str(row["evidence_refs_json"])),
                    weight=float(row["weight"]),
                )
                for row in relation_rows
            ],
        )

    @staticmethod
    def _topic_from_row(row: sqlite3.Row) -> TopicSubscription:
        try:
            source_recipe = json.loads(str(row["source_recipe_json"] or "{}"))
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            source_recipe = {}
        if not isinstance(source_recipe, dict):
            source_recipe = {}
        return TopicSubscription(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            chat_id=str(row["chat_id"]),
            topic=str(row["topic"]),
            enabled=bool(row["enabled"]),
            source_recipe={str(key): float(value) for key, value in source_recipe.items() if _is_number(value)},
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _insert_trajectory_log(self, connection: sqlite3.Connection, package: AnswerPackage) -> None:
        interaction = connection.execute(
            "SELECT user_id, chat_id, text, source FROM interactions WHERE id = ?",
            (package.question_id,),
        ).fetchone()
        context = package.trajectory_context
        payload = {
            "question_id": package.question_id,
            "question": str(interaction["text"]) if interaction is not None else "",
            "classification": package.classification,
            "search_record": package.search_record.model_dump(mode="json") if package.search_record else None,
            "draft_answer": context.get("draft_answer"),
            "calibration": package.calibration,
            "review": package.review,
            "final_answer": package.answer_text,
            "verified_claims": [claim.model_dump(mode="json") for claim in package.verified_claims],
            "unverified_claims": package.unverified_claims,
            "active_skills": context.get("active_skills", []),
            "answer_strategy": context.get("answer_strategy", {}),
            "runtime_metadata": context.get("runtime_metadata", {}),
            "execution_flags": context.get("execution_flags", {}),
        }
        connection.execute(
            """
            INSERT OR IGNORE INTO trajectory_logs (
                id, question_id, user_id, chat_id, source, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _new_id("traj"),
                package.question_id,
                str(interaction["user_id"]) if interaction is not None else "system",
                str(interaction["chat_id"]) if interaction is not None else "system",
                str(interaction["source"]) if interaction is not None else "direct-store",
                json.dumps(payload, ensure_ascii=False),
                _now_iso(),
            ),
        )

    @staticmethod
    def _trajectory_row(row: sqlite3.Row) -> dict[str, Any]:
        return {**dict(row), "payload": json.loads(row["payload_json"])}

    @staticmethod
    def _domain_candidate_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            **dict(row),
            "evidence": json.loads(row["evidence_json"]),
            "contradictions": json.loads(row["contradictions_json"]),
            "source_ids": json.loads(row["source_ids_json"]),
        }

    @staticmethod
    def _json_row(row: sqlite3.Row, fields: tuple[str, ...]) -> dict[str, Any]:
        data = dict(row)
        for field in fields:
            json_field = f"{field}_json"
            if json_field in data:
                data[field] = json.loads(str(data.pop(json_field)))
        return data

    @staticmethod
    def _insert_candidate_event(
        connection: sqlite3.Connection,
        candidate_id: str,
        from_status: str | None,
        to_status: str,
        reason: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO domain_knowledge_candidate_events (
                id, candidate_id, from_status, to_status, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (_new_id("knowledge_event"), candidate_id, from_status, to_status, reason, _now_iso()),
        )

    def _insert_missing_verified_claims(
        self,
        connection: sqlite3.Connection,
        question_id: str,
        verified_claims: list[VerifiedClaim],
    ) -> int:
        inserted = 0
        for claim in verified_claims:
            existing = connection.execute(
                """
                SELECT 1 FROM evidence
                WHERE question_id = ? AND claim = ? AND verdict = ? AND COALESCE(source, '') = COALESCE(?, '')
                LIMIT 1
                """,
                (question_id, claim.claim, claim.verdict, claim.source),
            ).fetchone()
            if existing is not None:
                continue
            connection.execute(
                """
                INSERT INTO evidence (
                    id, question_id, claim, verdict, source, checked_at, notes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _new_id("ev"),
                    question_id,
                    claim.claim,
                    claim.verdict,
                    claim.source,
                    claim.checked_at,
                    claim.notes,
                    _now_iso(),
                ),
            )
            inserted += 1
        return inserted

    def _delete_stale_verified_claims(
        self,
        connection: sqlite3.Connection,
        question_id: str,
        verified_claims: list[VerifiedClaim],
    ) -> None:
        current_keys = {
            (claim.claim, claim.verdict, claim.source or "")
            for claim in verified_claims
        }
        rows = connection.execute(
            """
            SELECT id, claim, verdict, COALESCE(source, '') AS source
            FROM evidence
            WHERE question_id = ?
            """,
            (question_id,),
        ).fetchall()
        stale_ids = [
            row["id"]
            for row in rows
            if (row["claim"], row["verdict"], row["source"]) not in current_keys
        ]
        if not stale_ids:
            return
        connection.executemany(
            "DELETE FROM evidence WHERE id = ?",
            [(evidence_id,) for evidence_id in stale_ids],
        )
