import json
import sqlite3
from pathlib import Path

import pytest

from search_assistant.memory.store import MemoryStore


def test_agentops_audit_records_are_queryable_and_immutable(tmp_path):
    database_path = tmp_path / "assistant.sqlite3"
    store = MemoryStore(database_path)
    store.initialize()

    ledger_id = store.add_project_ledger_entry(
        entry_type="test_run",
        subject="agentops smoke",
        status="completed",
        summary="Recorded an auditable local smoke test.",
        evidence_refs=["trace:one"],
        risk="unit risk",
        rollback="delete isolated data directory",
        metadata={"sources": 2},
    )
    gate_id = store.add_gate_record(
        gate_type="candidate_validation",
        subject_type="domain_knowledge_candidate",
        subject_id="knowledge-1",
        result="passed",
        reason="two sources and no contradictions",
        evidence_refs=["source:one", "source:two"],
        metadata={"reviewer": "unit"},
    )
    trace_id = store.add_trace_event(
        run_id="run-1",
        event_type="briefing",
        name="collect_sources",
        status="completed",
        duration_ms=12.5,
        metadata={"query_count": 3},
    )
    snapshot_id = store.add_project_ledger_snapshot(
        project_id="agentops-test",
        objective="Keep project state queryable.",
        phase="unit-test",
        status="active",
        next_decision="ship if tests pass",
        open_blockers=[],
        constraints={"secrets": "never log"},
        decisions=[{"id": "D1", "decision": "use append-only snapshots"}],
        evidence_refs=["README.md"],
    )
    checkpoint_id = store.add_run_checkpoint(
        run_id="run-1",
        workflow="daily_briefing",
        subject="agentops smoke",
        step="sources_collected",
        status="completed",
        payload={"source_count": 4},
    )
    health_id = store.record_search_provider_health(
        run_id="run-1",
        requested_platform="GitHub",
        provider="mcp:public:github",
        query="agentops github",
        ok=True,
        result_count=4,
        duration_ms=20.0,
    )

    assert store.diagnostic_counts()["project_ledger_entries"] == 1
    assert store.diagnostic_counts()["project_ledger_snapshots"] == 1
    assert store.diagnostic_counts()["agentops_run_checkpoints"] == 1
    assert store.list_project_ledger_entries()[0]["evidence_refs"] == ["trace:one"]
    assert store.latest_project_ledger_snapshot("agentops-test")["constraints"] == {"secrets": "never log"}
    assert store.list_gate_records()[0]["metadata"] == {"reviewer": "unit"}
    assert store.list_trace_events()[0]["metadata"] == {"query_count": 3}
    assert store.list_run_checkpoints(run_id="run-1")[0]["payload"] == {"source_count": 4}
    health = store.list_search_provider_health()[0]
    assert health["ok"] is True
    assert health["result_count"] == 4
    assert store.search_provider_health_summary()["providers"] == [
        {
            "provider": "mcp:public:github",
            "attempts": 1,
            "successes": 1,
            "results": 4,
            "errors": 0,
            "empty_results": 0,
        }
    ]

    with sqlite3.connect(database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="project ledger entries are immutable"):
            connection.execute("UPDATE project_ledger_entries SET status = 'changed' WHERE id = ?", (ledger_id,))
        with pytest.raises(sqlite3.IntegrityError, match="project ledger snapshots are immutable"):
            connection.execute("UPDATE project_ledger_snapshots SET status = 'changed' WHERE id = ?", (snapshot_id,))
        with pytest.raises(sqlite3.IntegrityError, match="agentops gate records are immutable"):
            connection.execute("UPDATE agentops_gate_records SET result = 'failed' WHERE id = ?", (gate_id,))
        with pytest.raises(sqlite3.IntegrityError, match="agentops trace events are immutable"):
            connection.execute("UPDATE agentops_trace_events SET status = 'failed' WHERE id = ?", (trace_id,))
        with pytest.raises(sqlite3.IntegrityError, match="agentops run checkpoints are immutable"):
            connection.execute("UPDATE agentops_run_checkpoints SET status = 'failed' WHERE id = ?", (checkpoint_id,))
        with pytest.raises(sqlite3.IntegrityError, match="search provider health records are immutable"):
            connection.execute("UPDATE search_provider_health SET ok = 0 WHERE id = ?", (health_id,))


def test_cli_agentops_report_and_audit_lists_emit_sanitized_json(tmp_path, capsys):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.add_project_ledger_entry(
        entry_type="manual",
        subject="local test",
        status="completed",
        summary="A manual audit record.",
        evidence_refs=["evidence:1"],
        metadata={"safe": True},
    )
    store.add_gate_record(
        gate_type="manual_gate",
        subject_type="manual",
        subject_id="subject-1",
        result="passed",
        reason="reviewed",
    )
    store.add_trace_event("run-1", "manual", "step", "completed", metadata={"safe": True})
    store.add_project_ledger_snapshot(
        project_id="pydantic-ai-tech-intel-briefing",
        objective="AgentOps readiness",
        phase="unit-test",
        status="active",
        next_decision="inspect report",
    )
    store.add_run_checkpoint("run-1", "manual", "local test", "step", "completed", {"safe": True})
    store.record_search_provider_health(
        "run-1",
        requested_platform="web",
        provider="browser-bing",
        query="agentops readiness",
        ok=True,
        result_count=2,
    )

    assert cli.main(["agentops-report", "--data-dir", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["counts"]["project_ledger_entries"] == 1
    assert report["dependency_lock"]["has_pydantic_ai_slim_pin"] is True
    assert report["latest_project_state"]["phase"] == "unit-test"
    assert report["gate_summary"]["passed"] == 1
    assert report["checkpoint_summary"]["total_sampled"] == 1
    assert report["provider_health"]["records"] == 1

    assert cli.main(["provider-health", "--data-dir", str(tmp_path)]) == 0
    provider_output = json.loads(capsys.readouterr().out)
    assert provider_output["summary"]["platforms"][0]["requested_platform"] == "web"

    assert cli.main(["ledger-list", "--data-dir", str(tmp_path)]) == 0
    ledger_output = json.loads(capsys.readouterr().out)
    assert ledger_output[0]["subject"] == "local test"

    assert cli.main(["ledger-state", "--data-dir", str(tmp_path)]) == 0
    state_output = json.loads(capsys.readouterr().out)
    assert state_output["latest"]["objective"] == "AgentOps readiness"

    assert cli.main(["checkpoint-list", "--data-dir", str(tmp_path)]) == 0
    checkpoint_output = json.loads(capsys.readouterr().out)
    assert checkpoint_output[0]["step"] == "step"


def test_dependency_constraints_pin_the_backtracking_sensitive_pydantic_ai_version():
    root = Path(__file__).resolve().parents[1]

    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    constraints = (root / "constraints-dev.txt").read_text(encoding="utf-8")

    assert '"pydantic-ai-slim[openai,mcp]==1.107.1"' in pyproject
    assert '"openai==2.53.0"' in pyproject
    assert "pydantic-ai-slim==1.107.1" in constraints
    assert "openai==2.53.0" in constraints
    assert "agent-framework-core" not in pyproject
    assert "agent-framework-openai" not in pyproject
    assert "agent-framework-core" not in constraints
    assert "agent-framework-openai" not in constraints


def test_business_code_does_not_import_agent_framework():
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import agent_framework" in text or "from agent_framework" in text:
            offenders.append(str(path.relative_to(root)))

    assert offenders == []
