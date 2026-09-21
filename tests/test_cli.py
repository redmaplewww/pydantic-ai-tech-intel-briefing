import os
import json
import sqlite3
import subprocess
import sys
from io import BytesIO, TextIOWrapper
from pathlib import Path

from search_assistant.contracts import AnswerPackage, SourceEvidence
from search_assistant.memory.store import MemoryStore
from search_assistant.skills.service import SkillDraftService


def test_cli_ask_returns_answer_package(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    env["SEARCH_ASSISTANT_MODEL_PROVIDER"] = "fake"
    env["SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME"] = "true"
    env["SEARCH_ASSISTANT_SEARCH_PROVIDER"] = "fake"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "search_assistant.cli",
            "ask",
            "hello",
            "--data-dir",
            str(tmp_path),
        ],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        env=env,
    )

    assert "answer_text" in result.stdout
    assert "classification" in result.stdout


def test_cli_import_does_not_load_lark_sdk():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import search_assistant.cli; print('lark_oapi' in sys.modules)",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    assert result.stdout.strip() == "False"


def test_cli_feishu_long_connection_starts_runner(monkeypatch, tmp_path):
    from search_assistant import cli

    called = {}

    def fake_run_long_connection(data_dir=None):
        called["data_dir"] = data_dir

    monkeypatch.setattr(cli, "run_long_connection", fake_run_long_connection, raising=False)

    assert cli.main(["feishu-long-connection", "--data-dir", str(tmp_path)]) == 0
    assert called["data_dir"] == tmp_path


def test_cli_ask_writes_unicode_json_as_utf8(monkeypatch, tmp_path):
    from search_assistant import cli

    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    package = AnswerPackage(
        question_id="q-cn",
        answer_text="中文回答",
        classification="research",
        confidence="medium",
    )

    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(cli, "_answer_cli_question", lambda store, question: package)

    assert cli.main(["ask", "CXL是什么？", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    assert "中文回答" in stream.buffer.getvalue().decode("utf-8")


def test_cli_feishu_doctor_outputs_sanitized_json(monkeypatch):
    from search_assistant import cli

    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    report = {
        "ok": True,
        "checks": [
            {"name": "credentials", "ok": True, "detail": "tenant token ok"},
            {"name": "bot_info", "ok": True, "detail": "bot active", "data": {"app_name": "智能搜索助手"}},
            {"name": "long_connection_endpoint", "ok": True, "detail": "endpoint generated"},
        ],
        "hints": [],
    }

    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(cli, "run_feishu_diagnostics", lambda settings=None: report, raising=False)

    assert cli.main(["feishu-doctor"]) == 0
    stream.flush()
    output = stream.buffer.getvalue().decode("utf-8")
    assert "智能搜索助手" in output
    assert "tenant token ok" in output
    assert "app_secret" not in output.lower()
    assert "tenant-token" not in output


def test_cli_doctor_outputs_readiness_json(monkeypatch, tmp_path):
    from search_assistant import cli

    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    report = {
        "ok": True,
        "checks": [{"name": "model_runtime", "ok": True, "detail": "deepseek configured"}],
        "hints": [],
    }

    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(cli, "run_readiness_diagnostics", lambda store, data_dir, settings=None: report)

    assert cli.main(["doctor", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))
    assert output["ok"] is True
    assert output["checks"][0]["name"] == "model_runtime"


def test_cli_source_contracts_and_topic_recipe_commands(monkeypatch, tmp_path):
    from search_assistant import cli

    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["source-contracts", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    contracts = json.loads(stream.buffer.getvalue().decode("utf-8"))
    assert any(contract["slug"] == "bilibili" for contract in contracts)

    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)
    assert (
        cli.main(
            [
                "topic-recipe-set",
                "industrial AI",
                '{"web": 3, "bili": 1}',
                "--data-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))
    weights = {item["slug"]: item["weight"] for item in output["source_recipe"]["sources"]}
    assert weights["general-web"] == 3.0
    assert weights["bilibili"] == 1.0


def test_cli_feishu_fixture_updates_profile_snapshot(monkeypatch, tmp_path):
    from search_assistant import cli

    fixture = tmp_path / "event.json"
    fixture.write_text(
        json.dumps(
            {
                "schema": "2.0",
                "header": {"event_id": "evt_profile", "event_type": "im.message.receive_v1"},
                "event": {
                    "message": {
                        "message_id": "om_profile",
                        "chat_id": "oc_profile",
                        "content": json.dumps({"text": "Feishu bot profile replay"}, ensure_ascii=False),
                    },
                    "sender": {"sender_id": {"open_id": "ou_profile"}},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_ASSISTANT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME", "true")

    assert cli.main(["feishu-fixture", str(fixture), "--data-dir", str(tmp_path)]) == 0

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    snapshots = store.list_profile_snapshots()
    assert snapshots
    assert "Feishu integration" in snapshots[0]["summary_json"]


def test_cli_feishu_fixture_accepts_utf8_bom(monkeypatch, tmp_path):
    from search_assistant import cli

    fixture = tmp_path / "event-bom.json"
    fixture.write_text(
        json.dumps(
            {
                "schema": "2.0",
                "header": {"event_id": "evt_bom", "event_type": "im.message.receive_v1"},
                "event": {
                    "message": {
                        "message_id": "om_bom",
                        "chat_id": "oc_bom",
                        "content": json.dumps({"text": "Feishu bot BOM replay"}, ensure_ascii=False),
                    },
                    "sender": {"sender_id": {"open_id": "ou_bom"}},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8-sig",
    )
    monkeypatch.setenv("SEARCH_ASSISTANT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME", "true")

    assert cli.main(["feishu-fixture", str(fixture), "--data-dir", str(tmp_path)]) == 0


def test_cli_feishu_status_outputs_counts_and_reply_attempts(monkeypatch, tmp_path):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    question_id = store.record_interaction(
        cli.IncomingMessage(
            message_id="om_status_latest",
            event_id="evt_status_latest",
            user_id="ou_status",
            chat_id="oc_status",
            text="status should show this latest Feishu message without raw event secrets",
            source="feishu-long-connection",
            raw_event={"token": "secret_should_not_print"},
        )
    )
    store.record_reply_attempt(
        question_id="q-1",
        message_id="om_1",
        channel="feishu-long-connection",
        status="failure",
        response={"sender": {"id": "cli_should_not_print"}},
        error="permission denied",
    )
    store.record_runtime_session(
        kind="feishu-long-connection",
        process_id=4321,
        status="started",
        metadata={"model_provider": "deepseek", "deepseek_api_key_configured": True},
    )
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["feishu-status", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert output["counts"]["reply_attempts"] == 1
    assert output["counts"]["runtime_sessions"] == 1
    assert output["latest_reply_attempts"][0]["status"] == "failure"
    assert output["latest_reply_attempts"][0]["error"] == "permission denied"
    assert output["latest_reply_attempts"][0]["response_recorded"] is True
    assert "response_json" not in output["latest_reply_attempts"][0]
    assert "cli_should_not_print" not in json.dumps(output)
    assert output["latest_runtime_sessions"][0]["process_id"] == 4321
    assert output["latest_interactions"][0]["question_id"] == question_id
    assert output["latest_interactions"][0]["message_id"] == "om_status_latest"
    assert output["latest_interactions"][0]["source"] == "feishu-long-connection"
    assert output["latest_interactions"][0]["chat_id"] == "oc_status"
    assert output["latest_interactions"][0]["answer_id"] is None
    assert output["latest_interactions"][0]["text_preview"].startswith("status should show")
    assert "raw_event_json" not in output["latest_interactions"][0]
    assert "secret_should_not_print" not in json.dumps(output)


def test_cli_feishu_reply_probe_uses_latest_interaction_and_records_attempt(monkeypatch, tmp_path):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    question_id = store.record_interaction(
        cli.IncomingMessage(
            message_id="om_probe",
            event_id="evt_probe",
            user_id="u-1",
            chat_id="c-1",
            text="probe source",
            source="feishu-long-connection",
        )
    )

    class ProbeClient:
        def reply_text(self, message_id, text):
            assert message_id == "om_probe"
            assert text == "probe text"
            return {"message_id": message_id, "response": {"message_id": "reply_probe"}}

    monkeypatch.setattr(cli, "feishu_client_from_settings", lambda settings: ProbeClient())
    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret")
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["feishu-reply-probe", "--latest", "--text", "probe text", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))
    attempts = store.list_reply_attempts()

    assert output["ok"] is True
    assert output["message_id"] == "om_probe"
    assert output["question_id"] == question_id
    assert attempts[0]["channel"] == "feishu-probe"
    assert attempts[0]["status"] == "success"


def test_cli_feishu_wait_reply_succeeds_for_long_connection_attempt(monkeypatch, tmp_path):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_reply_attempt("q-probe", "om-probe", "feishu-probe", "success")
    store.record_reply_attempt("q-live", "om-live", "feishu-long-connection", "success")
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["feishu-wait-reply", "--timeout-seconds", "0", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert output["ok"] is True
    assert output["attempt"]["channel"] == "feishu-long-connection"
    assert output["attempt"]["message_id"] == "om-live"


def test_cli_feishu_wait_reply_ignores_attempts_before_latest_runtime(monkeypatch, tmp_path):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_reply_attempt("q-old", "om-old", "feishu-long-connection", "success")
    runtime_id = store.record_runtime_session(
        kind="feishu-long-connection",
        process_id=1234,
        status="started",
        metadata={"model_provider": "deepseek"},
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE reply_attempts SET created_at = ? WHERE message_id = ?",
            ("2026-07-03T10:00:00+00:00", "om-old"),
        )
        connection.execute(
            "UPDATE runtime_sessions SET created_at = ? WHERE id = ?",
            ("2026-07-03T11:00:00+00:00", runtime_id),
        )
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert (
        cli.main(
            [
                "feishu-wait-reply",
                "--timeout-seconds",
                "0",
                "--poll-seconds",
                "0",
                "--data-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert output["ok"] is False
    assert output["not_before"] == "2026-07-03T11:00:00+00:00"
    assert output["latest_attempt"]["message_id"] == "om-old"


def test_cli_feishu_wait_reply_times_out_without_long_connection_attempt(monkeypatch, tmp_path):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_reply_attempt("q-probe", "om-probe", "feishu-probe", "success")
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert (
        cli.main(
            [
                "feishu-wait-reply",
                "--timeout-seconds",
                "0",
                "--poll-seconds",
                "0",
                "--data-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert output["ok"] is False
    assert output["latest_attempt"]["channel"] == "feishu-probe"


def test_cli_feishu_poll_once_uses_latest_chat(monkeypatch, tmp_path):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_interaction(
        cli.IncomingMessage(
            message_id="om_seed",
            event_id="evt_seed",
            user_id="ou_seed",
            chat_id="oc_seed",
            text="seed",
            source="feishu-long-connection",
        )
    )
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(cli, "_run_feishu_poll_once", lambda store, chat_id, lookback_seconds: {"chat_id": chat_id, "processed": 0})

    assert cli.main(["feishu-poll-once", "--latest-chat", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert output["chat_id"] == "oc_seed"


def test_cli_feishu_poll_once_latest_chat_skips_non_feishu_interactions(monkeypatch, tmp_path):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_interaction(
        cli.IncomingMessage(
            message_id="om_feishu",
            event_id="evt_feishu",
            user_id="ou_feishu",
            chat_id="oc_feishu",
            text="feishu question",
            source="feishu-long-connection",
        )
    )
    store.record_interaction(
        cli.IncomingMessage(
            message_id="cli_latest",
            event_id="cli_latest",
            user_id="local",
            chat_id="local",
            text="local eval question",
            source="cli",
        )
    )
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(cli, "_run_feishu_poll_once", lambda store, chat_id, lookback_seconds: {"chat_id": chat_id, "processed": 0})

    assert cli.main(["feishu-poll-once", "--latest-chat", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert output["chat_id"] == "oc_feishu"


def test_cli_eval_suite_runs_questions_file(monkeypatch, tmp_path):
    from search_assistant import cli

    questions = tmp_path / "questions.txt"
    questions.write_text("What is CXL?\nWhat is Feishu bot verification?\n", encoding="utf-8")
    monkeypatch.setenv("SEARCH_ASSISTANT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME", "true")
    monkeypatch.setenv("SEARCH_ASSISTANT_SEARCH_PROVIDER", "duckduckgo")

    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(cli, "search_client_from_settings", lambda settings: EmptySearchClient())

    assert cli.main(["eval-suite", "--questions", str(questions), "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert output["total_questions"] == 2
    assert output["summary"]["answers_recorded"] == 2
    assert output["summary"]["learning_report_written"] is True


def test_cli_eval_suite_accepts_inline_json_questions(monkeypatch, tmp_path):
    from search_assistant import cli

    monkeypatch.setenv("SEARCH_ASSISTANT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME", "true")
    monkeypatch.setenv("SEARCH_ASSISTANT_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(cli, "search_client_from_settings", lambda settings: EmptySearchClient())
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)
    questions_json = json.dumps(["What is CXL?", "分布式大模型是什么？"], ensure_ascii=False)

    assert cli.main(["eval-suite", "--questions", questions_json, "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert output["total_questions"] == 2
    assert [item["question"] for item in output["items"]] == ["What is CXL?", "分布式大模型是什么？"]


def test_cli_eval_suite_uses_default_questions(monkeypatch, tmp_path):
    from search_assistant import cli

    monkeypatch.setenv("SEARCH_ASSISTANT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME", "true")
    monkeypatch.setattr(cli, "search_client_from_settings", lambda settings: EmptySearchClient())
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["eval-suite", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert output["total_questions"] == 12  # layered pool is the default
    assert any("KV Cache" in item["question"] for item in output["items"])
    assert any("MCP" in item["question"] or "RAG" in item["question"] for item in output["items"])


def test_cli_eval_suite_limits_default_questions_with_max_questions(monkeypatch, tmp_path):
    from search_assistant import cli

    monkeypatch.setenv("SEARCH_ASSISTANT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME", "true")
    monkeypatch.setattr(cli, "search_client_from_settings", lambda settings: EmptySearchClient())
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["eval-suite", "--max-questions", "1", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    assert output["total_questions"] == 1
    assert len(output["items"]) == 1
    assert "KV Cache" in output["items"][0]["question"]
    assert len(store.list_answers()) == 1


def test_cli_eval_replay_reruns_previous_evaluation_questions(monkeypatch, tmp_path):
    from search_assistant import cli

    monkeypatch.setenv("SEARCH_ASSISTANT_MODEL_PROVIDER", "fake")
    monkeypatch.setenv("SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME", "true")
    monkeypatch.setattr(cli, "search_client_from_settings", lambda settings: EmptySearchClient())
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()
    report = {
        "total_questions": 1,
        "items": [
            {
                "index": 1,
                "question": "What is CXL?",
                "question_id": "q-old",
                "answer_excerpt": "old answer",
                "source_urls": [],
                "quality_flags": [],
            }
        ],
        "summary": {
            "flagged_answers": 0,
            "review_rejected_answers": 0,
            "result_failed_answers": 0,
            "process_flagged_answers": 0,
        },
    }
    (eval_dir / "evaluation-report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["eval-replay", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    assert output["ok"] is True
    assert output["replayed"] == 1
    assert Path(output["replay_report_path"]).exists()
    assert store.list_gate_records(gate_type="evaluation_replay")[0]["result"] == "passed"
    assert store.list_project_ledger_entries(entry_type="evaluation_replay")[0]["status"] == "completed"


def test_cli_skill_draft_uses_existing_profile_sources(monkeypatch, tmp_path):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    profile_id = store.add_profile_snapshot(
        {
            "recurring_topics": ["Feishu integration"],
            "preferred_answer_style": "precise answers with verification notes",
        }
    )

    assert cli.main(["skill-draft", "Feishu Reply Practice", "--data-dir", str(tmp_path)]) == 0

    drafts = store.list_skill_drafts()
    assert drafts[0]["source_ids_json"] == json.dumps([profile_id])


def test_cli_skill_promote_outputs_active_skill_path_and_updates_status(monkeypatch, tmp_path):
    from search_assistant import cli
    from search_assistant.skills.service import SkillDraftService

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts").create_from_experience(
        "Feishu Reply Practice",
        source_ids=["manual"],
    )
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["skill-promote", "feishu-reply-practice", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    active_path = tmp_path / "skills" / "active" / "feishu-reply-practice" / "SKILL.md"
    assert output == {
        "status": "promoted",
        "name": "Feishu Reply Practice",
        "path": str(active_path),
    }
    assert active_path.exists()
    assert store.list_skill_drafts()[0]["review_status"] == "promoted"


def test_cli_skill_list_outputs_review_status_counts(monkeypatch, tmp_path):
    from search_assistant import cli
    from search_assistant.skills.service import SkillDraftService

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = SkillDraftService(
        store,
        drafts_dir=tmp_path / "skills" / "drafts",
        active_dir=tmp_path / "skills" / "active",
    )
    service.create_from_experience("Draft Only Guidance", source_ids=["manual"])
    service.create_from_experience("Feishu Reply Practice", source_ids=["manual"])
    service.promote("feishu-reply-practice")
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["skill-list", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert output["counts"] == {
        "total": 2,
        "draft": 1,
        "promoted": 1,
        "active_files": 1,
        "missing_files": 0,
    }
    assert [item["slug"] for item in output["skills"]] == ["draft-only-guidance", "feishu-reply-practice"]
    assert output["skills"][1]["review_status"] == "promoted"
    assert output["skills"][1]["active"] is True


def test_cli_evidence_backfill_updates_historical_answers(monkeypatch, tmp_path):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_answer(
        AnswerPackage(
            question_id="q-backfill",
            answer_text=(
                "| 来源 | 确认了什么 | 缺少什么 |\n"
                "|------|-----------|----------|\n"
                "| NVIDIA DGX Spark 官方规格 | 128GB 统一内存、ConnectX-7 网卡 | 无性能实测 |\n"
            ),
            classification="research",
            confidence="medium",
            sources=[
                SourceEvidence(
                    title="NVIDIA DGX Spark 官方规格",
                    url="https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
                    snippet="NVIDIA DGX Spark 官方规格确认 128GB 统一内存、ConnectX-7 网卡。",
                    provider="direct-official",
                    checked_at="2026-07-03T00:00:00Z",
                )
            ],
        )
    )
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["evidence-backfill", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()

    assert output["answers_scanned"] == 1
    assert output["evidence_inserted"] == 1
    assert store.list_evidence()[0]["claim"] == "NVIDIA DGX Spark 官方规格 确认 128GB 统一内存、ConnectX-7 网卡"


def test_cli_evolve_generates_report_and_auto_skill_draft(monkeypatch, tmp_path):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.add_profile_snapshot(
        {
            "recurring_topics": ["Feishu integration"],
            "preferred_answer_style": "precise answers with verification notes",
        }
    )
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["evolve", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    assert output["report_path"].endswith("learning-report.md")
    assert output["skill_paths"]
    assert store.list_learning_reports()
    assert store.list_skill_drafts()


def test_cli_evolve_reports_refreshed_review_only_skill_drafts(monkeypatch, tmp_path):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.add_profile_snapshot(
        {
            "recurring_topics": ["AI infrastructure and model deployment"],
            "preferred_answer_style": "precise answers with verification notes",
        }
    )
    service = SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts")
    paths = service.auto_create_from_experience(max_drafts=1)
    draft_path = Path(paths[0])
    draft_path.write_text(
        (
            "---\n"
            "name: ai-infrastructure-and-model-deployment-practice\n"
            "description: old generic draft\n"
            "---\n\n"
            "## Instructions\n\n"
            "Answer similar future questions with explicit assumptions, verification notes, and a short final recommendation.\n"
        ),
        encoding="utf-8",
    )
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert cli.main(["evolve", "--data-dir", str(tmp_path)]) == 0
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    assert output["skill_paths"] == []
    assert output["refreshed_skill_paths"] == [str(draft_path)]
    assert "ModelScope/魔搭/魔塔" in draft_path.read_text(encoding="utf-8")


def test_cli_evolve_loop_runs_periodically_with_max_runs(monkeypatch, tmp_path):
    from search_assistant import cli

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.add_profile_snapshot(
        {
            "recurring_topics": ["Feishu integration"],
            "preferred_answer_style": "precise answers with verification notes",
        }
    )
    sleeps = []
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: sleeps.append(seconds))
    stream = TextIOWrapper(BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)

    assert (
        cli.main(
            [
                "evolve-loop",
                "--data-dir",
                str(tmp_path),
                "--interval-seconds",
                "5",
                "--max-runs",
                "2",
            ]
        )
        == 0
    )
    stream.flush()
    output = json.loads(stream.buffer.getvalue().decode("utf-8"))

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    assert output["runs"] == 2
    assert sleeps == [5.0]
    assert len(store.list_learning_reports()) == 2
    assert len(store.list_skill_drafts()) == 1
    sessions = store.list_runtime_sessions()
    assert len(sessions) == 1
    assert sessions[0]["kind"] == "evolve-loop"
    assert sessions[0]["status"] == "completed"
    metadata = json.loads(sessions[0]["metadata_json"])
    assert metadata["interval_seconds"] == 5.0
    assert metadata["max_runs"] == 2
    assert metadata["runs"] == 2


class EmptySearchClient:
    def search(self, query, limit=5):
        return []
