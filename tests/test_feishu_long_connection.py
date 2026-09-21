import json
import logging
import pytest

from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

from search_assistant.config import Settings
from search_assistant.feishu.client import FakeFeishuClient
from search_assistant.feishu.long_connection import (
    FeishuLongConnectionHandler,
    build_event_handler,
    parse_lark_message_event,
    run_long_connection,
    _redact_log_message,
)
from search_assistant.memory.store import MemoryStore
from search_assistant.runtime import FakeAgentRuntime
from search_assistant.workflow.service import SearchAssistantWorkflow


def test_parse_lark_message_event_from_sdk_object():
    event = P2ImMessageReceiveV1(_receive_payload(text="hello from long connection"))

    message = parse_lark_message_event(event)

    assert message.message_id == "om_long_1"
    assert message.event_id == "evt_long_1"
    assert message.user_id == "ou_long_1"
    assert message.chat_id == "oc_long_1"
    assert message.text == "hello from long connection"
    assert message.source == "feishu-long-connection"


def test_long_connection_event_handler_replies_and_updates_profile(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    feishu_client = FakeFeishuClient()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=FakeAgentRuntime(answer_text="Feishu long connection answer."),
        search_client=EmptySearchClient(),
    )
    handler = FeishuLongConnectionHandler(
        workflow=workflow,
        feishu_client=feishu_client,
        store=store,
    )
    event_handler = build_event_handler(handler)

    event_handler._do_without_validation(
        json.dumps(
            _receive_payload(
                text="I want a Feishu bot learning direction with Agent Framework and verification.",
                event_id="evt_long_learning",
                message_id="om_long_learning",
            ),
            ensure_ascii=False,
        ).encode("utf-8")
    )

    assert feishu_client.replies[0]["message_id"] == "om_long_learning"
    assert "Feishu long connection answer." in _post_text(feishu_client.replies[0]["post"])
    assert "classification:" in _post_text(feishu_client.replies[0]["post"])
    stored_answer = store.list_answers()[0].answer_text
    assert stored_answer.startswith("Feishu long connection answer.")
    assert "搜索记录:" in stored_answer
    assert store.list_profile_snapshots()
    assert any(item["content"] == "Feishu integration" for item in store.list_memory_items())
    attempts = store.list_reply_attempts()
    assert attempts[0]["message_id"] == "om_long_learning"
    assert attempts[0]["status"] == "success"
    assert attempts[0]["channel"] == "feishu-long-connection"


def test_long_connection_handler_skips_duplicate_event_without_replying_again(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    feishu_client = FakeFeishuClient()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=FakeAgentRuntime(answer_text="deduped answer"),
        search_client=EmptySearchClient(),
    )
    handler = FeishuLongConnectionHandler(
        workflow=workflow,
        feishu_client=feishu_client,
        store=store,
    )
    event = P2ImMessageReceiveV1(
        _receive_payload(text="duplicate event", event_id="evt_duplicate", message_id="om_duplicate")
    )

    handler.handle_message_event(event)
    handler.handle_message_event(event)

    assert len(feishu_client.replies) == 1
    assert len(store.list_reply_attempts()) == 1


def test_long_connection_handler_skips_duplicate_message_id_with_new_event_id(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    feishu_client = FakeFeishuClient()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=FakeAgentRuntime(answer_text="deduped answer"),
        search_client=EmptySearchClient(),
    )
    handler = FeishuLongConnectionHandler(
        workflow=workflow,
        feishu_client=feishu_client,
        store=store,
    )

    handler.handle_message_event(
        P2ImMessageReceiveV1(
            _receive_payload(text="same message", event_id="evt_duplicate_1", message_id="om_duplicate")
        )
    )
    handler.handle_message_event(
        P2ImMessageReceiveV1(
            _receive_payload(text="same message", event_id="evt_duplicate_2", message_id="om_duplicate")
        )
    )

    assert len(feishu_client.replies) == 1
    assert len(store.list_reply_attempts()) == 1
    assert len(store.list_interactions()) == 1


def test_long_connection_handler_logs_event_and_reply(caplog, tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    feishu_client = FakeFeishuClient()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=FakeAgentRuntime(answer_text="logged answer"),
        search_client=EmptySearchClient(),
    )
    handler = FeishuLongConnectionHandler(
        workflow=workflow,
        feishu_client=feishu_client,
        store=store,
    )
    event = P2ImMessageReceiveV1(_receive_payload(text="hello log"))

    with caplog.at_level(logging.INFO, logger="search_assistant.feishu.long_connection"):
        handler.handle_message_event(event)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "received Feishu long-connection event" in messages
    assert "evt_long_1" in messages
    assert "om_long_1" in messages
    assert "replied to Feishu message" in messages


def test_long_connection_handler_records_reply_failure(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=FakeAgentRuntime(answer_text="reply failure answer"),
        search_client=EmptySearchClient(),
    )
    handler = FeishuLongConnectionHandler(
        workflow=workflow,
        feishu_client=FailingFeishuClient(),
        store=store,
    )
    event = P2ImMessageReceiveV1(_receive_payload(text="hello failure"))

    with pytest.raises(RuntimeError, match="reply blocked"):
        handler.handle_message_event(event)

    attempts = store.list_reply_attempts()
    assert attempts[0]["message_id"] == "om_long_1"
    assert attempts[0]["status"] == "failure"
    assert "reply blocked" in attempts[0]["error"]


def test_long_connection_handler_records_workflow_failure_before_reply(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    feishu_client = FakeFeishuClient()
    handler = FeishuLongConnectionHandler(
        workflow=FailingWorkflow(),
        feishu_client=feishu_client,
        store=store,
    )
    event = P2ImMessageReceiveV1(_receive_payload(text="hello workflow failure"))

    with pytest.raises(RuntimeError, match="workflow blocked"):
        handler.handle_message_event(event)

    attempts = store.list_reply_attempts()
    assert attempts[0]["message_id"] == "om_long_1"
    assert attempts[0]["question_id"] == "om_long_1"
    assert attempts[0]["status"] == "failure"
    assert "workflow blocked" in attempts[0]["error"]
    assert feishu_client.replies == []


def test_long_connection_handler_retries_duplicate_event_after_reply_failure(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=FakeAgentRuntime(answer_text="retry answer"),
        search_client=EmptySearchClient(),
    )
    feishu_client = FailingOnceFeishuClient()
    handler = FeishuLongConnectionHandler(
        workflow=workflow,
        feishu_client=feishu_client,
        store=store,
    )
    event = P2ImMessageReceiveV1(
        _receive_payload(text="retry duplicate", event_id="evt_retry", message_id="om_retry")
    )

    with pytest.raises(RuntimeError, match="temporary reply failure"):
        handler.handle_message_event(event)
    handler.handle_message_event(
        P2ImMessageReceiveV1(
            _receive_payload(text="retry duplicate", event_id="evt_retry_second", message_id="om_retry")
        )
    )

    assert len(feishu_client.replies) == 1
    attempts = store.list_reply_attempts()
    assert [attempt["status"] for attempt in attempts] == ["success", "failure"]
    assert len(store.list_interactions()) == 1


def test_long_connection_redacts_sdk_websocket_log_secrets():
    access_key_param = "access" + "_key"
    ticket_param = "tick" + "et"
    message = (
        "connected to wss://msg-frontier.feishu.cn/ws/v2?"
        f"{access_key_param}=temporary-access&{ticket_param}=temporary-ticket&device_id=device"
    )

    redacted = _redact_log_message(message)

    assert "temporary-access" not in redacted
    assert "temporary-ticket" not in redacted
    assert f"{access_key_param}=<redacted>" in redacted
    assert f"{ticket_param}=<redacted>" in redacted


def test_run_long_connection_records_runtime_session(monkeypatch, tmp_path):
    from search_assistant.feishu import long_connection

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    handler = FeishuLongConnectionHandler(
        workflow=SearchAssistantWorkflow(
            store=store,
            runtime=FakeAgentRuntime(answer_text="started"),
            search_client=EmptySearchClient(),
        ),
        feishu_client=FakeFeishuClient(),
        store=store,
    )
    started = {}

    class FakeWsClient:
        def __init__(self, **kwargs):
            started["kwargs"] = kwargs

        def start(self):
            started["called"] = True

    monkeypatch.setattr(long_connection, "create_long_connection_handler", lambda **kwargs: handler)
    monkeypatch.setattr(long_connection, "build_event_handler", lambda *args, **kwargs: object())
    monkeypatch.setattr(long_connection.lark.ws, "Client", FakeWsClient)

    run_long_connection(
        data_dir=tmp_path,
        settings=Settings(
            feishu_app_id="cli_test",
            feishu_app_secret="secret",
            model_provider="deepseek",
            search_provider="browser",
            deepseek_api_key="key",
        ),
    )

    sessions = store.list_runtime_sessions()
    assert started["called"] is True
    assert sessions[0]["kind"] == "feishu-long-connection"
    assert sessions[0]["status"] == "started"
    assert "deepseek" in sessions[0]["metadata_json"]
    metadata = json.loads(sessions[0]["metadata_json"])
    assert "key" not in metadata.values()
    assert metadata["deepseek_api_key_configured"] is True
    assert metadata["deepseek_model"] == "deepseek-v4-flash"


def _receive_payload(
    text: str,
    event_id: str = "evt_long_1",
    message_id: str = "om_long_1",
) -> dict:
    return {
        "schema": "2.0",
        "header": {
            "event_id": event_id,
            "event_type": "im.message.receive_v1",
            "create_time": "1700000000000",
            "tenant_key": "tenant_test",
            "app_id": "cli_test",
        },
        "event": {
            "sender": {
                "sender_id": {
                    "open_id": "ou_long_1",
                    "user_id": "user_long_1",
                    "union_id": "union_long_1",
                },
                "sender_type": "user",
                "tenant_key": "tenant_test",
            },
            "message": {
                "message_id": message_id,
                "chat_id": "oc_long_1",
                "chat_type": "p2p",
                "message_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            },
        },
    }


class EmptySearchClient:
    def search(self, query, limit=5):
        return []


class FailingFeishuClient:
    def reply_text(self, message_id, text):
        raise RuntimeError("reply blocked")


class FailingOnceFeishuClient:
    def __init__(self):
        self.calls = 0
        self.replies = []

    def reply_text(self, message_id, text):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary reply failure")
        self.replies.append({"message_id": message_id, "text": text})
        return {"message_id": message_id, "response": {"message_id": "reply_retry"}}


class FailingWorkflow:
    def answer(self, message):
        raise RuntimeError("workflow blocked")


def _post_text(post):
    return "\n".join(
        str(element.get("text", ""))
        for line in post["zh_cn"]["content"]
        for element in line
    )
