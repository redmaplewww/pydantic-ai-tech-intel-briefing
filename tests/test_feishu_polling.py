import json

from search_assistant.feishu.polling import FeishuPollingService
from search_assistant.memory.store import MemoryStore
from search_assistant.runtime import FakeAgentRuntime


def test_polling_service_answers_new_user_messages_and_skips_existing_or_app_messages(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_interaction(
        IncomingMessageFactory.message(
            message_id="om_old",
            text="already seen",
            chat_id="oc_1",
            source="feishu-polling",
        )
    )
    store.record_reply_attempt(
        question_id="q_old",
        message_id="om_old",
        channel="feishu-polling",
        status="success",
    )
    client = PollingFeishuClient(
        [
            _message_item("om_app", "app", "bot message"),
            _message_item("om_old", "user", "already seen"),
            _message_item("om_new", "user", "What is CXL?"),
        ]
    )
    service = FeishuPollingService(
        store=store,
        runtime=FakeAgentRuntime(answer_text="CXL answer"),
        search_client=EmptySearchClient(),
        feishu_client=client,
    )

    result = service.poll_once(chat_id="oc_1", lookback_seconds=3600)

    assert result["processed"] == 1
    assert result["skipped"] == 2
    assert result["skip_reasons"] == {"app_message": 1, "already_replied": 1}
    assert result["message_diagnostics"] == [
        {
            "message_id": "om_new",
            "msg_type": "text",
            "sender_type": "user",
            "skipped_reason": None,
            "text_preview": "What is CXL?",
        },
        {
            "message_id": "om_old",
            "msg_type": "text",
            "sender_type": "user",
            "skipped_reason": "already_replied",
            "text_preview": "already seen",
        },
        {
            "message_id": "om_app",
            "msg_type": "text",
            "sender_type": "app",
            "skipped_reason": "app_message",
            "text_preview": "bot message",
        },
    ]
    assert client.replies[0]["message_id"] == "om_new"
    assert "CXL answer" in _post_text(client.replies[0]["post"])
    attempts = store.list_reply_attempts()
    assert attempts[0]["channel"] == "feishu-polling"
    assert attempts[0]["status"] == "success"
    assert store.has_interaction("om_new")
    assert store.list_profile_snapshots()


def test_polling_service_retries_message_after_reply_failure(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    client = FailingOncePollingFeishuClient([_message_item("om_retry", "user", "retry reply")])
    service = FeishuPollingService(
        store=store,
        runtime=FakeAgentRuntime(answer_text="retry answer"),
        search_client=EmptySearchClient(),
        feishu_client=client,
    )

    first_result = service.poll_once(chat_id="oc_1", lookback_seconds=3600)
    second_result = service.poll_once(chat_id="oc_1", lookback_seconds=3600)

    assert first_result["failed"] == 1
    assert second_result["processed"] == 1
    assert len(client.replies) == 1
    attempts = store.list_reply_attempts()
    assert [attempt["status"] for attempt in attempts] == ["success", "failure"]
    assert len(store.list_interactions()) == 1


def test_polling_service_skips_message_already_seen_by_long_connection(tmp_path):
    from search_assistant.contracts import IncomingMessage

    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_interaction(
        IncomingMessage(
            message_id="om_seen",
            event_id="evt_seen",
            user_id="ou_1",
            chat_id="oc_1",
            text="already handled by long connection",
            source="feishu-long-connection",
        )
    )
    store.record_reply_attempt(
        question_id="q_seen",
        message_id="om_seen",
        channel="feishu-long-connection",
        status="success",
    )
    client = PollingFeishuClient([_message_item("om_seen", "user", "already handled by long connection")])
    service = FeishuPollingService(
        store=store,
        runtime=FakeAgentRuntime(answer_text="duplicate answer"),
        search_client=EmptySearchClient(),
        feishu_client=client,
    )

    result = service.poll_once(chat_id="oc_1", lookback_seconds=3600)

    assert result["processed"] == 0
    assert result["skipped"] == 1
    assert client.replies == []


class IncomingMessageFactory:
    @staticmethod
    def message(message_id, text, chat_id, source):
        from search_assistant.contracts import IncomingMessage

        return IncomingMessage(
            message_id=message_id,
            event_id=None,
            user_id="ou_1",
            chat_id=chat_id,
            text=text,
            source=source,
        )


class PollingFeishuClient:
    def __init__(self, messages):
        self.messages = messages
        self.replies = []

    def list_chat_messages(self, **kwargs):
        return self.messages

    def reply_text(self, message_id, text):
        self.replies.append({"message_id": message_id, "text": text})
        return {"message_id": message_id, "response": {"message_id": "reply_1"}}

    def reply_post(self, message_id, post):
        self.replies.append({"message_id": message_id, "post": post})
        return {"message_id": message_id, "response": {"message_id": "reply_1"}}


class FailingOncePollingFeishuClient(PollingFeishuClient):
    def __init__(self, messages):
        super().__init__(messages)
        self.calls = 0

    def reply_text(self, message_id, text):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary reply failure")
        return super().reply_text(message_id, text)

    def reply_post(self, message_id, post):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary reply failure")
        return super().reply_post(message_id, post)


class EmptySearchClient:
    def search(self, query, limit=5):
        return []


def _message_item(message_id, sender_type, text):
    return {
        "message_id": message_id,
        "chat_id": "oc_1",
        "msg_type": "text",
        "sender": {"sender_type": sender_type, "id": "ou_1" if sender_type == "user" else "cli_test"},
        "body": {"content": json.dumps({"text": text}, ensure_ascii=False)},
        "create_time": "1782586000000",
    }


def _post_text(post):
    return "\n".join(
        str(element.get("text", ""))
        for line in post["zh_cn"]["content"]
        for element in line
    )
