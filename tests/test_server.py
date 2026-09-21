from fastapi.testclient import TestClient

from search_assistant.feishu.client import FakeFeishuClient
from search_assistant.memory.store import MemoryStore
from search_assistant.server import create_app
from search_assistant.runtime import FakeAgentRuntime


def test_feishu_webhook_replies_with_answer(tmp_path):
    fake_client = FakeFeishuClient()
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    app = create_app(
        store=store,
        feishu_client=fake_client,
        runtime=FakeAgentRuntime(),
        search_client=EmptySearchClient(),
    )
    client = TestClient(app)
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt_1", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {"message_id": "om_1", "chat_id": "oc_1", "content": "{\"text\":\"hello\"}"},
            "sender": {"sender_id": {"open_id": "ou_1"}},
        },
    }

    response = client.post("/feishu/events", json=payload)

    assert response.status_code == 200
    assert fake_client.replies[0]["message_id"] == "om_1"
    assert "post" in fake_client.replies[0]
    attempts = store.list_reply_attempts()
    assert attempts[0]["message_id"] == "om_1"
    assert attempts[0]["status"] == "success"
    assert attempts[0]["channel"] == "feishu"


def test_server_uses_feishu_http_client_when_enabled(monkeypatch, tmp_path):
    selected_client = FakeFeishuClient()

    def fake_http_client(app_id, app_secret):
        selected_client.app_id = app_id
        selected_client.app_secret = app_secret
        return selected_client

    monkeypatch.setenv("SEARCH_ASSISTANT_FEISHU_ENABLED", "true")
    monkeypatch.setenv("FEISHU_APP_ID", "cli_x")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret")
    monkeypatch.setattr("search_assistant.server.FeishuHttpClient", fake_http_client)

    app = create_app(data_dir=tmp_path, runtime=FakeAgentRuntime(), search_client=EmptySearchClient())
    client = TestClient(app)
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt_enabled", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {"message_id": "om_enabled", "chat_id": "oc_1", "content": "{\"text\":\"hello\"}"},
            "sender": {"sender_id": {"open_id": "ou_1"}},
        },
    }

    response = client.post("/feishu/events", json=payload)

    assert response.status_code == 200
    assert selected_client.app_id == "cli_x"
    assert selected_client.app_secret == "secret"
    assert selected_client.replies[0]["message_id"] == "om_enabled"
    assert "post" in selected_client.replies[0]


def test_feishu_webhook_updates_profile_snapshot(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    fake_client = FakeFeishuClient()
    app = create_app(
        store=store,
        feishu_client=fake_client,
        runtime=FakeAgentRuntime(),
        search_client=EmptySearchClient(),
    )
    client = TestClient(app)
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt_profile", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {
                "message_id": "om_profile",
                "chat_id": "oc_profile",
                "content": "{\"text\":\"Feishu bot webhook profile\"}",
            },
            "sender": {"sender_id": {"open_id": "ou_profile"}},
        },
    }

    response = client.post("/feishu/events", json=payload)

    assert response.status_code == 200
    assert store.list_profile_snapshots()


def test_feishu_webhook_skips_duplicate_event_without_replying_again(tmp_path):
    fake_client = FakeFeishuClient()
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    app = create_app(
        store=store,
        feishu_client=fake_client,
        runtime=FakeAgentRuntime(answer_text="deduped answer"),
        search_client=EmptySearchClient(),
    )
    client = TestClient(app)
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt_duplicate", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {"message_id": "om_duplicate", "chat_id": "oc_1", "content": "{\"text\":\"hello\"}"},
            "sender": {"sender_id": {"open_id": "ou_1"}},
        },
    }

    first_response = client.post("/feishu/events", json=payload)
    second_response = client.post("/feishu/events", json=payload)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert second_response.json()["duplicate"] is True
    assert len(fake_client.replies) == 1
    assert len(store.list_reply_attempts()) == 1


def test_feishu_webhook_records_reply_failure(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    app = create_app(
        store=store,
        feishu_client=FailingFeishuClient(),
        runtime=FakeAgentRuntime(answer_text="failure answer"),
        search_client=EmptySearchClient(),
    )
    client = TestClient(app, raise_server_exceptions=False)
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt_reply_failure", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {
                "message_id": "om_reply_failure",
                "chat_id": "oc_profile",
                "content": "{\"text\":\"hello failure\"}",
            },
            "sender": {"sender_id": {"open_id": "ou_profile"}},
        },
    }

    response = client.post("/feishu/events", json=payload)

    assert response.status_code == 502
    attempts = store.list_reply_attempts()
    assert attempts[0]["message_id"] == "om_reply_failure"
    assert attempts[0]["status"] == "failure"
    assert "reply blocked" in attempts[0]["error"]


class EmptySearchClient:
    def search(self, query, limit=5):
        return []


class FailingFeishuClient:
    def reply_text(self, message_id, text):
        raise RuntimeError("reply blocked")
