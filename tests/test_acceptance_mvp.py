from fastapi.testclient import TestClient

from search_assistant.feishu.client import FakeFeishuClient
from search_assistant.server import create_app
from search_assistant.runtime import FakeAgentRuntime


def test_phase_1_acceptance_flow(tmp_path):
    feishu = FakeFeishuClient()
    app = create_app(
        data_dir=tmp_path,
        feishu_client=feishu,
        runtime=FakeAgentRuntime(),
        search_client=EmptySearchClient(),
    )
    client = TestClient(app)
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt_accept", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {
                "message_id": "om_accept",
                "chat_id": "oc_accept",
                "content": "{\"text\":\"Compare current Feishu bot event APIs and Agent Framework workflow risks.\"}",
            },
            "sender": {"sender_id": {"open_id": "ou_accept"}},
        },
    }

    response = client.post("/feishu/events", json=payload)

    assert response.status_code == 200
    assert feishu.replies
    assert "confidence" in _post_text(feishu.replies[0]["post"])


class EmptySearchClient:
    def search(self, query, limit=5):
        return []


def _post_text(post):
    return "\n".join(
        str(element.get("text", ""))
        for line in post["zh_cn"]["content"]
        for element in line
    )
