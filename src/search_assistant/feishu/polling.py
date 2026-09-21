from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Protocol

from search_assistant.contracts import IncomingMessage
from search_assistant.feishu.formatting import reply_answer_package
from search_assistant.memory.store import MemoryStore
from search_assistant.profile.service import ProfileService
from search_assistant.search.provider import SearchClient
from search_assistant.runtime import AgentRuntime
from search_assistant.workflow.service import SearchAssistantWorkflow


class PollingFeishuClient(Protocol):
    def list_chat_messages(
        self,
        chat_id: str,
        start_time: int,
        end_time: int,
        page_size: int = 20,
    ) -> list[dict[str, object]]:
        ...

    def reply_text(self, message_id: str, text: str) -> dict[str, object]:
        ...

    def reply_post(self, message_id: str, post: dict[str, object]) -> dict[str, object]:
        ...


class FeishuPollingService:
    def __init__(
        self,
        store: MemoryStore,
        runtime: AgentRuntime,
        search_client: SearchClient,
        feishu_client: PollingFeishuClient,
        search_budget_seconds: float = 45.0,
        admin_user_ids: set[str] | None = None,
    ):
        self.store = store
        self.workflow = SearchAssistantWorkflow(
            store=store,
            runtime=runtime,
            search_client=search_client,
            search_budget_seconds=search_budget_seconds,
            skill_drafts_dir=Path(store.database_path).parent / "skills" / "drafts",
            report_output_dir=Path(store.database_path).parent / "reports",
            admin_user_ids=admin_user_ids,
        )
        self.feishu_client = feishu_client
        self.profile_service = ProfileService(store)

    def poll_once(
        self,
        chat_id: str,
        lookback_seconds: int = 3600,
        page_size: int = 20,
    ) -> dict[str, Any]:
        now = int(time.time())
        items = self.feishu_client.list_chat_messages(
            chat_id=chat_id,
            start_time=now - lookback_seconds,
            end_time=now + 60,
            page_size=page_size,
        )
        processed = 0
        skipped = 0
        failed = 0
        skip_reasons: dict[str, int] = {}
        message_diagnostics: list[dict[str, object]] = []
        for item in reversed(items):
            message, skipped_reason, diagnostic = _message_from_item_with_diagnostic(
                item,
                fallback_chat_id=chat_id,
            )
            if message is not None and self.store.has_successful_reply(message.message_id):
                skipped_reason = "already_replied"
                diagnostic["skipped_reason"] = skipped_reason
            message_diagnostics.append(diagnostic)
            if message is None or skipped_reason is not None:
                skipped += 1
                reason = skipped_reason or "invalid_message"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            package = self.workflow.answer(message)
            self.profile_service.update_from_answer(package)
            try:
                response = reply_answer_package(self.feishu_client, message.message_id, package)
            except Exception as exc:
                failed += 1
                self.store.record_reply_attempt(
                    question_id=package.question_id,
                    message_id=message.message_id,
                    channel="feishu-polling",
                    status="failure",
                    error=str(exc),
                )
                continue
            self.store.record_reply_attempt(
                question_id=package.question_id,
                message_id=message.message_id,
                channel="feishu-polling",
                status="success",
                response=response,
            )
            processed += 1
        return {
            "chat_id": chat_id,
            "seen": len(items),
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "skip_reasons": skip_reasons,
            "message_diagnostics": message_diagnostics,
        }


def _message_from_item(item: dict[str, object], fallback_chat_id: str) -> IncomingMessage | None:
    message, _, _ = _message_from_item_with_diagnostic(item, fallback_chat_id=fallback_chat_id)
    return message


def _message_from_item_with_diagnostic(
    item: dict[str, object],
    fallback_chat_id: str,
) -> tuple[IncomingMessage | None, str | None, dict[str, object]]:
    sender = item.get("sender") if isinstance(item.get("sender"), dict) else {}
    sender_type = str(sender.get("sender_type") or "")
    message_id = str(item.get("message_id") or "").strip()
    msg_type = str(item.get("msg_type") or "")
    body = item.get("body") if isinstance(item.get("body"), dict) else {}
    text = _text_from_body(body)
    diagnostic: dict[str, object] = {
        "message_id": message_id,
        "msg_type": msg_type,
        "sender_type": sender_type,
        "skipped_reason": None,
        "text_preview": _preview_text(text),
    }
    if item.get("msg_type") != "text":
        diagnostic["skipped_reason"] = "non_text"
        return None, "non_text", diagnostic
    if sender.get("sender_type") == "app":
        diagnostic["skipped_reason"] = "app_message"
        return None, "app_message", diagnostic
    if not message_id:
        diagnostic["skipped_reason"] = "missing_message_id"
        return None, "missing_message_id", diagnostic
    if not text:
        diagnostic["skipped_reason"] = "empty_text"
        return None, "empty_text", diagnostic
    return (
        IncomingMessage(
            message_id=message_id,
            event_id=None,
            user_id=str(sender.get("id") or sender.get("open_id") or "unknown-user"),
            chat_id=str(item.get("chat_id") or fallback_chat_id),
            text=text,
            source="feishu-polling",
            raw_event=item,
        ),
        None,
        diagnostic,
    )


def _text_from_body(body: dict[str, object]) -> str:
    content = body.get("content")
    if not isinstance(content, str):
        return ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content.strip()
    text = parsed.get("text") if isinstance(parsed, dict) else None
    return text.strip() if isinstance(text, str) else ""


def _preview_text(text: str, max_chars: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1] + "..."
