from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import lark_oapi as lark
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1
from lark_oapi.core.enum import LogLevel

from search_assistant.config import Settings
from search_assistant.contracts import AnswerPackage, IncomingMessage
from search_assistant.feishu.client import FeishuClient, FeishuHttpClient
from search_assistant.feishu.events import parse_feishu_event
from search_assistant.feishu.formatting import package_to_plain_text, reply_answer_package
from search_assistant.memory.store import MemoryStore
from search_assistant.profile.service import ProfileService
from search_assistant.search.provider import SearchClient, search_client_from_settings
from search_assistant.runtime import AgentRuntime
from search_assistant.runtime.pydantic_ai import runtime_from_settings
from search_assistant.workflow.service import SearchAssistantWorkflow


logger = logging.getLogger(__name__)


class _RedactingLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _redact_log_message(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(_redact_log_message(arg) if isinstance(arg, str) else arg for arg in record.args)
            elif isinstance(record.args, dict):
                record.args = {
                    key: _redact_log_message(value) if isinstance(value, str) else value
                    for key, value in record.args.items()
                }
        return True


class FeishuLongConnectionHandler:
    def __init__(
        self,
        workflow: SearchAssistantWorkflow,
        feishu_client: FeishuClient,
        store: MemoryStore,
    ):
        self.workflow = workflow
        self.feishu_client = feishu_client
        self.store = store
        self.profile_service = ProfileService(store)

    def handle_message_event(self, event: P2ImMessageReceiveV1) -> None:
        message = parse_lark_message_event(event)
        logger.info(
            "received Feishu long-connection event event_id=%s message_id=%s chat_id=%s",
            message.event_id,
            message.message_id,
            message.chat_id,
        )
        if self.store.has_successful_reply(message.message_id):
            logger.info(
                "skipping duplicate Feishu long-connection event event_id=%s message_id=%s",
                message.event_id,
                message.message_id,
            )
            return
        try:
            package = self.workflow.answer(message)
        except Exception as exc:
            self.store.record_reply_attempt(
                question_id=message.message_id,
                message_id=message.message_id,
                channel=message.source,
                status="failure",
                error=str(exc),
            )
            logger.exception("failed to generate answer for Feishu message message_id=%s", message.message_id)
            raise
        logger.info(
            "generated answer for Feishu message message_id=%s classification=%s confidence=%s",
            message.message_id,
            package.classification,
            package.confidence,
        )
        self.profile_service.update_from_answer(package)
        try:
            response = reply_answer_package(self.feishu_client, message.message_id, package)
        except Exception as exc:
            self.store.record_reply_attempt(
                question_id=package.question_id,
                message_id=message.message_id,
                channel=message.source,
                status="failure",
                error=str(exc),
            )
            logger.exception("failed to reply to Feishu message message_id=%s", message.message_id)
            raise
        self.store.record_reply_attempt(
            question_id=package.question_id,
            message_id=message.message_id,
            channel=message.source,
            status="success",
            response=response,
        )
        logger.info("replied to Feishu message message_id=%s", message.message_id)


def parse_lark_message_event(event: P2ImMessageReceiveV1) -> IncomingMessage:
    message = parse_feishu_event(lark_message_event_to_payload(event))
    return message.model_copy(update={"source": "feishu-long-connection"})


def build_event_handler(
    handler: FeishuLongConnectionHandler,
    encrypt_key: str | None = None,
    verification_token: str | None = None,
) -> lark.EventDispatcherHandler:
    return (
        lark.EventDispatcherHandler.builder(encrypt_key or "", verification_token or "")
        .register_p2_im_message_receive_v1(handler.handle_message_event)
        .build()
    )


def create_long_connection_handler(
    data_dir: str | Path | None = None,
    store: MemoryStore | None = None,
    runtime: AgentRuntime | None = None,
    search_client: SearchClient | None = None,
    feishu_client: FeishuClient | None = None,
    settings: Settings | None = None,
) -> FeishuLongConnectionHandler:
    settings = settings or Settings.from_env()
    base_dir = Path(data_dir or settings.data_dir)
    memory_store = store or MemoryStore(base_dir / "assistant.sqlite3")
    memory_store.initialize()
    workflow = SearchAssistantWorkflow(
        store=memory_store,
        runtime=runtime or runtime_from_settings(settings),
        search_client=search_client or search_client_from_settings(settings),
        search_budget_seconds=settings.workflow_search_budget_seconds,
        skill_drafts_dir=base_dir / "skills" / "drafts",
        report_output_dir=base_dir / "reports",
        admin_user_ids=set(settings.admin_user_ids),
    )
    client = feishu_client or FeishuHttpClient(settings.feishu_app_id, settings.feishu_app_secret)
    return FeishuLongConnectionHandler(workflow=workflow, feishu_client=client, store=memory_store)


def run_long_connection(
    data_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _install_log_redaction()
    settings = settings or Settings.from_env()
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        raise RuntimeError("FEISHU_APP_ID and FEISHU_APP_SECRET are required for Feishu long connection")

    handler = create_long_connection_handler(data_dir=data_dir, settings=settings)
    event_handler = build_event_handler(
        handler,
        encrypt_key=settings.feishu_encrypt_key,
        verification_token=settings.feishu_verification_token,
    )
    handler.store.record_runtime_session(
        kind="feishu-long-connection",
        process_id=os.getpid(),
        status="started",
        metadata=_runtime_session_metadata(settings),
    )
    logger.info("Starting Feishu long connection client")
    client = lark.ws.Client(
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
        log_level=LogLevel.INFO,
        event_handler=event_handler,
    )
    client.start()


def _install_log_redaction() -> None:
    redacting_filter = _RedactingLogFilter()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if not any(isinstance(existing, _RedactingLogFilter) for existing in handler.filters):
            handler.addFilter(redacting_filter)
    for name in ("Lark", "lark_oapi"):
        target_logger = logging.getLogger(name)
        if not any(isinstance(existing, _RedactingLogFilter) for existing in target_logger.filters):
            target_logger.addFilter(redacting_filter)


def _redact_log_message(message: str) -> str:
    return re.sub(
        r"([?&](?:access_key|ticket|app_secret|tenant_access_token)=)[^&\s]+",
        r"\1<redacted>",
        message,
    )


def _runtime_session_metadata(settings: Settings) -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "model_provider": settings.model_provider,
        "deepseek_model": settings.deepseek_model,
        "deepseek_api_key_configured": bool(settings.deepseek_api_key),
        "glm_model": settings.glm_model,
        "glm_api_key_configured": bool(settings.glm_api_key),
        "allow_fake_runtime": settings.allow_fake_runtime,
        "search_provider": settings.search_provider,
        "mcp_search_config_path": str(settings.mcp_search_config_path) if settings.mcp_search_config_path else None,
        "browser_search_engines": settings.browser_search_engines,
        "browser_search_timeout_seconds": settings.browser_search_timeout_seconds,
        "browser_content_timeout_seconds": settings.browser_content_timeout_seconds,
        "browser_enrich_max_results": settings.browser_enrich_max_results,
        "workflow_search_budget_seconds": settings.workflow_search_budget_seconds,
        "deepseek_timeout_seconds": settings.deepseek_timeout_seconds,
        "feishu_app_id_configured": bool(settings.feishu_app_id),
        "feishu_app_secret_configured": bool(settings.feishu_app_secret),
        "deepseek_api_key_configured": bool(settings.deepseek_api_key),
        "allow_fake_runtime": settings.allow_fake_runtime,
    }


def lark_message_event_to_payload(event: P2ImMessageReceiveV1) -> dict[str, Any]:
    header = _object_to_dict(
        getattr(event, "header", None),
        ("event_id", "token", "create_time", "event_type", "tenant_key", "app_id"),
    )
    header.setdefault("event_type", "im.message.receive_v1")

    event_data = getattr(event, "event", None)
    message = getattr(event_data, "message", None)
    sender = getattr(event_data, "sender", None)
    sender_id = getattr(sender, "sender_id", None)

    return {
        "schema": getattr(event, "schema", None) or "2.0",
        "header": header,
        "event": {
            "sender": {
                **_object_to_dict(sender, ("sender_type", "tenant_key")),
                "sender_id": _object_to_dict(sender_id, ("open_id", "user_id", "union_id")),
            },
            "message": _object_to_dict(
                message,
                (
                    "message_id",
                    "root_id",
                    "parent_id",
                    "create_time",
                    "update_time",
                    "chat_id",
                    "thread_id",
                    "chat_type",
                    "message_type",
                    "content",
                ),
            ),
        },
    }


def format_feishu_reply(package: AnswerPackage) -> str:
    return package_to_plain_text(package)


def _object_to_dict(obj: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if obj is None:
        return {}
    data: dict[str, Any] = {}
    for field in fields:
        value = getattr(obj, field, None)
        if value is not None:
            data[field] = value
    return data
