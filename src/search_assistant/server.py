from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from search_assistant.config import Settings
from search_assistant.contracts import AnswerPackage
from search_assistant.feishu.client import FeishuClient, FeishuHttpClient, FakeFeishuClient
from search_assistant.feishu.events import handle_challenge, parse_feishu_event, validate_verification_token
from search_assistant.feishu.formatting import package_to_plain_text, reply_answer_package
from search_assistant.memory.store import MemoryStore
from search_assistant.profile.service import ProfileService
from search_assistant.search.provider import SearchClient, search_client_from_settings
from search_assistant.runtime import AgentRuntime
from search_assistant.runtime.pydantic_ai import runtime_from_settings
from search_assistant.workflow.service import SearchAssistantWorkflow


logger = logging.getLogger(__name__)


def create_app(
    data_dir: str | Path | None = None,
    store: MemoryStore | None = None,
    runtime: AgentRuntime | None = None,
    search_client: SearchClient | None = None,
    feishu_client: FeishuClient | None = None,
) -> FastAPI:
    app = FastAPI(title="Search Assistant")
    settings = Settings.from_env()
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
    if feishu_client is not None:
        client = feishu_client
    elif settings.feishu_enabled:
        client = FeishuHttpClient(settings.feishu_app_id, settings.feishu_app_secret)
    else:
        client = FakeFeishuClient()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/feishu/events")
    async def feishu_events(request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            validate_verification_token(payload, settings.feishu_verification_token)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        challenge = handle_challenge(payload)
        if challenge is not None:
            return challenge
        try:
            message = parse_feishu_event(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if memory_store.has_successful_reply(message.message_id):
            logger.info(
                "skipping duplicate Feishu webhook event event_id=%s message_id=%s",
                message.event_id,
                message.message_id,
            )
            return {"ok": True, "duplicate": True}

        package = workflow.answer(message)
        ProfileService(memory_store).update_from_answer(package)
        try:
            reply = reply_answer_package(client, message.message_id, package)
        except Exception as exc:
            memory_store.record_reply_attempt(
                question_id=package.question_id,
                message_id=message.message_id,
                channel=message.source,
                status="failure",
                error=str(exc),
            )
            logger.exception("failed to reply to Feishu webhook message message_id=%s", message.message_id)
            raise HTTPException(status_code=502, detail="Feishu reply failed") from exc
        memory_store.record_reply_attempt(
            question_id=package.question_id,
            message_id=message.message_id,
            channel=message.source,
            status="success",
            response=reply,
        )
        return {"ok": True, "reply": reply}

    return app


def _format_feishu_reply(package: dict[str, Any]) -> str:
    return package_to_plain_text(
        AnswerPackage(
            question_id=str(package["question_id"]),
            answer_text=str(package["answer_text"]),
            classification=package["classification"],
            confidence=package["confidence"],
        )
    )
