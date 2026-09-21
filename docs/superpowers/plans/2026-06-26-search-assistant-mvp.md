# Search Assistant MVP Implementation Plan

> Historical / Superseded: this June 2026 MVP plan is preserved for
> traceability, including its Microsoft Agent Framework assumptions. It is not
> the current runtime architecture. See
> `docs/adr/0001-pydantic-ai-primary-runtime.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 Feishu-connected search assistant MVP described in `docs/superpowers/specs/2026-06-26-search-assistant-design.md`.

**Architecture:** A modular Python service exposes both FastAPI HTTP endpoints and CLI commands. Microsoft Agent Framework is used behind an `AgentRuntime` adapter so the answer workflow remains testable with deterministic fakes while production can use `agent-framework==1.9.0`.

**Tech Stack:** Python 3.13 local runtime, `agent-framework==1.9.0`, `fastapi==0.138.1`, `pydantic==2.13.4`, stdlib `sqlite3`, stdlib `argparse`, `pytest`, and FastAPI `TestClient`.

## Global Constraints

- The system must use Microsoft Agent Framework as the agent/workflow foundation.
- The first implementation phase builds a reliable local MVP that can be tested end to end, while keeping the larger goal intact for later production hardening.
- The assistant does not auto-enable generated skills in Phase 1. It creates reviewable skill drafts only.
- Every answer produced by the workflow must contain `question_id`, `answer_text`, `classification`, `confidence`, `verified_claims`, `unverified_claims`, `calibration`, and `memory_updates` as defined in the spec.
- For `hard`, `research`, and `high_stakes` requests, the system must run a second pass before responding.
- If a claim cannot be verified, the answer must either remove the claim or label it as unverified.
- Phase 1 uses SQLite as the canonical store.
- Production credentials must come from environment variables or an ignored local env file. Secrets must not be committed.
- Implementation must follow TDD.
- Network and model calls must be behind interfaces so tests can use deterministic fakes.

---

## File Structure

- Create `pyproject.toml`: package metadata, runtime dependencies, pytest config.
- Create `.gitignore`: ignore virtualenv, caches, local env files, SQLite data, generated reports, and draft skills review outputs only where appropriate.
- Create `.env.example`: documented configuration keys with empty sample values.
- Create `src/search_assistant/__init__.py`: package marker and version.
- Create `src/search_assistant/config.py`: environment-backed settings.
- Create `src/search_assistant/contracts.py`: Pydantic contracts shared by all modules.
- Create `src/search_assistant/memory/store.py`: SQLite schema and repository methods.
- Create `src/search_assistant/verification/policy.py`: claim extraction and verification policy.
- Create `src/search_assistant/workflow/runtime.py`: `AgentRuntime` protocol, fake runtime, and Microsoft Agent Framework adapter.
- Create `src/search_assistant/workflow/service.py`: deterministic answer workflow orchestration.
- Create `src/search_assistant/feishu/events.py`: Feishu payload parsing and challenge handling.
- Create `src/search_assistant/feishu/client.py`: Feishu client protocol, fake client, and HTTP client shell.
- Create `src/search_assistant/server.py`: FastAPI app factory.
- Create `src/search_assistant/profile/service.py`: profile update service.
- Create `src/search_assistant/reports/service.py`: learning report generation.
- Create `src/search_assistant/skills/service.py`: skill draft generation.
- Create `src/search_assistant/cli.py`: local ask, report, skill draft, and fixture commands.
- Create `README.md`: user setup and operation guide.
- Create `docs/code_manual.md`: code manual and extension guide.
- Create `tests/`: focused tests per component plus one end-to-end flow.

---

### Task 1: Project Scaffold And Shared Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/search_assistant/__init__.py`
- Create: `src/search_assistant/config.py`
- Create: `src/search_assistant/contracts.py`
- Test: `tests/test_contracts.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.from_env(env: Mapping[str, str] | None = None) -> Settings`
- Produces: `IncomingMessage`, `VerifiedClaim`, `CalibrationResult`, `MemoryUpdate`, `AnswerPackage`
- Consumes: none

- [ ] **Step 1: Write failing contract tests**

```python
from search_assistant.contracts import AnswerPackage, IncomingMessage


def test_answer_package_requires_verification_fields():
    package = AnswerPackage(
        question_id="q-1",
        answer_text="Use verified sources.",
        classification="research",
        confidence="medium",
        verified_claims=[],
        unverified_claims=["current pricing"],
        calibration={"ran": True, "critique": "Needs source", "revision": "Marked unverified"},
        memory_updates=[],
    )

    assert package.classification == "research"
    assert package.unverified_claims == ["current pricing"]


def test_incoming_message_keeps_feishu_metadata():
    message = IncomingMessage(
        message_id="om_1",
        event_id="evt_1",
        user_id="ou_1",
        chat_id="oc_1",
        text="What changed in Agent Framework?",
        source="feishu",
    )

    assert message.dedupe_key == "evt_1"
```

- [ ] **Step 2: Run contract tests and verify red**

Run: `python -m pytest tests/test_contracts.py -q`

Expected: import failure for `search_assistant.contracts`.

- [ ] **Step 3: Implement scaffold and contracts**

Create `pyproject.toml` with package discovery under `src`, dependencies pinned to the stack above, and pytest `pythonpath = ["src"]`. Implement Pydantic models using literal fields:

```python
Classification = Literal["simple", "research", "hard", "high_stakes"]
Confidence = Literal["low", "medium", "high"]
```

`IncomingMessage.dedupe_key` returns `event_id` when present and `message_id` otherwise.

- [ ] **Step 4: Add failing settings tests**

```python
from search_assistant.config import Settings


def test_settings_loads_defaults_for_local_mode(tmp_path):
    settings = Settings.from_env({"SEARCH_ASSISTANT_DATA_DIR": str(tmp_path)})

    assert settings.data_dir == tmp_path
    assert settings.database_path == tmp_path / "assistant.sqlite3"
    assert settings.feishu_enabled is False
```

- [ ] **Step 5: Implement settings and verify green**

Run: `python -m pytest tests/test_contracts.py tests/test_config.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/search_assistant tests/test_contracts.py tests/test_config.py
git commit -m "feat: add project scaffold and contracts"
```

---

### Task 2: SQLite Memory Store

**Files:**
- Create: `src/search_assistant/memory/__init__.py`
- Create: `src/search_assistant/memory/store.py`
- Test: `tests/test_memory_store.py`

**Interfaces:**
- Consumes: `IncomingMessage`, `AnswerPackage`, `VerifiedClaim`, `MemoryUpdate`
- Produces: `MemoryStore.initialize() -> None`
- Produces: `MemoryStore.record_interaction(message: IncomingMessage) -> str`
- Produces: `MemoryStore.record_answer(package: AnswerPackage) -> None`
- Produces: `MemoryStore.has_interaction(dedupe_key: str) -> bool`
- Produces: `MemoryStore.latest_answer_for_dedupe_key(dedupe_key: str) -> AnswerPackage | None`
- Produces: `MemoryStore.add_memory_item(kind: str, content: str, source_id: str, supersedes_id: str | None = None) -> str`
- Produces: `MemoryStore.add_experience_item(title: str, body: str, source_ids: list[str]) -> str`

- [ ] **Step 1: Write failing persistence tests**

```python
from search_assistant.contracts import AnswerPackage, IncomingMessage
from search_assistant.memory.store import MemoryStore


def test_records_interaction_answer_and_evidence(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    message = IncomingMessage(
        message_id="m-1",
        event_id="e-1",
        user_id="u-1",
        chat_id="c-1",
        text="What is the latest FastAPI version?",
        source="cli",
    )

    question_id = store.record_interaction(message)
    package = AnswerPackage(
        question_id=question_id,
        answer_text="FastAPI is current only after checking PyPI.",
        classification="research",
        confidence="medium",
        verified_claims=[],
        unverified_claims=["FastAPI latest version"],
        calibration={"ran": True, "critique": "Needs package source", "revision": "Marked unverified"},
        memory_updates=[],
    )
    store.record_answer(package)

    assert store.has_interaction("e-1")
    assert store.latest_answer_for_dedupe_key("e-1").question_id == question_id
```

- [ ] **Step 2: Run memory tests and verify red**

Run: `python -m pytest tests/test_memory_store.py -q`

Expected: import failure for `search_assistant.memory.store`.

- [ ] **Step 3: Implement schema and repository**

Use stdlib `sqlite3`. Create all tables from the spec with `created_at` ISO timestamps. Store nested answer fields as JSON text. Use deterministic ids generated with `uuid.uuid4().hex`.

- [ ] **Step 4: Add append-first memory test**

```python
def test_memory_supersession_keeps_old_record(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()

    old_id = store.add_memory_item("preference", "likes concise answers", "manual")
    new_id = store.add_memory_item("preference", "likes concise answers with sources", "manual", supersedes_id=old_id)

    rows = store.list_memory_items()
    assert [row["id"] for row in rows] == [old_id, new_id]
    assert rows[1]["supersedes_id"] == old_id
```

- [ ] **Step 5: Implement list helpers and verify green**

Run: `python -m pytest tests/test_memory_store.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/search_assistant/memory tests/test_memory_store.py
git commit -m "feat: add sqlite memory store"
```

---

### Task 3: Verification Policy And Calibration Rules

**Files:**
- Create: `src/search_assistant/verification/__init__.py`
- Create: `src/search_assistant/verification/policy.py`
- Test: `tests/test_verification_policy.py`

**Interfaces:**
- Consumes: `Classification`
- Produces: `requires_verification(text: str, classification: Classification) -> bool`
- Produces: `extract_key_claims(text: str) -> list[str]`
- Produces: `requires_calibration(classification: Classification, has_unverified_claims: bool) -> bool`

- [ ] **Step 1: Write failing policy tests**

```python
from search_assistant.verification.policy import (
    extract_key_claims,
    requires_calibration,
    requires_verification,
)


def test_current_version_claim_requires_verification():
    assert requires_verification("FastAPI 0.138.1 is the latest version.", "research")


def test_hard_and_high_stakes_require_calibration():
    assert requires_calibration("hard", has_unverified_claims=False)
    assert requires_calibration("high_stakes", has_unverified_claims=False)
    assert requires_calibration("simple", has_unverified_claims=True)


def test_extract_key_claims_keeps_numbered_claims():
    claims = extract_key_claims("FastAPI 0.138.1 was released in 2026. Use official docs.")

    assert "FastAPI 0.138.1 was released in 2026" in claims
```

- [ ] **Step 2: Run policy tests and verify red**

Run: `python -m pytest tests/test_verification_policy.py -q`

Expected: import failure for `search_assistant.verification.policy`.

- [ ] **Step 3: Implement deterministic policy**

Use keyword and regex detection for freshness words, versions, dates, money, percentages, high-stakes domains, and named-entity-like phrases. The first implementation is conservative and deterministic.

- [ ] **Step 4: Verify green**

Run: `python -m pytest tests/test_verification_policy.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/search_assistant/verification tests/test_verification_policy.py
git commit -m "feat: add verification and calibration policy"
```

---

### Task 4: Agent Framework Runtime Adapter And Answer Workflow

**Files:**
- Create: `src/search_assistant/workflow/__init__.py`
- Create: `src/search_assistant/workflow/runtime.py`
- Create: `src/search_assistant/workflow/service.py`
- Test: `tests/test_workflow_service.py`

**Interfaces:**
- Consumes: `MemoryStore`, verification policy, shared contracts
- Produces: `AgentRuntime.generate_answer(question: str, context: dict[str, object]) -> str`
- Produces: `AgentRuntime.calibrate(draft: str, context: dict[str, object]) -> CalibrationResult`
- Produces: `SearchAssistantWorkflow.answer(message: IncomingMessage) -> AnswerPackage`

- [ ] **Step 1: Write failing workflow tests**

```python
from search_assistant.contracts import IncomingMessage
from search_assistant.memory.store import MemoryStore
from search_assistant.workflow.runtime import FakeAgentRuntime
from search_assistant.workflow.service import SearchAssistantWorkflow


def test_hard_question_runs_calibration_and_persists_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="The answer depends on verified API behavior in 2026.")
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime)
    message = IncomingMessage(
        message_id="m-1",
        event_id="e-1",
        user_id="u-1",
        chat_id="c-1",
        text="Compare the newest Agent Framework workflow API with Semantic Kernel and give migration risks.",
        source="cli",
    )

    package = workflow.answer(message)

    assert package.classification == "hard"
    assert package.calibration["ran"] is True
    assert runtime.calibration_calls == 1
    assert store.latest_answer_for_dedupe_key("e-1").question_id == package.question_id
```

- [ ] **Step 2: Run workflow tests and verify red**

Run: `python -m pytest tests/test_workflow_service.py -q`

Expected: import failure for `search_assistant.workflow.service`.

- [ ] **Step 3: Implement runtime protocol and fake runtime**

`FakeAgentRuntime` returns deterministic answer text and calibration objects. `MicrosoftAgentRuntime` imports `agent_framework` lazily and exposes the same protocol. It must raise a clear setup error if production settings are incomplete.

- [ ] **Step 4: Implement workflow orchestration**

Classify with deterministic rules. Record the interaction before generation. Retrieve simple memory context. Generate draft. Extract claims. Mark unverified claims when no verifier source is configured. Calibrate when required. Persist and return the answer package.

- [ ] **Step 5: Add duplicate event test**

```python
def test_duplicate_event_returns_stored_answer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = FakeAgentRuntime(answer_text="First answer.")
    workflow = SearchAssistantWorkflow(store=store, runtime=runtime)
    message = IncomingMessage(message_id="m-1", event_id="e-1", user_id="u-1", chat_id="c-1", text="simple note", source="cli")

    first = workflow.answer(message)
    second = workflow.answer(message)

    assert second.question_id == first.question_id
    assert runtime.answer_calls == 1
```

- [ ] **Step 6: Verify green**

Run: `python -m pytest tests/test_workflow_service.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/search_assistant/workflow tests/test_workflow_service.py
git commit -m "feat: add answer workflow"
```

---

### Task 5: Feishu Gateway And HTTP Server

**Files:**
- Create: `src/search_assistant/feishu/__init__.py`
- Create: `src/search_assistant/feishu/events.py`
- Create: `src/search_assistant/feishu/client.py`
- Create: `src/search_assistant/server.py`
- Test: `tests/test_feishu_events.py`
- Test: `tests/test_server.py`
- Create: `tests/fixtures/feishu_message_event.json`

**Interfaces:**
- Consumes: `IncomingMessage`, `SearchAssistantWorkflow`
- Produces: `parse_feishu_event(payload: dict[str, object]) -> IncomingMessage`
- Produces: `handle_challenge(payload: dict[str, object]) -> dict[str, str] | None`
- Produces: `FeishuClient.reply_text(message_id: str, text: str) -> dict[str, object]`
- Produces: `create_app(store: MemoryStore | None = None, runtime: AgentRuntime | None = None, feishu_client: FeishuClient | None = None) -> FastAPI`

- [ ] **Step 1: Write failing Feishu parsing tests**

```python
from search_assistant.feishu.events import handle_challenge, parse_feishu_event


def test_feishu_challenge_response():
    assert handle_challenge({"type": "url_verification", "challenge": "abc"}) == {"challenge": "abc"}


def test_parse_receive_message_event():
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt_1", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {"message_id": "om_1", "chat_id": "oc_1", "content": "{\"text\":\"hello\"}"},
            "sender": {"sender_id": {"open_id": "ou_1"}},
        },
    }

    message = parse_feishu_event(payload)

    assert message.text == "hello"
    assert message.event_id == "evt_1"
```

- [ ] **Step 2: Run Feishu tests and verify red**

Run: `python -m pytest tests/test_feishu_events.py -q`

Expected: import failure for `search_assistant.feishu.events`.

- [ ] **Step 3: Implement Feishu event parser and fake client**

Parse Feishu v2 event envelopes. Return clear `ValueError` for unsupported event types or missing text. `FakeFeishuClient` records replies in memory for tests.

- [ ] **Step 4: Write failing server test**

```python
from fastapi.testclient import TestClient
from search_assistant.feishu.client import FakeFeishuClient
from search_assistant.server import create_app


def test_feishu_webhook_replies_with_answer(tmp_path):
    fake_client = FakeFeishuClient()
    app = create_app(data_dir=tmp_path, feishu_client=fake_client)
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
```

- [ ] **Step 5: Implement FastAPI app factory**

Create `/healthz` and `/feishu/events`. Challenge payloads return the challenge. Message payloads call the workflow and then `reply_text`.

- [ ] **Step 6: Verify green**

Run: `python -m pytest tests/test_feishu_events.py tests/test_server.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/search_assistant/feishu src/search_assistant/server.py tests/test_feishu_events.py tests/test_server.py tests/fixtures
git commit -m "feat: add feishu gateway"
```

---

### Task 6: Profile, Learning Reports, And Skill Drafts

**Files:**
- Create: `src/search_assistant/profile/__init__.py`
- Create: `src/search_assistant/profile/service.py`
- Create: `src/search_assistant/reports/__init__.py`
- Create: `src/search_assistant/reports/service.py`
- Create: `src/search_assistant/skills/__init__.py`
- Create: `src/search_assistant/skills/service.py`
- Test: `tests/test_profile_reports_skills.py`

**Interfaces:**
- Consumes: `MemoryStore`, `AnswerPackage`
- Produces: `ProfileService.update_from_answer(package: AnswerPackage) -> str`
- Produces: `ReportService.generate_markdown(start: str | None = None, end: str | None = None) -> str`
- Produces: `SkillDraftService.create_from_experience(title: str, source_ids: list[str]) -> str`

- [ ] **Step 1: Write failing profile/report/skill test**

```python
from search_assistant.contracts import AnswerPackage
from search_assistant.memory.store import MemoryStore
from search_assistant.profile.service import ProfileService
from search_assistant.reports.service import ReportService
from search_assistant.skills.service import SkillDraftService


def test_profile_report_and_skill_draft_are_persisted(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    package = AnswerPackage(
        question_id="q-1",
        answer_text="You ask about Feishu and agent workflows.",
        classification="research",
        confidence="medium",
        verified_claims=[],
        unverified_claims=["Feishu API detail"],
        calibration={"ran": True, "critique": "Needs source", "revision": "Marked unverified"},
        memory_updates=[{"kind": "topic", "content": "Feishu agent workflow"}],
    )
    store.record_answer(package)

    profile_id = ProfileService(store).update_from_answer(package)
    report = ReportService(store, output_dir=tmp_path / "reports").generate_markdown()
    draft_path = SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts").create_from_experience(
        title="Feishu Workflow Answers",
        source_ids=[profile_id],
    )

    assert "Feishu" in report
    assert draft_path.endswith("SKILL.md")
```

- [ ] **Step 2: Run tests and verify red**

Run: `python -m pytest tests/test_profile_reports_skills.py -q`

Expected: import failure for profile service.

- [ ] **Step 3: Implement services**

Profile service writes compact JSON snapshots. Report service aggregates stored answers and exports markdown under the configured reports directory. Skill draft service writes `skills/drafts/<slug>/SKILL.md` and records the draft in SQLite with review status `draft`.

- [ ] **Step 4: Verify green**

Run: `python -m pytest tests/test_profile_reports_skills.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/search_assistant/profile src/search_assistant/reports src/search_assistant/skills tests/test_profile_reports_skills.py
git commit -m "feat: add profile reports and skill drafts"
```

---

### Task 7: CLI, README, And Code Manual

**Files:**
- Create: `src/search_assistant/cli.py`
- Create: `README.md`
- Create: `docs/code_manual.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Settings`, `MemoryStore`, `SearchAssistantWorkflow`, report and skill services
- Produces: `python -m search_assistant.cli ask "question"`
- Produces: `python -m search_assistant.cli report`
- Produces: `python -m search_assistant.cli skill-draft "title"`
- Produces: `python -m search_assistant.cli feishu-fixture tests/fixtures/feishu_message_event.json`

- [ ] **Step 1: Write failing CLI tests**

```python
import subprocess
import sys


def test_cli_ask_returns_answer_package(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "search_assistant.cli",
            "ask",
            "What should be verified in current data?",
            "--data-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "answer_text" in result.stdout
    assert "classification" in result.stdout
```

- [ ] **Step 2: Run CLI test and verify red**

Run: `python -m pytest tests/test_cli.py -q`

Expected: module execution failure for `search_assistant.cli`.

- [ ] **Step 3: Implement CLI**

Use stdlib `argparse`. Print JSON for `ask`. Print markdown path for `report` and `skill-draft`. For `feishu-fixture`, load JSON and run the same app workflow with fake Feishu client.

- [ ] **Step 4: Write README and code manual**

README must include purpose, architecture, setup with `python -m venv` and `pip install -e ".[dev]"`, Feishu configuration, local CLI examples, test command, safety model, and Phase 1 limitations. Code manual must include module map, data contracts, workflow steps, extension points, and development rules.

- [ ] **Step 5: Verify green**

Run: `python -m pytest tests/test_cli.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/search_assistant/cli.py README.md docs/code_manual.md tests/test_cli.py
git commit -m "feat: add cli and project documentation"
```

---

### Task 8: End-To-End Acceptance Verification

**Files:**
- Create: `tests/test_acceptance_mvp.py`
- Modify: `README.md`
- Modify: `docs/code_manual.md`

**Interfaces:**
- Consumes: all previous public interfaces
- Produces: verified Phase 1 MVP behavior

- [ ] **Step 1: Write failing acceptance test**

```python
from fastapi.testclient import TestClient
from search_assistant.feishu.client import FakeFeishuClient
from search_assistant.server import create_app


def test_phase_1_acceptance_flow(tmp_path):
    feishu = FakeFeishuClient()
    app = create_app(data_dir=tmp_path, feishu_client=feishu)
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
    assert "confidence" in feishu.replies[0]["text"]
```

- [ ] **Step 2: Run full test suite and verify red if any acceptance behavior is missing**

Run: `python -m pytest -q`

Expected before fixes: any missing acceptance behavior fails with a specific assertion.

- [ ] **Step 3: Patch missing integration behavior only**

Use the already-created modules. Do not add new subsystems. Update README and code manual if commands or behavior changed.

- [ ] **Step 4: Verify full green suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Run local smoke commands**

Run:

```bash
python -m search_assistant.cli ask "What should be verified before answering current API questions?" --data-dir ./.local-data
python -m search_assistant.cli report --data-dir ./.local-data
python -m search_assistant.cli skill-draft "Reliable API Answers" --data-dir ./.local-data
```

Expected: first command prints JSON answer package, second prints a report markdown path, third prints a draft `SKILL.md` path.

- [ ] **Step 6: Commit**

```bash
git add tests/test_acceptance_mvp.py README.md docs/code_manual.md src/search_assistant
git commit -m "test: verify phase 1 mvp acceptance"
```

---

## Self-Review Notes

- Spec coverage: Tasks cover Feishu ingress, Agent Framework adapter, answer workflow, verification, calibration, SQLite memory, profile, reports, skill drafts, README, code manual, CLI, and automated acceptance tests.
- Placeholder scan: This plan intentionally avoids unfinished placeholder markers.
- Type consistency: Shared contract names are defined in Task 1 and reused consistently by later tasks.
- Scope control: Phase 1 creates reviewable skill drafts and local fixture testing. Live Feishu deployment, scheduled reports, and automatic skill promotion remain later-phase work, matching the approved design.

