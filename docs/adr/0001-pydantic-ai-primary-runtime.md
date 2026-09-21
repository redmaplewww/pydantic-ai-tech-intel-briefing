# ADR 0001: Pydantic AI Is The Primary Runtime

Date: 2026-08-20
Status: Accepted

## Context

The repository previously carried Microsoft Agent Framework dependencies and
historical MVP documents that described it as the agent/workflow foundation.
The current implementation has moved to Pydantic AI plus an in-repo harness:
CLI, Feishu, server, question-answering, briefing, retrieval, verification,
calibration, review, memory, evaluation, and evolution all depend on local
protocols and services rather than Microsoft Agent Framework workflow APIs.

DeepSeek and GLM are model providers exposed through OpenAI-compatible
endpoints. They should be selectable without changing the agent runtime or
business workflow.

## Decision

Pydantic AI is the only primary agent runtime. The project harness owns
workflow orchestration, retrieval, verification, calibration, review, memory,
evaluation, and offline evolution. MCP, public web engines, RSSHub/API routes,
and optional Agent Reach are retrieval/tool layers. SQLite remains the
persistence and memory layer.

Microsoft Agent Framework is not part of the default execution path and is not
required for DeepSeek. Historical superpowers specs and plans that mention it
remain in the repository as historical records only.

## Consequences

- Runtime protocols live in `src/search_assistant/runtime/base.py`.
- Deterministic tests use `src/search_assistant/runtime/fake.py`.
- GLM and DeepSeek provider adapters live in
  `src/search_assistant/runtime/pydantic_ai.py`.
- `src/search_assistant/workflow/runtime.py` is a compatibility import shim
  only; new code should import from `search_assistant.runtime`.
- Business code must not import Microsoft Agent Framework packages directly.
- Provider selection remains configuration-driven through
  `SEARCH_ASSISTANT_MODEL_PROVIDER`, but the value selects a model provider,
  not an agent framework.
