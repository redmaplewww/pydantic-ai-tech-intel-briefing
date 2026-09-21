# Code Manual

This document is the current module guide. It is intentionally shorter than a
line-by-line API reference; public behavior and deployment instructions belong
in the other documents under `docs/`.

## Packages

| Package | Responsibility |
| --- | --- |
| `contracts` | Pydantic models for messages, source evidence, daily briefings, and synthesis output. |
| `config` | Environment-backed settings. Secrets are read at runtime and never written to reports. |
| `search` | Browser, MCP, RSSHub, public Bilibili, Brave, SearXNG, and fallback search adapters. |
| `mcp` | Read-only local MCP servers for public sources and domestic RSS feeds. |
| `runtime` | Runtime protocols, deterministic fake runtime, and Pydantic AI provider adapters for GLM and DeepSeek. |
| `briefing` | Topic planning, source filtering, ranking, synthesis, Markdown rendering, and feedback handling. |
| `knowledge_graph` | Reviewed domain graph seeds and query/export helpers for GraphRAG-style entity-relation knowledge. |
| `workflow` | Ordinary question-answering, verification, calibration, final review, and active-skill context. |
| `memory` | SQLite schema and scoped persistence. |
| `verification` | Claim extraction, deterministic source matching, and evidence backfill. |
| `evaluation` | Repeatable workflow evaluation and persisted quality records. |
| `feishu` | Callback parsing, long connection, polling, reply formatting, and safe diagnostics. |
| `reports`, `profile`, `skills` | Learning reports, compact user profiles, and human-reviewed local skill drafts. |

## Briefing Flow

1. `DailyBriefingService.build_search_plan()` combines user feedback, the
   configured Pydantic AI model provider's planning output, deterministic
   technical coverage queries, and platform routes.
2. `_collect_sources()` runs retrieval concurrently under a budget, rejects
   search-page dumps, generic references, login pages, invalid platform URLs,
   and obvious false positives, then stores compact source evidence.
3. Sources are deduplicated and ranked. The full retained set remains in SQLite
   and the report; only the highest-ranked evidence is sent to the synthesis
   model.
4. `PydanticAIModelRuntime.synthesize_briefing()` returns validated JSON with a
   concise technical summary, free-form detailed analysis, evidence anchors,
   next research directions, and implementation suggestions. A timeout retries
   with a smaller evidence set.
5. `render_markdown()` writes the Chinese report and preserves all retained
   original URLs.

## Data Boundaries

- `.local-data/` contains SQLite databases, generated reports, and runtime
  evidence. It is not source-controlled.
- `.env.local` is ignored and should be the only local file containing model or
  Feishu credentials.
- Checked-in MCP configurations describe commands and bindings only; they must
  remain read-only and URL-bearing.
- Generated skills are drafts until a human explicitly promotes them.

## Extension Points

- Add a public source through `search/provider.py` or a read-only MCP binding.
- Add a reviewed domain graph through `knowledge_graph/seeds.py`, then cover it
  with query/export tests before using it in retrieval or briefing logic.
- Add platform constraints in `configs/domestic-rss.sources.json` and verify
  original-domain enforcement with tests.
- Add a model provider by extending `runtime/pydantic_ai.py` and wiring it in
  `runtime_from_settings()`. Keep business code behind `AgentRuntime` or
  `BriefingRuntime` protocols instead of importing a provider SDK directly.
- Add quality cases in the existing test modules before changing filtering,
  report shape, or verification policy.

For deployment behavior, see [operations.md](operations.md). For source access
limits, see [platform-coverage.md](platform-coverage.md).
