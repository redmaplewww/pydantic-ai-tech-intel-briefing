# Architecture

## Runtime Topology

The project has two adjacent workflows:

- **Question answering:** retrieve public evidence, generate an answer,
  calibrate and review it, then persist the visible search record and evidence.
- **Daily briefing:** expand a topic into technical queries, retrieve and rank
  public sources, synthesize a Chinese technical report, and store the report
  with all retained original URLs.

```mermaid
flowchart TD
    Input[CLI, Feishu, or scheduler] --> Router[Workflow or briefing router]
    Router --> Memory[(SQLite)]
    Router --> Planner[Pydantic AI model planning]
    Planner --> Search[Hybrid search client]
    Search --> MCP[Read-only MCP tools]
    Search --> Engines[Public web engines]
    Search --> RSS[Optional local RSSHub]
    MCP --> Filter[URL, relevance, and source-quality filters]
    Engines --> Filter
    RSS --> Filter
    Filter --> Rank[Deduplicate and rank]
    Rank --> Model[Pydantic AI synthesis]
    Model --> Report[Chinese Markdown report]
    Report --> Memory
    Report --> Delivery[Optional Feishu delivery]
```

## Daily Briefing Lifecycle

1. A topic is created or resumed from a subscription.
2. User feedback and case URLs are loaded as planning signals. A case URL is
   preserved even when public indexing cannot enrich it.
3. The configured Pydantic AI model provider proposes focused technical
   queries. Deterministic coverage queries add architecture, implementation,
   evaluation, and source-family routes.
4. Retrieval runs concurrently with a bounded budget. Slow sources do not
   erase completed results.
5. The service rejects malformed search-page text, generic reference pages,
   platform wrapper URLs, login pages, domain leakage from `site:` queries, and
   topic-specific false positives.
6. Results are normalized, compacted, deduplicated, ranked, and persisted.
7. The configured Pydantic AI model provider receives ranked evidence and
   produces structured JSON. The report's detailed section remains free-form;
   evidence anchors preserve source traceability without forcing a repeated
   per-theme schema.
8. On a synthesis timeout, the runtime retries once with a smaller ranked
   evidence set. If both attempts fail, the deterministic fallback names the
   evidence limitation rather than asserting unobserved deployment facts.

## Persistence

`MemoryStore` uses SQLite. Key tables include topic subscriptions, topic
feedback, collected sources, daily briefings, interactions, answers, claim
evidence, profiles, reports, runtime sessions, skill drafts, immutable
trajectories, trajectory evaluations, domain knowledge candidates, reviewed
domain knowledge graphs, and candidate-to-graph links.

Domain knowledge graphs are stored as graph headers, entity nodes, and typed
relation edges. They intentionally keep natural-language summaries next to
entity-relation triples: the triples support GraphRAG-style navigation and
entity disambiguation, while the summaries preserve the semantic nuance that
would be lost by reducing every domain claim to a bare triple.

Self-evolution output remains a reviewable candidate first. Search-derived
domain knowledge candidates are linked to the best matching reviewed graph
entity through immutable bridge records, but they are not promoted into graph
facts until a human or later review workflow validates the evidence.

Source observability adds four local-only tables:

- `source_candidates` records accepted, duplicate, feedback-seeded, and
  rejected briefing candidates with short reason codes.
- `provider_trace_events` records whether a provider succeeded, returned no
  results, timed out, errored, or was skipped.
- `topic_feedback_signals` classifies explicit user feedback as style,
  evidence, provider, fact-correction, case, or general feedback.
- `layered_memory_items` separates event, preference, domain-knowledge, and
  run-experience memory so self-improvement does not mix user style feedback
  with factual knowledge.

The database is scoped by user and chat where applicable. The default location
is `.local-data/assistant.sqlite3`; use `--data-dir` for isolated test or
deployment environments.

## Source Contracts and Replay

Public source families are described in a local source contract registry.  A
contract records aliases, URL hosts, provider families, allowed read-only
actions, unsupported login/write actions, and default search-budget weight.  The
registry is intentionally descriptive: it does not bypass platform controls and
does not add logged-in automation.

Each `brief-run` writes the Markdown report and a sidecar JSON artifact with
ranked sources, source candidates, and provider events.  The sidecar is the
minimum replay unit for comparing PRs or diagnosing why a report had too few
sources.

## Model Boundary

`PydanticAIModelRuntime` is the only primary agent runtime. `GLMPydanticAIRuntime`
and `DeepSeekChatRuntime` are provider-specific configurations of that runtime
over OpenAI-compatible endpoints. The in-repo harness owns workflow control,
retrieval, verification, calibration, final review, memory, evaluation, and
offline evolution; the model runtime only performs bounded text tasks and the
workflow-specific planning/synthesis/review prompts. Microsoft Agent Framework
is not in the default or DeepSeek execution path. Older superpowers MVP
documents that mention it are retained as historical records, not current
architecture.

Model secrets are passed only through runtime environment configuration, never
through reports, SQLite source text, or checked-in files.

## Safety Boundaries

- Retrieval sources must return an original `http` or `https` URL.
- MCP search bindings must be read-only.
- Login-gated platform content and private social data are not collected.
- Source titles alone do not justify an architecture claim; the synthesis prompt
  asks the model to state missing evidence.
- A separate scope review reduces unsupported industry-wide or inevitable
  claims while preserving source-backed technical detail.

## Continuous Evolution: Phase One

Phase one separates stable online execution from offline, reviewable evolution:

1. Every persisted answer appends one trajectory snapshot containing the
   question, search record, draft, calibration, review, final answer, active
   skills, and runtime metadata. SQLite triggers reject trajectory updates and
   deletes.
2. The evaluation suite verifies the result, process, and answer quality as
   separate structures. A diagnosis service routes failures to a candidate
   update carrier without changing prompts, skills, or program code.
3. Daily briefing themes become deduplicated domain knowledge candidates backed
   by original source URLs. Candidates remain reviewable until an explicit
   evidence gate validates them; deprecated records and status events are kept
   for rollback and audit.
4. The offline evolution loop (`evolution-offline-run`) aggregates immutable
   trajectories and runs a three-layer verifier on each: a code-based result
   layer (answer present, search executed, sources returned), a code-based
   process layer (required search, calibration/review discipline, citation
   provenance, uncertainty disclosure), and a rubric quality layer (deterministic
   rule judge by default; an LLM rubric judge can be enabled and calibrated
   against expert labels before use). Verdicts carry evidence locations and may
   abstain with `uncertain` instead of guessing. Failures are routed to
   reviewable experience items; release stays human-gated.
5. Knowledge candidates accumulate evidence across qualified trajectories. The
   validation gate now allows one high-quality briefing trajectory to produce a
   reviewable candidate only when it also has independent source-domain
   coverage, sufficient source authority, technical-not-marketing semantic
   quality, and non-low confidence. Automatic distillation moves candidates
   that pass every gate to `pending_user_confirm`, not straight to `validated`.
   Only explicit user or human confirmation promotes a candidate to `validated`,
   and only validated knowledge enters later planning or context. The offline
   loop re-runs these gates (distillation) and monitors the eval report and
   regressed candidates without auto-promoting or auto-deprecating anything.
6. Learning reports expose trajectory and candidate status. They are reporting
   artifacts, not trusted evidence and not an automatic release mechanism.
