# Search Assistant Design

> Historical / Superseded: this June 2026 design records the initial MVP
> request and is preserved for traceability. It is superseded by
> `docs/adr/0001-pydantic-ai-primary-runtime.md`, which defines Pydantic AI as
> the primary runtime and treats GLM/DeepSeek as model providers.

Date: 2026-06-26
Status: Approved for design by user; awaiting written-spec review before implementation planning
Workspace: `F:\opencode\search-assistant` logical project; actual local folder is the current workspace.

## Goal

Build a Feishu-connected search assistant that can answer the user's questions with reasonable precision, record every interaction, accumulate reusable knowledge and experience, generate learning reports, maintain a user profile, and improve later answers through that accumulated context.

The system must use Microsoft Agent Framework as the agent/workflow foundation. The first implementation phase builds a reliable local MVP that can be tested end to end, while keeping the larger goal intact for later production hardening.

## Current Project Context

- The workspace is an empty project directory.
- There is no existing source code, package manifest, README, code manual, or git repository.
- The user explicitly requested superpowers workflow, Microsoft Agent Framework, Feishu integration, accurate answers, second-pass calibration for hard questions, key-data verification, self-evolution through knowledge and skill accumulation, periodic learning reports, and long-term user profiling.

## Design References

- Microsoft Agent Framework overview: https://learn.microsoft.com/en-us/agent-framework/overview/
- Microsoft Agent Framework repository: https://github.com/microsoft/agent-framework
- Feishu bot overview: https://open.feishu.cn/document/client-docs/bot-v3/bot-overview
- Feishu receive message event: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/events/receive
- Feishu send/reply message API documentation entry point: https://open.feishu.cn/document/server-docs/im-v1/message

Implementation planning must re-check official documentation before pinning exact package names, event field names, or SDK APIs because both Agent Framework and Feishu APIs can change.

## Scope Decomposition

The full objective contains multiple subsystems:

1. Feishu message ingress and response delivery.
2. Reliable answer workflow using Microsoft Agent Framework.
3. Question, answer, evidence, and experience memory.
4. Difficulty classification, second-pass calibration, and key-data verification.
5. User profile construction and retrieval-aware personalization.
6. Periodic learning reports.
7. Self-evolution through experience extraction and skill generation.
8. Project documentation and operational maintenance.

Phase 1 implements a local, testable MVP across all of these areas. Later phases can replace local adapters with production infrastructure without changing the core boundaries.

## Phase 1 Outcome

Phase 1 is complete when a developer can run the app locally, send a Feishu-style message payload or local CLI question, receive a structured answer, and inspect stored records showing:

- the original question;
- the generated answer;
- the answer difficulty;
- whether second-pass calibration ran;
- which key claims were verified;
- sources or explicit "not verified" markers;
- extracted memory and experience candidates;
- profile updates;
- a generated learning report;
- any skill draft proposed from repeated patterns.

The assistant does not auto-enable generated skills in Phase 1. It creates reviewable skill drafts only.

## Architecture

Use a modular Python service with narrow boundaries:

- `feishu_gateway`: receives Feishu callback requests, verifies challenge/signature fields when configured, normalizes incoming message events, and sends replies through a Feishu client adapter.
- `assistant_workflow`: owns the Microsoft Agent Framework workflow. It classifies the question, retrieves relevant memory, calls answer tools, runs verification/calibration gates, and returns an answer package.
- `verification`: extracts key claims, verifies data-bearing claims through configured tools, records evidence, and marks unverified claims explicitly.
- `memory`: stores questions, answers, evidence, experience items, profile facts, reports, and skill drafts in SQLite plus markdown artifacts where useful.
- `profile`: updates a compact user profile from question history, topics, preferences, recurring goals, and correction feedback.
- `reports`: creates periodic learning reports from stored interactions and profile changes.
- `skill_evolution`: turns repeated successful patterns or user requests into draft Codex-style skill folders for human review.
- `config`: loads environment-driven settings and rejects missing production secrets with clear errors.

The service should expose both:

- HTTP endpoints for Feishu integration.
- CLI commands for local testing, report generation, and skill draft generation.

## Agent Workflow

The answer workflow is a deterministic outer process with model/tool calls inside controlled steps:

1. Normalize the incoming question and attach user/session metadata.
2. Persist the incoming question before answering.
3. Retrieve relevant memory, profile facts, previous similar questions, and approved experience items.
4. Classify the request:
   - `simple`: stable answer, no obvious data freshness requirement.
   - `research`: benefits from retrieval, source checking, or current data.
   - `high_stakes`: legal, medical, financial, safety, or other high-impact content.
   - `hard`: ambiguous, multi-step, math-heavy, technical, or likely to contain fragile factual claims.
5. Generate a draft answer with a structured `AnswerPackage`.
6. Extract key claims from the draft answer.
7. Verify claims that contain current facts, numbers, dates, named entities, high-stakes advice, or external-source assertions.
8. Run second-pass calibration when the classification is `hard`, `research`, or `high_stakes`, or when verification confidence is below threshold.
9. Produce the final answer with confidence, caveats, and concise source notes.
10. Persist answer, evidence, calibration output, memory updates, and experience candidates.
11. Return a Feishu-safe text response.

## Answer Package Contract

Every answer produced by the workflow must contain:

- `question_id`: durable id for the stored interaction.
- `answer_text`: final response text.
- `classification`: one of `simple`, `research`, `hard`, or `high_stakes`.
- `confidence`: `low`, `medium`, or `high`.
- `verified_claims`: list of claims with source, checked timestamp, and verdict.
- `unverified_claims`: list of claims that could not be verified.
- `calibration`: absent only for `simple` answers that passed verification; otherwise contains critique and revisions.
- `memory_updates`: extracted durable facts, preferences, or experience candidates.

## Verification Policy

The assistant must not present fragile data as certain without verification.

Claims require verification when they include:

- current or recent facts;
- prices, laws, policies, schedules, software/library versions, APIs, or rankings;
- dates, statistics, quantities, metrics, or benchmarks;
- named public people, companies, products, or institutions;
- medical, legal, financial, security, or safety guidance;
- any answer where outdated information could mislead the user.

If a claim cannot be verified, the answer must either remove the claim or label it as unverified. Verification records must be stored with source URL or tool name, timestamp, and verdict.

## Second-Pass Calibration

For `hard`, `research`, and `high_stakes` requests, the system must run a second pass before responding. The calibration pass must:

- identify possible mistakes, missing context, unstated assumptions, and overconfident claims;
- compare the draft against retrieved memory and verification results;
- downgrade confidence when evidence is incomplete;
- produce a revised answer or a clear reason for asking a follow-up question.

Calibration output is persisted even when it does not change the final answer.

## Memory And Experience

Phase 1 uses SQLite as the canonical store. Markdown files can be generated from the store for human reading.

Core tables:

- `interactions`: incoming questions, channel metadata, answer ids, timestamps.
- `answers`: final answer packages, confidence, classification, and response text.
- `evidence`: claim-level verification records.
- `memory_items`: durable user facts, preferences, knowledge snippets, and accepted corrections.
- `experience_items`: reusable answer strategies, failure lessons, prompts, and tool-use notes.
- `profile_snapshots`: periodic user profile summaries.
- `learning_reports`: generated report metadata and markdown body.
- `skill_drafts`: proposed skill name, trigger, instructions, source experience ids, review status.

Memory writes must be append-first. Updates that supersede older facts must keep a link to the superseded record so later behavior remains auditable.

## User Profile

The profile should be a compact, inspectable summary, not an opaque embedding-only memory.

Phase 1 profile fields:

- recurring topics;
- preferred answer style;
- active projects;
- known tools and platforms;
- correction history;
- open learning goals;
- confidence notes, such as areas where the assistant should verify more aggressively.

The profile is used as context for later answers but must not override explicit user instructions in the current question.

## Learning Reports

Phase 1 provides a CLI command to generate a report for a date range. Later phases can schedule this command.

Each report includes:

- question volume and topic distribution;
- important questions and distilled answers;
- newly learned concepts;
- unresolved or weakly verified areas;
- profile changes;
- recommended next learning actions;
- suggested skill drafts or experience items to review.

The report is stored in SQLite and exported as markdown under `reports/`.

## Skill Evolution

The system supports two skill creation paths:

- Automatic draft: repeated patterns in high-quality answers or corrected failures create a skill draft.
- User-requested draft: explicit requests such as "turn this into a skill" create a skill draft from selected interactions.

Phase 1 never activates generated skills automatically. Skill drafts are written under `skills/drafts/<skill-name>/SKILL.md` with metadata, trigger rules, instructions, and source interaction ids. A human must review and promote a draft before it becomes active.

## Feishu Integration

Phase 1 must support:

- Feishu callback URL challenge response.
- Message event parsing into an internal `IncomingMessage`.
- Idempotency by Feishu event id or message id.
- Reply delivery through a Feishu client adapter.
- Local test mode that accepts fixture payloads without real Feishu credentials.

Production credentials must come from environment variables or an ignored local env file. Secrets must not be committed.

## Error Handling

The assistant should fail visibly and record the failure.

- Invalid Feishu callbacks return a clear HTTP error and no answer workflow run.
- Duplicate events return the stored response when possible.
- Missing model/search/Feishu credentials produce setup errors, not silent fallback answers.
- Verification tool failures are recorded and reflected in answer confidence.
- Storage failures prevent response generation unless the incoming event has already been safely persisted elsewhere.

## Testing Strategy

Implementation must follow TDD.

Minimum Phase 1 coverage:

- Feishu challenge and message event parsing.
- Idempotent event handling.
- Workflow classification and calibration trigger rules.
- Claim verification policy.
- SQLite persistence and append-first memory behavior.
- Profile update from interactions.
- Learning report generation.
- Skill draft generation from repeated experience.
- CLI local question path.

Network and model calls should be behind interfaces so tests can use deterministic fakes while production still uses Microsoft Agent Framework and configured tools.

## Documentation Requirements

Phase 1 must create and maintain:

- `README.md`: purpose, architecture, setup, commands, Feishu configuration, local testing, and safety model.
- `docs/code_manual.md`: module map, data contracts, workflow steps, extension points, and development rules.
- `docs/superpowers/specs/2026-06-26-search-assistant-design.md`: this design.
- `docs/superpowers/plans/<date>-<feature>.md`: implementation plan after written-spec approval.
- `.env.example`: documented configuration keys with no real secrets.

## Security And Privacy

- Store secrets only in environment variables or ignored local files.
- Do not log Feishu secrets, model keys, verification tokens, or user-private message content outside the local store.
- Keep local data paths explicit so the user can back up or delete their records.
- Treat user profile data as private operational memory.

## Phase 1 Acceptance Criteria

Phase 1 is accepted when:

1. A local test command can ask a question and receive an `AnswerPackage`.
2. A Feishu fixture payload can pass through the gateway and produce a reply payload through a fake Feishu client.
3. A hard question triggers calibration before final answer generation.
4. A data-bearing answer records at least one verified or explicitly unverified claim.
5. Interactions, answers, evidence, memory items, profile snapshots, reports, and skill drafts persist in SQLite.
6. A learning report markdown file can be generated from stored interactions.
7. A skill draft markdown file can be generated from repeated experience or explicit user request.
8. README and code manual explain how to run, test, configure, and extend the system.
9. Automated tests cover the behaviors above.

## Later Phases

After Phase 1, continue toward the full goal with:

- real Feishu app deployment and callback verification against live traffic;
- production model and search providers;
- scheduled report generation;
- human review UI or Feishu commands for approving memory and skill drafts;
- richer retrieval over long-term knowledge;
- monitoring, backup, and data retention controls.
