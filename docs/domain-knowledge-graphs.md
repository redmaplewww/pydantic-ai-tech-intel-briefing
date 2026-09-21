# Domain Knowledge Graphs

This project keeps a small reviewed domain graph layer alongside the existing
briefing evidence store. It follows the knowledge-organization pattern described
in chapter 3 of *AI Agents in Depth*: do not rely only on flat text chunks when a
domain needs multi-hop relationships, entity disambiguation, and durable shared
knowledge.

## Design

Each graph contains:

- a domain overview, kept as natural-language context;
- entities, such as systems, workflows, concepts, metrics, controls, and risks;
- typed relations, stored as source entity - relation - target entity triples;
- evidence references, which identify whether a node came from the tutorial,
  project contract, or a reviewed seed taxonomy.

The graph is deliberately not a replacement for source evidence. A briefing must
still preserve original URLs and evidence snippets. The graph acts as a semantic
backbone: it helps future retrieval and synthesis decide whether a source is
about a model, system interface, workflow, validation metric, or safety boundary.

## Seeded domains

The first three graphs are:

| Domain ID | Domain | Purpose |
| --- | --- | --- |
| `agent-engineering` | AI Agent 工程 | Harness, context management, tool loops, verification, memory, Agentic RAG, and GraphRAG. |
| `industrial-ai` | 工业 AI | Equipment events, MES, controlled work orders, machine-vision quality inspection, traceability, and approval/rollback controls. |
| `medical-imaging-ai` | 医学影像 AI | DICOM/PACS, imaging foundation models, annotation quality, lesion detection/segmentation, clinical validation, privacy, and human review. |

## Commands

Seed all default graphs into an isolated data directory:

```powershell
python -m search_assistant.cli knowledge-graph-seed --data-dir .local-data
```

Seed one graph:

```powershell
python -m search_assistant.cli knowledge-graph-seed --domain industrial-ai --data-dir .local-data
```

List graphs:

```powershell
python -m search_assistant.cli knowledge-graph-list --data-dir .local-data
```

Query graph nodes and their relation context:

```powershell
python -m search_assistant.cli knowledge-graph-query "MES 工单 写回" --domain industrial-ai --data-dir .local-data
```

Export a graph to Markdown for review:

```powershell
python -m search_assistant.cli knowledge-graph-export agent-engineering --data-dir .local-data
```

By default the export is written to `.local-data/knowledge-graphs/<domain>.md`,
which remains outside source control.

## Self-evolution compatibility

The self-evolution loop writes search-derived lessons as
`domain_knowledge_candidates`. These records are intentionally reviewable
candidates, not accepted graph facts. When a briefing or `evolve` run produces a
candidate, the project now searches the reviewed domain graphs and stores an
immutable candidate-to-entity link when there is a matching graph node.

This gives the project a safer growth path:

1. public-source evidence is preserved in the normal briefing store;
2. the candidate captures the possible reusable lesson;
3. the graph link shows which reviewed domain entity it may refine;
4. a human or later validation workflow can decide whether to update the graph
   seed itself.

Running `evolve` also backfills links for older candidates:

```powershell
python -m search_assistant.cli evolve --data-dir .local-data
```

## Update policy

The checked-in graph seeds are a reviewed starting point, not a live claim of
complete domain coverage. When adding new knowledge:

1. keep the original natural-language source and URL in the normal briefing
   evidence store;
2. add or update the relevant graph entity and relation only when the source
   supports the relationship;
3. preserve both prose and triples, because triples alone can lose conditions,
   time boundaries, and uncertainty;
4. add regression tests for query and export behavior.

This mirrors the tutorial's recommendation to combine incremental updates with
periodic review: source evidence changes first, reviewed graph structure follows
through auditable code or data diffs.
