# Configuration

## Local Files

Copy `.env.example` to `.env.local` and keep the populated file out of Git.
The application loads `.env.local` only when no explicit environment mapping is
provided. Process environment variables take precedence.

```powershell
Copy-Item .env.example .env.local
```

Use a deployment secret manager for hosted instances. Do not place credentials
in command lines, reports, SQLite fixtures, or checked-in configuration files.

## Required Model Settings

| Variable | Purpose |
| --- | --- |
| `SEARCH_ASSISTANT_MODEL_PROVIDER` | `glm` by default; `deepseek` selects the DeepSeek OpenAI-compatible provider through the Pydantic AI runtime. |
| `GLM_API_KEY` | Required when `SEARCH_ASSISTANT_MODEL_PROVIDER=glm`. |
| `GLM_BASE_URL` | OpenAI-compatible GLM endpoint. |
| `GLM_MODEL` | Default: `glm-4.7`. |
| `GLM_TIMEOUT_SECONDS` | Per-model request time budget. |
| `DEEPSEEK_API_KEY` | Required when `SEARCH_ASSISTANT_MODEL_PROVIDER=deepseek`. |
| `DEEPSEEK_BASE_URL` | OpenAI-compatible DeepSeek endpoint. |
| `DEEPSEEK_MODEL` | Default: `deepseek-v4-flash`. |
| `DEEPSEEK_TIMEOUT_SECONDS` | Per-model request time budget for DeepSeek. |

`SEARCH_ASSISTANT_MODEL_PROVIDER` chooses the model provider only. GLM and
DeepSeek both run through the same Pydantic AI runtime boundary; retrieval,
verification, calibration, review, memory, evaluation, and evolution remain in
the project harness.

The project supports local encrypted configuration tooling in its development
environment, but the public repository intentionally documents only variable
names and never a secret value.

## Retrieval Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEARCH_ASSISTANT_SEARCH_PROVIDER` | `hybrid` | `hybrid`, `agent-reach`, `mcp`, `browser`, `brave`, `searxng`, or `duckduckgo`. Use `agent-reach` only on machines with the Agent Reach CLI installed. |
| `SEARCH_ASSISTANT_MCP_SEARCH_CONFIG` | checked-in default when unset | MCP server and read-only search bindings. |
| `MCP_SEARCH_TIMEOUT_SECONDS` | `18` | MCP request timeout. |
| `SEARCH_ASSISTANT_AGENT_REACH_ENABLED` | `false` | When `true` and the provider is `hybrid`, run Agent Reach as a **supplementary pass after** MCP/browser search. Agent Reach routes spawn a subprocess per query (mcporter for Exa), so they run last to avoid starving the briefing search budget; MCP/browser results are collected first and Agent Reach adds scoped or Exa results when the budget allows. |
| `AGENT_REACH_COMMAND` | `agent-reach` | Agent Reach CLI executable or absolute path. |
| `AGENT_REACH_TIMEOUT_SECONDS` | `30` | Per Agent Reach routed command timeout. For the daily briefing, lower values (10-15) stop a slow route from consuming the shared search budget. |
| `AGENT_REACH_DOCTOR_CACHE_SECONDS` | `300` | Cache window for `agent-reach doctor --json` results. |
| `BROWSER_SEARCH_ENGINES` | `bing,baidu,google` | Public engines for browser search; `duckduckgo` can also be included. |
| `BROWSER_SEARCH_BASE_URL` | `https://cn.bing.com/search` | Bing public-result endpoint. |
| `BAIDU_SEARCH_BASE_URL` | `https://www.baidu.com/baidu` | Baidu public-index base URL; the client can fall back to other public Baidu endpoints when one is challenged. |
| `BILIBILI_SEARCH_BASE_URL` | public Bilibili endpoint | No-login Bilibili search endpoint. |
| `RSSHUB_BASE_URL` | `http://127.0.0.1:1200` | Local optional RSSHub instance. |
| `SEARCH_ASSISTANT_DOMESTIC_RSS_CONFIG` | unset | Optional operator-owned RSS source catalog. |
| `BRAVE_SEARCH_API_KEY` | unset | Required only when the Brave provider is selected. |
| `SEARXNG_BASE_URL` | `http://localhost:8080/search` | Self-hosted SearXNG JSON endpoint. |

### Optional Agent Reach Install

Agent Reach is an optional CLI capability router. Do not commit a local virtual
environment or a populated `.env.local` file to make it available on another
machine. Install the checked-in extra instead:

```powershell
python -m pip install -c constraints-dev.txt -e ".[dev,agent-reach]"
agent-reach doctor --json
```

When the project virtual environment is activated, the default
`AGENT_REACH_COMMAND=agent-reach` works because the CLI entry point is on the
virtual environment `PATH`. If the application is started by a scheduler or
service without activating the virtual environment, set `AGENT_REACH_COMMAND`
to the absolute path of that environment's `agent-reach` executable in local
or deployment-only configuration.

The Python extra installs the Agent Reach CLI and its Python dependencies.
**Most platform backends still require external tools or API configuration**;
`agent-reach doctor --json` lists which channels are ready (`status: ok`),
missing a backend (`warn`), or entirely unavailable (`off`). Channel
availability is checked at runtime, so an unconfigured route returns empty
results instead of raising; the hybrid provider then falls back to MCP/browser
search. Per-channel requirements:

| Channel | Backend | Install / configure |
| --- | --- | --- |
| **General web (exa_search)** | mcporter + Exa MCP | `npm install -g mcporter`, then `mcporter config add exa https://mcp.exa.ai/mcp` and set the Exa API key. Without this, plain (non-`site:`) web queries have **no** Agent Reach route and stay empty. |
| GitHub | gh CLI | Install from <https://cli.github.com>; the channel activates once `gh` is on `PATH`. |
| YouTube | yt-dlp | `yt-dlp` must be installed; add `--js-runtimes node` to `~/.config/yt-dlp/config` (Node.js is usually already present). |
| Bilibili | built-in public API | Zero-config; the search API is reached via `curl` with browser headers. |
| V2EX / RSS | built-in | Zero-config public endpoints / feedparser. |
| X / Reddit / Xiaohongshu / WeChat / LinkedIn / Facebook / Instagram | twitter-cli, rdt-cli, OpenCLI, ... | Require installing the CLI **and** a browser login state / cookies. This project deliberately does not automate logged-in accounts, so these channels are left `off` unless the operator enables them explicitly. |

Recommended enable path for the daily briefing:

1. Run `agent-reach doctor --json` and confirm at least `bilibili` and `v2ex`/`rss` show `ok`, and `exa_search` shows `ok` after installing mcporter.
2. Keep `SEARCH_ASSISTANT_AGENT_REACH_ENABLED=true` with the `hybrid` provider so MCP/browser search runs first and Agent Reach adds supplementary routes when the budget allows.
3. If no general-web backend will be installed, set `SEARCH_ASSISTANT_AGENT_REACH_ENABLED=false` to avoid the empty-route calls (each unresolved query is recorded as `AgentReachSearchClient:empty` in the provider trace).

The project treats Agent Reach as a read-only capability layer: it never passes
login state and never performs writes on behalf of the user. See the upstream
Agent Reach install guide before enabling channels with
`agent-reach install --system`.

## Daily Briefing Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `BRIEFING_TIMEZONE` | `Asia/Shanghai` | Local calendar date used for daily report paths and database records. |
| `BRIEFING_MAX_QUERIES` | `28` | Maximum combined planned and deterministic queries. |
| `BRIEFING_RESULTS_PER_QUERY` | `10` | Per-query retrieval limit. |
| `BRIEFING_MAX_SOURCES` | `50` | Full retained source cap. |
| `BRIEFING_MODEL_MAX_SOURCES` | `12` | Ranked evidence cap supplied to the configured Pydantic AI model provider. |
| `BRIEFING_SEARCH_BUDGET_SECONDS` | `90` | Concurrent retrieval budget. |
| `BRIEFING_PLANNING_TIMEOUT_SECONDS` | `40` | Planning budget passed to the configured Pydantic AI model provider. |
| `BRIEFING_SYNTHESIS_TIMEOUT_SECONDS` | `120` | Synthesis budget passed to the configured Pydantic AI model provider. |

## Optional Feishu Settings

Set `SEARCH_ASSISTANT_FEISHU_ENABLED=true` and provide the Feishu app
credentials through your local or deployment secret store. The app supports
webhook parsing, long connection, and polling. Start with `feishu-doctor` and
an isolated fixture before sending a real message.

```powershell
python -m search_assistant.cli feishu-doctor
python -m search_assistant.cli feishu-fixture tests/fixtures/feishu_message_event.json
```

## Knowledge Graph Semantic Matching

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEARCH_ASSISTANT_EMBEDDING_ENABLED` | `true` | When `true`, enable semantic matching for English-to-Chinese knowledge-graph queries. |
| `OPENAI_API_KEY` | unset | Primary OpenAI-compatible embedding API key. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Primary OpenAI-compatible embedding endpoint. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name. |

If `OPENAI_API_KEY` is empty, the provider falls back to the resolved
`DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` from `Settings` (this includes
`.env.local` values), and only as a last resort to process environment
variables. If no key is available, semantic matching is disabled and the graph
uses literal matching only.

**DeepSeek's API has no embedding endpoint.** To actually enable semantic
matching, point the three `OPENAI_*` variables at an OpenAI-compatible embedding
service:

```ini
# Zhipu GLM (same vendor as the chat model)
OPENAI_API_KEY=<GLM_API_KEY>
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
OPENAI_EMBEDDING_MODEL=embedding-3

# Alibaba DashScope
OPENAI_API_KEY=<DashScope key>
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_EMBEDDING_MODEL=text-embedding-v4

# SiliconFlow (open models, e.g. BGE)
OPENAI_API_KEY=<SiliconFlow key>
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_EMBEDDING_MODEL=BAAI/bge-m3
```

After switching the embedding model, clear the `entity_embeddings` cache table
so stale vectors from the previous model are not reused. Run
`python -m search_assistant.cli knowledge-graph-query "<query>" --domain <id>` to
confirm semantic hits appear.

## MCP Configuration

`configs/public-sources.mcp.json` starts two local stdio servers:

- `scripts/public_sources_mcp.py` for GitHub, arXiv, Hacker News, and Stack
  Exchange.
- `scripts/domestic_rss_mcp.py` for a configured local RSSHub instance.

Only add trusted, read-only tools that return URL-bearing result objects. Do
not add write-capable MCP servers to a retrieval binding.
