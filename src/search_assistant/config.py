from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _to_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_list(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    items = [item.strip().lower() for item in value.split(",")]
    return [item for item in items if item]


def _to_float(value: str | float | int | None, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _to_int(value: str | int | None, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value if value > 0 else default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _load_env_local(path: Path = Path(".env.local")) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


class Settings(BaseModel):
    data_dir: Path = Field(default=Path(".local-data"))
    database_path: Path | None = None
    feishu_enabled: bool = False
    feishu_app_id: str | None = None
    feishu_app_secret: str | None = None
    feishu_verification_token: str | None = None
    feishu_encrypt_key: str | None = None
    model_provider: str = "glm"
    glm_api_key: str | None = None
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    glm_model: str = "glm-4.7"
    glm_timeout_seconds: float = 120.0
    search_provider: str = "hybrid"
    allow_fake_runtime: bool = False
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    brave_search_api_key: str | None = None
    brave_search_base_url: str = "https://api.search.brave.com/res/v1/web/search"
    searxng_base_url: str = "http://localhost:8080/search"
    searxng_timeout_seconds: float = 12.0
    mcp_search_config_path: Path | None = None
    mcp_search_timeout_seconds: float = 18.0
    agent_reach_enabled: bool = False
    agent_reach_command: str = "agent-reach"
    agent_reach_timeout_seconds: float = 30.0
    agent_reach_doctor_cache_seconds: float = 300.0
    domestic_rss_base_url: str = "http://127.0.0.1:1200"
    domestic_rss_config_path: Path | None = None
    domestic_rss_timeout_seconds: float = 8.0
    bilibili_search_base_url: str = "https://api.bilibili.com/x/web-interface/search/type"
    duckduckgo_timeout_seconds: float = 12.0
    browser_search_base_url: str = "https://cn.bing.com/search"
    browser_search_market: str = "zh-CN"
    browser_search_engines: list[str] = Field(default_factory=lambda: ["bing", "baidu", "google"])
    baidu_search_base_url: str = "https://www.baidu.com/baidu"
    google_search_base_url: str = "https://www.google.com/search"
    browser_search_timeout_seconds: float = 8.0
    browser_content_timeout_seconds: float = 3.0
    browser_enrich_max_results: int = 2
    workflow_search_budget_seconds: float = 45.0
    briefing_search_budget_seconds: float = 90.0
    briefing_planning_timeout_seconds: float = 40.0
    # Briefing synthesis needs more headroom than general calls: 25+ sources
    # regularly take 60-70s (measured), so the default is 120s instead of the
    # 60s general deepseek timeout.  A synthesis that times out silently
    # degrades the whole report to the deterministic fallback templates.
    briefing_synthesis_timeout_seconds: float = 120.0
    deepseek_timeout_seconds: float = 60.0
    briefing_max_queries: int = 28
    briefing_results_per_query: int = 10
    briefing_max_sources: int = 50
    briefing_model_max_sources: int = 12
    briefing_timezone: str = "Asia/Shanghai"
    admin_user_ids: list[str] = Field(default_factory=list)

    # Knowledge-graph semantic matching
    # These fields use an OpenAI-compatible embedding API.  If OPENAI_API_KEY is
    # not set, the provider falls back to DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL so
    # DeepSeek-compatible endpoints or third-party proxies can be reused.
    embedding_enabled: bool = True
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        source: Mapping[str, str]
        if env is None:
            import os

            source = {**_load_env_local(), **os.environ}
        else:
            source = env

        data_dir = Path(source.get("SEARCH_ASSISTANT_DATA_DIR", ".local-data"))
        raw_database_path = source.get("SEARCH_ASSISTANT_DATABASE_PATH")
        database_path = Path(raw_database_path) if raw_database_path else data_dir / "assistant.sqlite3"

        values: dict[str, Any] = {
            "data_dir": data_dir,
            "database_path": database_path,
            "feishu_enabled": _to_bool(source.get("SEARCH_ASSISTANT_FEISHU_ENABLED"), default=False),
            "feishu_app_id": source.get("FEISHU_APP_ID") or None,
            "feishu_app_secret": source.get("FEISHU_APP_SECRET") or None,
            "feishu_verification_token": source.get("FEISHU_VERIFICATION_TOKEN") or None,
            "feishu_encrypt_key": source.get("FEISHU_ENCRYPT_KEY") or None,
            "model_provider": source.get("SEARCH_ASSISTANT_MODEL_PROVIDER", "glm"),
            "glm_api_key": source.get("GLM_API_KEY") or None,
            "glm_base_url": source.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
            "glm_model": source.get("GLM_MODEL", "glm-4.7"),
            "glm_timeout_seconds": _to_float(source.get("GLM_TIMEOUT_SECONDS"), 120.0),
            "search_provider": source.get("SEARCH_ASSISTANT_SEARCH_PROVIDER", "hybrid"),
            "allow_fake_runtime": _to_bool(source.get("SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME"), default=False),
            "deepseek_api_key": source.get("DEEPSEEK_API_KEY") or None,
            "deepseek_base_url": source.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "deepseek_model": source.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "brave_search_api_key": source.get("BRAVE_SEARCH_API_KEY") or None,
            "brave_search_base_url": source.get(
                "BRAVE_SEARCH_BASE_URL",
                "https://api.search.brave.com/res/v1/web/search",
            ),
            "searxng_base_url": source.get("SEARXNG_BASE_URL", "http://localhost:8080/search"),
            "searxng_timeout_seconds": _to_float(source.get("SEARXNG_TIMEOUT_SECONDS"), 12.0),
            "mcp_search_config_path": (
                Path(source["SEARCH_ASSISTANT_MCP_SEARCH_CONFIG"])
                if source.get("SEARCH_ASSISTANT_MCP_SEARCH_CONFIG")
                else None
            ),
            "mcp_search_timeout_seconds": _to_float(source.get("MCP_SEARCH_TIMEOUT_SECONDS"), 18.0),
            "agent_reach_enabled": _to_bool(source.get("SEARCH_ASSISTANT_AGENT_REACH_ENABLED"), default=False),
            "agent_reach_command": source.get("AGENT_REACH_COMMAND", "agent-reach"),
            "agent_reach_timeout_seconds": _to_float(source.get("AGENT_REACH_TIMEOUT_SECONDS"), 30.0),
            "agent_reach_doctor_cache_seconds": _to_float(source.get("AGENT_REACH_DOCTOR_CACHE_SECONDS"), 300.0),
            "domestic_rss_base_url": source.get("RSSHUB_BASE_URL", "http://127.0.0.1:1200"),
            "domestic_rss_config_path": (
                Path(source["SEARCH_ASSISTANT_DOMESTIC_RSS_CONFIG"])
                if source.get("SEARCH_ASSISTANT_DOMESTIC_RSS_CONFIG")
                else None
            ),
            "domestic_rss_timeout_seconds": _to_float(source.get("DOMESTIC_RSS_TIMEOUT_SECONDS"), 8.0),
            "bilibili_search_base_url": source.get(
                "BILIBILI_SEARCH_BASE_URL",
                "https://api.bilibili.com/x/web-interface/search/type",
            ),
            "duckduckgo_timeout_seconds": _to_float(source.get("DUCKDUCKGO_TIMEOUT_SECONDS"), 12.0),
            "browser_search_base_url": source.get("BROWSER_SEARCH_BASE_URL", "https://cn.bing.com/search"),
            "browser_search_market": source.get("BROWSER_SEARCH_MARKET", "zh-CN"),
            "browser_search_engines": _to_list(
                source.get("BROWSER_SEARCH_ENGINES"),
                ["bing", "baidu", "google"],
            ),
            "baidu_search_base_url": source.get("BAIDU_SEARCH_BASE_URL", "https://www.baidu.com/baidu"),
            "google_search_base_url": source.get("GOOGLE_SEARCH_BASE_URL", "https://www.google.com/search"),
            "browser_search_timeout_seconds": _to_float(source.get("BROWSER_SEARCH_TIMEOUT_SECONDS"), 8.0),
            "browser_content_timeout_seconds": _to_float(source.get("BROWSER_CONTENT_TIMEOUT_SECONDS"), 3.0),
            "browser_enrich_max_results": _to_int(source.get("BROWSER_ENRICH_MAX_RESULTS"), 2),
            "workflow_search_budget_seconds": _to_float(source.get("WORKFLOW_SEARCH_BUDGET_SECONDS"), 45.0),
            "briefing_search_budget_seconds": _to_float(source.get("BRIEFING_SEARCH_BUDGET_SECONDS"), 90.0),
            "briefing_planning_timeout_seconds": _to_float(
                source.get("BRIEFING_PLANNING_TIMEOUT_SECONDS"),
                40.0,
            ),
            "briefing_synthesis_timeout_seconds": _to_float(
                source.get("BRIEFING_SYNTHESIS_TIMEOUT_SECONDS"),
                120.0,
            ),
            "deepseek_timeout_seconds": _to_float(source.get("DEEPSEEK_TIMEOUT_SECONDS"), 60.0),
            "briefing_max_queries": _to_int(source.get("BRIEFING_MAX_QUERIES"), 28),
            "briefing_results_per_query": _to_int(source.get("BRIEFING_RESULTS_PER_QUERY"), 10),
            "briefing_max_sources": _to_int(source.get("BRIEFING_MAX_SOURCES"), 50),
            "briefing_model_max_sources": _to_int(source.get("BRIEFING_MODEL_MAX_SOURCES"), 12),
            "briefing_timezone": source.get("BRIEFING_TIMEZONE", "Asia/Shanghai"),
            "admin_user_ids": _to_list(source.get("SEARCH_ASSISTANT_ADMIN_USER_IDS"), []),
            "embedding_enabled": _to_bool(source.get("SEARCH_ASSISTANT_EMBEDDING_ENABLED"), default=True),
            "openai_api_key": source.get("OPENAI_API_KEY") or source.get("DEEPSEEK_API_KEY") or None,
            "openai_base_url": source.get("OPENAI_BASE_URL")
            or source.get("DEEPSEEK_BASE_URL")
            or "https://api.openai.com/v1",
            "openai_embedding_model": source.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        }
        return cls(**values)
