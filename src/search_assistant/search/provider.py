from __future__ import annotations

import asyncio
import html
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from search_assistant.config import Settings
from search_assistant.contracts import ProviderTraceEvent, SourceEvidence


SearchResult = SourceEvidence
SearchTransport = Callable[[str, dict[str, str]], str]


class SearchClient(Protocol):
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        ...


class SearchProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchOutcome:
    results: list[SearchResult]
    provider_events: list[ProviderTraceEvent]


@dataclass(frozen=True)
class McpSearchBinding:
    name: str
    tool_name: str
    argument_template: dict[str, object]
    toolset: Any
    site_domains: tuple[str, ...] = ()


class McpSearchClient:
    """Calls explicitly configured, read-only MCP search tools and normalizes URL-bearing results."""

    def __init__(
        self,
        config_path: str | Path,
        timeout_seconds: float = 18.0,
        toolset_factory: Callable[[str, dict[str, object], Path, float], Any] | None = None,
        server_environment: Mapping[str, str] | None = None,
    ):
        self.config_path = Path(config_path).resolve()
        self.timeout_seconds = timeout_seconds
        self.toolset_factory = toolset_factory or _mcp_toolset_from_config
        self.server_environment = dict(server_environment or {})
        self.bindings = self._load_bindings()
        # Pydantic AI MCPToolset owns an async transport/session. Daily briefing
        # retrieval is threaded, so concurrent event loops must not reuse it.
        self._call_lock = Lock()

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self.search_with_events(query, limit=limit).results

    def search_with_events(self, query: str, limit: int = 5) -> SearchOutcome:
        with self._call_lock:
            return _run_async_from_sync(lambda: self._search_async_with_events(query, limit))

    async def _search_async(self, query: str, limit: int) -> list[SearchResult]:
        return (await self._search_async_with_events(query, limit)).results

    async def _search_async_with_events(self, query: str, limit: int) -> SearchOutcome:
        merged: list[SearchResult] = []
        seen: set[str] = set()
        provider_events: list[ProviderTraceEvent] = []
        scoped_domains = _site_domains(query)
        for binding in self.bindings:
            provider = f"mcp:{binding.name}"
            if scoped_domains and not _binding_supports_scoped_domains(binding, scoped_domains):
                provider_events.append(
                    _provider_event(
                        provider=provider,
                        query=query,
                        status="skipped",
                        reason="site_query_not_bound_to_mcp_source",
                    )
                )
                continue
            started = time.monotonic()
            try:
                payload = _materialize_mcp_arguments(binding.argument_template, query, limit)
                raw_result = await asyncio.wait_for(
                    binding.toolset.direct_call_tool(binding.tool_name, payload),
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                provider_events.append(
                    _provider_event(
                        provider=provider,
                        query=query,
                        status="error",
                        error=_safe_provider_error(exc),
                        elapsed_ms=_elapsed_ms(started),
                    )
                )
                continue

            binding_results: list[SearchResult] = []
            for result in _mcp_results_to_search_results(raw_result, binding.name):
                if scoped_domains and not _matches_scoped_domains(result.url, scoped_domains):
                    continue
                key = _dedupe_key(result.url)
                if key in seen:
                    continue
                seen.add(key)
                binding_results.append(result)
                merged.append(result)
                if len(merged) >= limit:
                    provider_events.append(
                        _provider_event(
                            provider=provider,
                            query=query,
                            status="success",
                            result_count=len(binding_results),
                            elapsed_ms=_elapsed_ms(started),
                        )
                    )
                    return SearchOutcome(merged, provider_events)
            provider_events.append(
                _provider_event(
                    provider=provider,
                    query=query,
                    status="success" if binding_results else "empty",
                    result_count=len(binding_results),
                    elapsed_ms=_elapsed_ms(started),
                )
            )
        return SearchOutcome(merged, provider_events)

    def health(self) -> dict[str, object]:
        return {
            "config_path": str(self.config_path),
            "bindings": [binding.name for binding in self.bindings],
            "tool_names": [binding.tool_name for binding in self.bindings],
        }

    def _load_bindings(self) -> list[McpSearchBinding]:
        if not self.config_path.exists():
            raise RuntimeError(f"MCP search config does not exist: {self.config_path}")
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"MCP search config is not valid JSON: {self.config_path}") from exc
        if not isinstance(raw, dict):
            raise RuntimeError("MCP search config must be a JSON object")

        servers = raw.get("mcpServers")
        bindings = raw.get("searchBindings")
        if not isinstance(servers, dict) or not isinstance(bindings, dict) or not bindings:
            raise RuntimeError("MCP search config requires non-empty mcpServers and searchBindings objects")

        loaded: list[McpSearchBinding] = []
        for name, raw_binding in bindings.items():
            server = servers.get(name)
            if not isinstance(name, str) or not isinstance(server, dict) or not isinstance(raw_binding, dict):
                raise RuntimeError("Every MCP search binding must reference a configured server")
            tool_name = str(raw_binding.get("tool") or "search").strip()
            argument_template = raw_binding.get("arguments", {"query": "{query}", "limit": "{limit}"})
            if not tool_name or not isinstance(argument_template, dict):
                raise RuntimeError(f"MCP search binding '{name}' has an invalid tool or arguments value")
            raw_site_domains = raw_binding.get("siteDomains", [])
            if isinstance(raw_site_domains, str):
                raw_site_domains = [raw_site_domains]
            if not isinstance(raw_site_domains, list) or not all(isinstance(domain, str) for domain in raw_site_domains):
                raise RuntimeError(f"MCP search binding '{name}' siteDomains must be a list of strings")

            server_config = dict(server)
            if self.server_environment:
                configured_env = server_config.get("env", {})
                if configured_env is not None and not isinstance(configured_env, dict):
                    raise RuntimeError(f"MCP server '{name}' env must be an object")
                server_config["env"] = {
                    **{str(key): str(value) for key, value in (configured_env or {}).items()},
                    **self.server_environment,
                }

            toolset = self.toolset_factory(name, server_config, self.config_path.parent, self.timeout_seconds)
            loaded.append(
                McpSearchBinding(
                    name=name,
                    tool_name=tool_name,
                    argument_template=dict(argument_template),
                    toolset=toolset,
                    site_domains=tuple(
                        domain for domain in (_normalize_domain(value) for value in raw_site_domains) if domain
                    ),
                )
            )
        return loaded


class CompositeSearchClient:
    """Merges independent retrieval backends while preserving the source provider for every URL."""

    def __init__(self, clients: list[SearchClient], primary_sufficient_results: int | None = None):
        self.clients = clients
        self.primary_sufficient_results = primary_sufficient_results

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self.search_with_events(query, limit=limit).results

    def search_with_events(self, query: str, limit: int = 5) -> SearchOutcome:
        merged: list[SearchResult] = []
        seen: set[str] = set()
        provider_events: list[ProviderTraceEvent] = []
        for index, client in enumerate(self.clients):
            outcome = search_with_provider_events(client, query, limit=limit)
            provider_events.extend(outcome.provider_events)
            candidates = outcome.results
            for result in candidates:
                key = _dedupe_key(result.url)
                if key in seen:
                    provider_events.append(
                        _provider_event(
                            provider=result.provider,
                            query=query,
                            status="skipped",
                            result_count=0,
                            reason="duplicate_url",
                        )
                    )
                    continue
                seen.add(key)
                merged.append(result)
                if len(merged) >= limit:
                    return SearchOutcome(merged, provider_events)
            if index == 0 and self.primary_sufficient_results is not None:
                if len(merged) >= min(limit, self.primary_sufficient_results):
                    provider_events.append(
                        _provider_event(
                            provider="composite",
                            query=query,
                            status="skipped",
                            result_count=len(merged),
                            reason="primary_sufficient_results",
                        )
                    )
                    return SearchOutcome(merged, provider_events)
        return SearchOutcome(merged, provider_events)


AgentReachCommandRunner = Callable[[list[str], float], str]


class AgentReachSearchClient:
    """Delegates retrieval to the installed Agent Reach capability router.

    Agent Reach is intentionally a CLI capability layer rather than a Python
    search SDK.  This adapter therefore first runs ``agent-reach doctor
    --json`` and then calls the read-only command selected by Agent Reach's
    public routing contract, such as ``mcporter`` for Exa search, ``bili`` for
    Bilibili, ``yt-dlp`` for YouTube, and ``opencli`` for login-state
    platforms.
    """

    def __init__(
        self,
        command: str = "agent-reach",
        timeout_seconds: float = 30.0,
        doctor_cache_seconds: float = 300.0,
        command_runner: AgentReachCommandRunner | None = None,
        require_available: bool = True,
    ):
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.doctor_cache_seconds = doctor_cache_seconds
        self.command_runner = command_runner or self._run_subprocess
        self.require_available = require_available
        self._doctor_cache: dict[str, Any] | None = None
        self._doctor_checked_at = 0.0

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if limit <= 0:
            return []

        doctor = self._doctor()
        commands = self._commands_for_query(query, limit, doctor)
        if not commands:
            message = _agent_reach_unavailable_message(query, doctor)
            if self.require_available:
                raise SearchProviderError(message)
            return []

        checked_at = datetime.now(UTC).isoformat()
        merged: list[SearchResult] = []
        seen: set[str] = set()
        scoped_domains = _site_domains(query)
        errors: list[str] = []
        for command, provider in commands:
            try:
                raw_output = self.command_runner(command, self.timeout_seconds)
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                continue
            for result in _agent_reach_output_to_search_results(raw_output, provider, checked_at):
                if scoped_domains and not _matches_scoped_domains(result.url, scoped_domains):
                    continue
                key = _dedupe_key(result.url)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(result)
                if len(merged) >= limit:
                    return merged

        if not merged and self.require_available and errors:
            raise SearchProviderError("Agent Reach commands returned no usable URLs: " + "; ".join(errors[:3]))
        return merged

    def health(self) -> dict[str, object]:
        doctor = self._doctor()
        active = {
            channel: result.get("active_backend")
            for channel, result in doctor.items()
            if isinstance(result, dict) and result.get("active_backend")
        }
        return {"command": self.command, "active_backends": active}

    def _doctor(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._doctor_cache is not None and now - self._doctor_checked_at <= self.doctor_cache_seconds:
            return self._doctor_cache

        try:
            raw_output = self.command_runner([self.command, "doctor", "--json"], self.timeout_seconds)
        except FileNotFoundError as exc:
            raise SearchProviderError(
                "Agent Reach CLI was not found. Install it from "
                "https://github.com/Panniantong/Agent-Reach and keep "
                "SEARCH_ASSISTANT_SEARCH_PROVIDER=agent-reach only on machines that have it."
            ) from exc
        except Exception as exc:
            raise SearchProviderError(f"Agent Reach doctor failed: {exc}") from exc

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise SearchProviderError("Agent Reach doctor did not return valid JSON") from exc
        if not isinstance(parsed, dict):
            raise SearchProviderError("Agent Reach doctor JSON must be an object")

        self._doctor_cache = parsed
        self._doctor_checked_at = now
        return parsed

    def _commands_for_query(
        self,
        query: str,
        limit: int,
        doctor: Mapping[str, Any],
    ) -> list[tuple[list[str], str]]:
        commands: list[tuple[list[str], str]] = []
        clean_query = _query_without_site_directives(query) or query
        scoped_domains = _site_domains(query)

        for channel in _agent_reach_channels_for_domains(scoped_domains):
            commands.extend(self._commands_for_channel(channel, clean_query, limit, doctor))

        if not scoped_domains or not commands:
            commands.extend(self._commands_for_channel("exa_search", query, limit, doctor))
        elif _agent_reach_channel_available(doctor, "exa_search"):
            # Keep Exa as the final Agent Reach fallback for scoped public
            # searches.  The original site: directive remains in the query so
            # the downstream URL-domain guard can still enforce scope.
            commands.extend(self._commands_for_channel("exa_search", query, limit, doctor))

        unique: list[tuple[list[str], str]] = []
        seen: set[tuple[str, ...]] = set()
        for command, provider in commands:
            key = tuple(command)
            if key in seen:
                continue
            seen.add(key)
            unique.append((command, provider))
        return unique

    def _commands_for_channel(
        self,
        channel: str,
        query: str,
        limit: int,
        doctor: Mapping[str, Any],
    ) -> list[tuple[list[str], str]]:
        if not _agent_reach_channel_available(doctor, channel):
            return []

        active_backend = _agent_reach_active_backend(doctor, channel).lower()
        limit_text = str(max(1, limit))

        if channel == "exa_search":
            return [
                (
                    ["mcporter", "call", "exa.web_search_exa", f"query={query}", f"numResults={limit_text}"],
                    "agent-reach:exa_search:mcporter",
                )
            ]
        if channel == "github":
            return [
                (
                    ["gh", "search", "repos", query, "--limit", limit_text, "--json", "fullName,description,url"],
                    "agent-reach:github:gh",
                )
            ]
        if channel == "youtube":
            return [
                (
                    ["yt-dlp", "--dump-json", f"ytsearch{limit_text}:{query}"],
                    "agent-reach:youtube:yt-dlp",
                )
            ]
        if channel == "bilibili":
            if "bili-cli" in active_backend:
                return [
                    (
                        ["bili", "search", query, "--type", "video", "-n", limit_text],
                        "agent-reach:bilibili:bili-cli",
                    )
                ]
            if "opencli" in active_backend:
                return [
                    (
                        ["opencli", "bilibili", "search", query, "-f", "yaml"],
                        "agent-reach:bilibili:opencli",
                    )
                ]
            return [
                (
                    ["curl", "-s", "-A", _AGENT_REACH_BROWSER_UA, _agent_reach_bilibili_search_api_url(query)],
                    "agent-reach:bilibili:search-api",
                )
            ]
        if channel == "twitter":
            if "opencli" in active_backend:
                return [
                    (
                        ["opencli", "twitter", "search", query, "-f", "yaml"],
                        "agent-reach:twitter:opencli",
                    )
                ]
            return [
                (
                    ["twitter", "search", query, "-n", limit_text],
                    "agent-reach:twitter:twitter-cli",
                )
            ]
        if channel == "reddit":
            if "rdt" in active_backend:
                return [
                    (
                        ["rdt", "search", query, "--limit", limit_text],
                        "agent-reach:reddit:rdt-cli",
                    )
                ]
            return [
                (
                    ["opencli", "reddit", "search", query, "-f", "yaml"],
                    "agent-reach:reddit:opencli",
                )
            ]
        if channel == "xiaohongshu":
            if "xiaohongshu-mcp" in active_backend:
                return [
                    (
                        [
                            "mcporter",
                            "call",
                            "xiaohongshu.search_feeds",
                            f"keyword={query}",
                            "--timeout",
                            "120000",
                        ],
                        "agent-reach:xiaohongshu:mcp",
                    )
                ]
            if "xhs-cli" in active_backend or active_backend == "xhs":
                return [(["xhs", "search", query], "agent-reach:xiaohongshu:xhs-cli")]
            return [
                (
                    ["opencli", "xiaohongshu", "search", query, "-f", "yaml"],
                    "agent-reach:xiaohongshu:opencli",
                )
            ]
        if channel in {"facebook", "instagram"}:
            return [
                (
                    ["opencli", channel, "search", query, "-f", "yaml"],
                    f"agent-reach:{channel}:opencli",
                )
            ]
        if channel == "linkedin":
            # Agent Reach currently exposes LinkedIn primarily through MCP or
            # Jina Reader for known pages.  For query-style retrieval, route
            # through Agent Reach's Exa search while preserving site scope.
            return self._commands_for_channel("exa_search", f"site:linkedin.com {query}", limit, doctor)
        return []

    @staticmethod
    def _run_subprocess(command: list[str], timeout_seconds: float) -> str:
        import shutil

        # Windows: CreateProcess cannot execute npm-style .cmd/.bat shims
        # directly (mcporter installed via `npm install -g` ships only
        # mcporter.cmd / mcporter.ps1).  Resolve the command and delegate
        # script shims to cmd.exe so the subprocess actually runs.
        if os.name == "nt" and command:
            resolved = shutil.which(command[0])
            if resolved and resolved.lower().endswith((".cmd", ".bat")):
                command = ["cmd.exe", "/c", *command]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            output = (completed.stderr or completed.stdout or "").strip()
            raise SearchProviderError(output or f"Command exited with status {completed.returncode}")
        return completed.stdout


class BrowserSearchClient:
    def __init__(
        self,
        engines: list[SearchClient],
        enrich_content: bool = False,
        content_transport: SearchTransport | None = None,
        engine_timeout_seconds: float = 8.0,
        content_timeout_seconds: float = 3.0,
        max_enriched_results: int = 2,
        max_supplemental_results: int = 3,
        scoped_search_clients: Mapping[str, SearchClient] | None = None,
        baidu_only_domains: Iterable[str] | None = None,
    ):
        self.engines = engines
        self.engine_names = [_browser_engine_name(engine) for engine in engines]
        self.enrich_content = enrich_content
        self.content_transport = content_transport or _urllib_get
        self.engine_timeout_seconds = engine_timeout_seconds
        self.content_timeout_seconds = content_timeout_seconds
        self.max_enriched_results = max_enriched_results
        self.max_supplemental_results = max_supplemental_results
        self.scoped_search_clients = {
            _normalize_domain(domain): client for domain, client in (scoped_search_clients or {}).items()
        }
        self.baidu_only_domains = frozenset(
            _normalize_domain(domain) for domain in (baidu_only_domains or ()) if _normalize_domain(domain)
        )

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self.search_with_events(query, limit=limit).results

    def search_with_events(self, query: str, limit: int = 5) -> SearchOutcome:
        provider_events: list[ProviderTraceEvent] = []
        scoped_domains = _site_domains(query)
        supplemental_results: list[SearchResult] = []
        if self._requires_baidu_only(scoped_domains):
            provider_events.append(
                _provider_event(
                    provider="browser-supplemental",
                    query=query,
                    status="skipped",
                    reason="baidu_only_scoped_query",
                )
            )
        else:
            supplemental_start = time.monotonic()
            supplemental_results = self._supplemental_results(query, limit)
            provider_events.append(
                _provider_event(
                    provider="browser-supplemental",
                    query=query,
                    status="success" if supplemental_results else "empty",
                    result_count=len(supplemental_results),
                    elapsed_ms=_elapsed_ms(supplemental_start),
                )
            )
        merged = self._filter_scoped_results(supplemental_results, scoped_domains)
        scoped_results, scoped_events = self._scoped_results_with_events(query, scoped_domains, limit)
        provider_events.extend(scoped_events)
        merged.extend(scoped_results)
        merged = self._dedupe_results(merged, limit)
        seen = {_dedupe_key(result.url) for result in merged}
        if len(merged) >= limit:
            return SearchOutcome(merged[:limit], provider_events)

        engine_results, engine_events = self._search_engines_with_events(query, limit)
        provider_events.extend(engine_events)

        max_length = max((len(results) for results in engine_results), default=0)
        for index in range(max_length):
            for results in engine_results:
                if index >= len(results):
                    continue
                result = results[index]
                if not _matches_scoped_domains(result.url, scoped_domains):
                    continue
                key = _dedupe_key(result.url)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(result)
                if len(merged) >= limit:
                    return SearchOutcome(self._enrich_results(query, merged), provider_events)
        return SearchOutcome(self._enrich_results(query, merged), provider_events)

    def _scoped_results(self, query: str, scoped_domains: tuple[str, ...], limit: int) -> list[SearchResult]:
        results, _ = self._scoped_results_with_events(query, scoped_domains, limit)
        return results

    def _scoped_results_with_events(
        self,
        query: str,
        scoped_domains: tuple[str, ...],
        limit: int,
    ) -> tuple[list[SearchResult], list[ProviderTraceEvent]]:
        results: list[SearchResult] = []
        provider_events: list[ProviderTraceEvent] = []
        for domain in scoped_domains:
            client = self.scoped_search_clients.get(domain)
            if client is None:
                provider_events.append(
                    _provider_event(
                        provider=f"scoped:{domain}",
                        query=query,
                        status="skipped",
                        reason="no_explicit_scoped_client",
                    )
                )
                continue
            started = time.monotonic()
            try:
                candidates = client.search(query, limit=limit)
            except Exception as exc:
                provider_events.append(
                    _provider_event(
                        provider=_client_provider_name(client),
                        query=query,
                        status="error",
                        reason="scoped_client_error",
                        error=_safe_provider_error(exc),
                        elapsed_ms=_elapsed_ms(started),
                    )
                )
                continue
            scoped = [result for result in candidates if _matches_scoped_domains(result.url, (domain,))]
            provider_events.append(
                _provider_event(
                    provider=_client_provider_name(client),
                    query=query,
                    status="success" if scoped else "empty",
                    result_count=len(scoped),
                    reason=None if scoped else "no_on_domain_results",
                    elapsed_ms=_elapsed_ms(started),
                )
            )
            results.extend(scoped)
            if len(results) >= limit:
                break
        return results[:limit], provider_events

    @staticmethod
    def _filter_scoped_results(results: list[SearchResult], scoped_domains: tuple[str, ...]) -> list[SearchResult]:
        return [result for result in results if _matches_scoped_domains(result.url, scoped_domains)]

    @staticmethod
    def _dedupe_results(results: list[SearchResult], limit: int) -> list[SearchResult]:
        unique: list[SearchResult] = []
        seen: set[str] = set()
        for result in results:
            key = _dedupe_key(result.url)
            if key in seen:
                continue
            seen.add(key)
            unique.append(result)
            if len(unique) >= limit:
                break
        return unique

    def _supplemental_results(self, query: str, limit: int) -> list[SearchResult]:
        candidates = self._supplemental_candidates(query)
        if not candidates:
            return []

        results: list[SearchResult] = []
        for title, url, provider in candidates[: min(limit, self.max_supplemental_results)]:
            result = self._supplemental_result_from_url(query, title, url, provider)
            if result is not None:
                results.append(result)
        return results

    def _supplemental_candidates(self, query: str) -> list[tuple[str, str, str]]:
        lowered = query.lower()
        candidates: list[tuple[str, str, str]] = []
        if "cxl" in lowered or "compute express link" in lowered:
            candidates.append(
                (
                    "Compute Express Link Consortium About CXL official page",
                    "https://www.computeexpresslink.org/about-cxl/",
                    "direct-official",
                )
            )
        if "dgx spark" in lowered or "gb10" in lowered:
            candidates.append(
                (
                    "NVIDIA DGX Spark official specifications",
                    "https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
                    "direct-official",
                )
            )
        if self._is_h100_inference_query(lowered):
            candidates.append(
                (
                    "NVIDIA H100 Tensor Core GPU official page",
                    "https://www.nvidia.com/en-us/data-center/h100/",
                    "direct-official",
                )
            )
        if "connectx-7" in lowered or "connectx 7" in lowered or "cx7" in lowered:
            candidates.append(
                (
                    "NVIDIA ConnectX-7 official product page",
                    "https://www.nvidia.com/en-us/networking/ethernet/connectx-7/",
                    "direct-official",
                )
            )
        for version in self._deepseek_versions(lowered):
            candidates.extend(self._deepseek_source_candidates(version, lowered))
        if "deepseek" in lowered and not self._deepseek_versions(lowered) and self._mentions_deepseek_source_family(lowered):
            candidates.extend(self._generic_deepseek_source_candidates(lowered))
        if self._is_distributed_inference_query(lowered):
            candidates.extend(self._distributed_inference_source_candidates())
        if self._is_physical_ai_query(lowered):
            candidates.extend(self._physical_ai_source_candidates(lowered))
        return candidates

    def _deepseek_versions(self, lowered_query: str) -> list[str]:
        versions: list[str] = []
        for match in re.finditer(r"deepseek[-\s]?v(\d+)", lowered_query):
            version = f"V{match.group(1)}"
            if version not in versions:
                versions.append(version)
        return versions

    def _deepseek_source_candidates(self, version: str, lowered_query: str) -> list[tuple[str, str, str]]:
        model_name = f"DeepSeek-{version}"
        candidates = [
            (
                f"{model_name} on ModelScope",
                f"https://www.modelscope.cn/models/deepseek-ai/{model_name}",
                "direct-model-hub",
            ),
            (
                f"{model_name} on Hugging Face",
                f"https://huggingface.co/deepseek-ai/{model_name}",
                "direct-model-hub",
            ),
            (
                f"{model_name} GitHub repository",
                f"https://github.com/deepseek-ai/{model_name}",
                "direct-repository",
            ),
        ]
        priority_terms = {
            "github": "github.com",
            "repository": "github.com",
            "repo": "github.com",
            "hugging face": "huggingface.co",
            "huggingface": "huggingface.co",
            "modelscope": "modelscope.cn",
            "魔搭": "modelscope.cn",
            "魔塔": "modelscope.cn",
        }
        priority_domains: list[str] = []
        for term, domain in priority_terms.items():
            if term in lowered_query and domain not in priority_domains:
                priority_domains.append(domain)
        if not priority_domains:
            return candidates

        def candidate_priority(candidate: tuple[str, str, str]) -> int:
            url = candidate[1]
            for index, domain in enumerate(priority_domains):
                if domain in url:
                    return index
            return len(priority_domains)

        return sorted(candidates, key=candidate_priority)

    def _mentions_deepseek_source_family(self, lowered_query: str) -> bool:
        return any(
            term in lowered_query
            for term in (
                "deploy",
                "deployment",
                "部署",
                "模型参数",
                "参数",
                "模型卡",
                "model parameter",
                "model parameters",
                "parameter",
                "parameters",
                "model card",
                "model cards",
                "modelscope",
                "魔搭",
                "魔塔",
                "hugging face",
                "huggingface",
                "github",
                "source",
                "official",
                "官方",
                "开源模型社区",
                "模型社区",
            )
        )

    def _generic_deepseek_source_candidates(self, lowered_query: str) -> list[tuple[str, str, str]]:
        candidates = [
            (
                "DeepSeek organization on ModelScope",
                "https://www.modelscope.cn/organization/deepseek-ai",
                "direct-model-hub",
            ),
            (
                "deepseek-ai organization on Hugging Face",
                "https://huggingface.co/deepseek-ai",
                "direct-model-hub",
            ),
            (
                "deepseek-ai organization on GitHub",
                "https://github.com/deepseek-ai",
                "direct-repository",
            ),
        ]
        if "nvidia" in lowered_query or "gb10" in lowered_query or "dgx" in lowered_query:
            candidates.append(
                (
                    "NVIDIA DGX Spark official specifications",
                    "https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
                    "direct-official",
                )
            )
        return candidates

    def _is_physical_ai_query(self, lowered_query: str) -> bool:
        return any(
            term in lowered_query
            for term in (
                "physical ai",
                "embodied ai",
                "embodied physical",
                "world model",
                "physics foundation",
                "gr00t",
                "genie",
                "具身",
                "物理大模型",
                "世界模型",
            )
        )

    def _is_h100_inference_query(self, lowered_query: str) -> bool:
        if "h100" not in lowered_query:
            return False
        return any(
            term in lowered_query
            for term in (
                "tok/s",
                "tokens/s",
                "token/s",
                "tokens per second",
                "tps",
                "llm",
                "inference",
                "decode",
                "throughput",
                "memory bandwidth",
                "bandwidth",
                "hbm",
                "hbm3",
                "kv cache",
                "tensor core",
                "transformer engine",
                "带宽",
                "推理",
                "吞吐",
            )
        )

    def _is_distributed_inference_query(self, lowered_query: str) -> bool:
        has_distributed_signal = any(
            term in lowered_query
            for term in (
                "distributed",
                "multi-node",
                "multinode",
                "model parallel",
                "tensor parallel",
                "pipeline parallel",
                "expert parallel",
                "all-reduce",
                "allreduce",
                "interconnect",
                "分布式",
                "多节点",
                "模型并行",
                "张量并行",
                "流水线并行",
                "专家并行",
                "节点",
                "互联",
            )
        )
        has_llm_inference_signal = any(
            term in lowered_query
            for term in (
                "llm",
                "large language model",
                "large model",
                "inference",
                "serving",
                "decode",
                "kv cache",
                "vllm",
                "deepspeed",
                "megatron",
                "大模型",
                "模型",
                "推理",
                "服务",
                "解码",
                "缓存",
            )
        )
        return has_distributed_signal and has_llm_inference_signal

    def _distributed_inference_source_candidates(self) -> list[tuple[str, str, str]]:
        return [
            (
                "vLLM official parallelism and scaling documentation",
                "https://docs.vllm.ai/en/stable/serving/parallelism_scaling/",
                "direct-official",
            ),
            (
                "NVIDIA Megatron Bridge official parallelisms documentation",
                "https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html",
                "direct-official",
            ),
            (
                "DeepSpeed inference official documentation",
                "https://deepspeed.readthedocs.io/en/latest/inference-init.html",
                "direct-official",
            ),
        ]

    def _physical_ai_source_candidates(self, lowered_query: str) -> list[tuple[str, str, str]]:
        candidates = [
            (
                "NVIDIA Cosmos official world foundation models page",
                "https://www.nvidia.com/en-us/ai-data-science/cosmos/",
                "direct-official",
            ),
            (
                "Google DeepMind Genie 2 official world model article",
                "https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/",
                "direct-official",
            ),
            (
                "arXiv search for world model embodied AI papers",
                "https://arxiv.org/search/?query=world+model+embodied+AI&searchtype=all&source=header",
                "direct-literature",
            ),
        ]
        if "gr00t" in lowered_query or "robot" in lowered_query or "具身" in lowered_query:
            candidates.insert(
                1,
                (
                    "NVIDIA Isaac GR00T official page",
                    "https://developer.nvidia.com/isaac/gr00t",
                    "direct-official",
                ),
            )
        return candidates

    def _supplemental_result_from_url(
        self,
        query: str,
        title: str,
        url: str,
        provider: str,
    ) -> SearchResult | None:
        try:
            html_body = self.content_transport(url, _browser_page_headers())
        except SearchProviderError:
            return SearchResult(
                title=title,
                url=url,
                snippet=f"Direct source selected for query: {query}. Page content fetch was unavailable.",
                provider=provider,
                checked_at=datetime.now(UTC).isoformat(),
            )
        except Exception:
            return None
        excerpt = _extract_relevant_page_excerpt(html_body, query, max_chars=1600)
        if not excerpt:
            excerpt = f"Direct source selected for query: {query}."
        return SearchResult(
            title=title,
            url=url,
            snippet=excerpt,
            provider=provider,
            checked_at=datetime.now(UTC).isoformat(),
        )

    def _search_engines(self, query: str, limit: int) -> list[list[SearchResult]]:
        results, _ = self._search_engines_with_events(query, limit)
        return results

    def _search_engines_with_events(
        self,
        query: str,
        limit: int,
    ) -> tuple[list[list[SearchResult]], list[ProviderTraceEvent]]:
        scoped_domains = _site_domains(query)
        engines = self.engines
        if self._requires_baidu_only(scoped_domains):
            engines = [engine for engine in engines if isinstance(engine, BaiduBrowserSearchClient)]
        if not engines:
            return [], [
                _provider_event(
                    provider="browser-engines",
                    query=query,
                    status="skipped",
                    reason="no_enabled_engine_for_query",
                )
            ]

        executor = ThreadPoolExecutor(max_workers=len(engines))
        started_at = {engine: time.monotonic() for engine in engines}
        futures = [(engine, executor.submit(engine.search, query, limit)) for engine in engines]
        wait([future for _, future in futures], timeout=self.engine_timeout_seconds)

        engine_results: list[list[SearchResult]] = []
        provider_events: list[ProviderTraceEvent] = []
        for engine, future in futures:
            provider = _client_provider_name(engine)
            started = started_at[engine]
            if not future.done():
                future.cancel()
                engine_results.append([])
                provider_events.append(
                    _provider_event(
                        provider=provider,
                        query=query,
                        status="timeout",
                        reason=f"exceeded_{self.engine_timeout_seconds:g}s_budget",
                        elapsed_ms=_elapsed_ms(started),
                    )
                )
                continue
            try:
                results = future.result()
            except Exception as exc:
                engine_results.append([])
                provider_events.append(
                    _provider_event(
                        provider=provider,
                        query=query,
                        status="error",
                        error=_safe_provider_error(exc),
                        elapsed_ms=_elapsed_ms(started),
                    )
                )
                continue
            engine_results.append(results)
            provider_events.append(
                _provider_event(
                    provider=provider,
                    query=query,
                    status="success" if results else "empty",
                    result_count=len(results),
                    elapsed_ms=_elapsed_ms(started),
                )
            )
        executor.shutdown(wait=False, cancel_futures=True)

        if scoped_domains and not self._has_scoped_engine_results(engine_results, scoped_domains):
            fallback_engines = self.engines
            if self._requires_baidu_only(scoped_domains):
                fallback_engines = [
                    engine for engine in self.engines if not isinstance(engine, BaiduBrowserSearchClient)
                ]
            engine_results.extend(
                self._scoped_query_fallback_results(
                    fallback_engines,
                    query,
                    scoped_domains,
                    limit,
                )
            )
        return engine_results, provider_events

    @staticmethod
    def _has_scoped_engine_results(
        engine_results: list[list[SearchResult]],
        scoped_domains: tuple[str, ...],
    ) -> bool:
        return any(
            _matches_scoped_domains(result.url, scoped_domains)
            for results in engine_results
            for result in results
        )

    def _search_with_engines(
        self,
        engines: list[SearchClient],
        query: str,
        limit: int,
    ) -> list[list[SearchResult]]:
        if not engines:
            return []
        executor = ThreadPoolExecutor(max_workers=len(engines))
        futures = [executor.submit(engine.search, query, limit) for engine in engines]
        wait(futures, timeout=self.engine_timeout_seconds)
        engine_results: list[list[SearchResult]] = []
        for future in futures:
            if not future.done():
                future.cancel()
                engine_results.append([])
                continue
            try:
                engine_results.append(future.result())
            except Exception:
                engine_results.append([])
        executor.shutdown(wait=False, cancel_futures=True)
        return engine_results

    def _scoped_query_fallback_results(
        self,
        engines: list[SearchClient],
        query: str,
        scoped_domains: tuple[str, ...],
        limit: int,
    ) -> list[list[SearchResult]]:
        fallback_results: list[list[SearchResult]] = []
        for fallback_query in _scoped_query_fallbacks(query, scoped_domains):
            engine_results = self._search_with_engines(engines, fallback_query, limit)
            fallback_results.extend(engine_results)
            if self._has_scoped_engine_results(engine_results, scoped_domains):
                break
        return fallback_results

    def _enrich_results(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not self.enrich_content or self._requires_baidu_only(_site_domains(query)):
            return results

        enrich_limit = max(0, min(self.max_enriched_results, len(results)))
        if enrich_limit == 0:
            return results

        executor = ThreadPoolExecutor(max_workers=enrich_limit)
        futures = [executor.submit(self._enrich_result, query, result) for result in results[:enrich_limit]]
        wait(futures, timeout=self.content_timeout_seconds)

        enriched_results: list[SearchResult] = []
        for index, future in enumerate(futures):
            if not future.done():
                future.cancel()
                enriched_results.append(results[index])
                continue
            try:
                enriched_results.append(future.result())
            except Exception:
                enriched_results.append(results[index])
        executor.shutdown(wait=False, cancel_futures=True)
        return enriched_results + results[enrich_limit:]

    def _requires_baidu_only(self, scoped_domains: tuple[str, ...]) -> bool:
        return bool(scoped_domains) and all(domain in self.baidu_only_domains for domain in scoped_domains)


    def _enrich_result(self, query: str, result: SearchResult) -> SearchResult:
        if not result.url.startswith(("http://", "https://")):
            return result
        try:
            html_body = self.content_transport(
                result.url,
                _browser_page_headers(),
            )
        except SearchProviderError:
            return result

        excerpt = _extract_relevant_page_excerpt(html_body, query)
        if not excerpt:
            return result
        snippet = _clean_text(f"{result.snippet} Page excerpt: {excerpt}")
        return result.model_copy(update={"snippet": snippet[:1600]})


class BingBrowserSearchClient:
    def __init__(
        self,
        base_url: str = "https://www.bing.com/search",
        market: str = "zh-CN",
        transport: SearchTransport | None = None,
    ):
        self.base_url = base_url
        self.market = market
        self.transport = transport or _urllib_get

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        market = _market_for_bing_query(query, self.market)
        url = self.base_url + "?" + urllib.parse.urlencode({"q": query, "mkt": market})
        html_body = self.transport(
            url,
            {
                "User-Agent": "Mozilla/5.0 search-assistant/0.1",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": _accept_language_for_market(market),
            },
        )
        return self.parse_results(html_body, checked_at=datetime.now(UTC).isoformat())[:limit]

    @staticmethod
    def parse_results(html_body: str, checked_at: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        for raw_item in re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', html_body, flags=re.DOTALL):
            title_match = re.search(r"<h2[^>]*>.*?<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>.*?</h2>", raw_item, flags=re.DOTALL)
            if not title_match:
                continue
            url = _decode_bing_url(title_match.group(1))
            title = _clean_text(_strip_tags(title_match.group(2)))
            snippet_match = re.search(r'<div class="b_caption"[^>]*>.*?<p[^>]*>(.*?)</p>', raw_item, flags=re.DOTALL)
            snippet = _clean_text(_strip_tags(snippet_match.group(1))) if snippet_match else ""
            if title and url:
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        provider="browser-bing",
                        checked_at=checked_at,
                    )
                )
        return results


class BaiduBrowserSearchClient:
    def __init__(
        self,
        base_url: str = "https://www.baidu.com/baidu",
        transport: SearchTransport | None = None,
        fallback_base_urls: Iterable[str] | None = None,
        request_interval_seconds: float = 0.0,
        challenge_cooldown_seconds: float = 0.0,
    ):
        self.base_url = base_url
        self.transport = transport or _urllib_get
        self.fallback_base_urls = tuple(
            fallback_base_urls
            if fallback_base_urls is not None
            else ("https://www.baidu.com/baidu", "https://m.baidu.com/s", "https://www.baidu.com/s")
        )
        self.request_interval_seconds = max(0.0, request_interval_seconds)
        self.challenge_cooldown_seconds = max(0.0, challenge_cooldown_seconds)
        self._request_lock = Lock()
        self._next_request_at = 0.0
        self._blocked_until = 0.0

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        last_error: SearchProviderError | None = None
        for base_url in _unique_urls((self.base_url, *self.fallback_base_urls)):
            url = _baidu_search_url(base_url, query)
            try:
                html_body = self._rate_limited_get(url, _baidu_page_headers(base_url))
            except SearchProviderError as exc:
                last_error = exc
                continue
            if _is_baidu_challenge_page(html_body):
                last_error = SearchProviderError("Baidu public search returned a verification page")
                continue
            results = self.parse_results(html_body, checked_at=datetime.now(UTC).isoformat())
            if results:
                return results[:limit]
        if last_error is not None:
            if "verification page" in str(last_error):
                self._mark_challenge_cooldown()
            raise last_error
        return []

    def _rate_limited_get(self, url: str, headers: dict[str, str]) -> str:
        with self._request_lock:
            if self._blocked_until > time.monotonic():
                raise SearchProviderError("Baidu public search is cooling down after a verification page")
            if self.request_interval_seconds <= 0:
                return self.transport(url, headers)
            wait_seconds = self._next_request_at - time.monotonic()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            try:
                return self.transport(url, headers)
            finally:
                self._next_request_at = time.monotonic() + self.request_interval_seconds

    def _mark_challenge_cooldown(self) -> None:
        if self.challenge_cooldown_seconds <= 0:
            return
        with self._request_lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + self.challenge_cooldown_seconds)

    @staticmethod
    def parse_results(html_body: str, checked_at: str) -> list[SearchResult]:
        results = _parse_baidu_desktop_results(html_body, checked_at)
        results.extend(_parse_baidu_mobile_results(html_body, checked_at))
        unique_results: list[SearchResult] = []
        seen: set[str] = set()
        for result in results:
            key = _dedupe_key(result.url)
            if key in seen:
                continue
            seen.add(key)
            unique_results.append(result)
        return unique_results


def _parse_baidu_desktop_results(html_body: str, checked_at: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    title_pattern = re.compile(
        r'<h3[^>]*class="[^"]*t[^"]*"[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?</h3>',
        flags=re.DOTALL,
    )
    matches = list(title_pattern.finditer(html_body))
    for index, title_match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(html_body)
        item_tail = html_body[title_match.end() : next_start]
        title = _clean_text(_strip_tags(title_match.group(2)))
        url = _baidu_original_result_url(html_body, title_match.start(), title_match.group(1))
        snippet_match = re.search(r'<div[^>]+class="[^"]*c-abstract[^"]*"[^>]*>(.*?)</div>', item_tail, flags=re.DOTALL)
        snippet = _clean_text(_strip_tags(snippet_match.group(1))) if snippet_match else ""
        if title and url and not _is_baidu_redirect_url(url):
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    provider="browser-baidu",
                    checked_at=checked_at,
                )
            )
    return results


def _parse_baidu_mobile_results(html_body: str, checked_at: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    result_pattern = re.compile(
        r'(?:dataUrl|data-url)(?:&quot;|["\'])\s*:\s*(?:&quot;|["\'])'
        r'(?P<url>.*?)(?:&quot;|["\']).{0,8000}?<h3[^>]*>(?P<title>.*?)</h3>',
        flags=re.DOTALL | re.IGNORECASE,
    )
    matches = list(result_pattern.finditer(html_body))
    for index, result_match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(html_body)
        item_tail = html_body[result_match.end() : next_start]
        title = _clean_text(_strip_tags(result_match.group("title")))
        url = html.unescape(result_match.group("url")).strip()
        snippet_match = re.search(
            r'(?:&quot;|["\'])text(?:&quot;|["\'])\s*:\s*(?:&quot;|["\'])(.*?)(?:&quot;|["\'])',
            item_tail,
            flags=re.DOTALL | re.IGNORECASE,
        )
        snippet = _clean_text(_strip_tags(html.unescape(snippet_match.group(1)))) if snippet_match else ""
        if title and url.startswith(("http://", "https://")) and not _is_baidu_result_url(url):
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    provider="browser-baidu",
                    checked_at=checked_at,
                )
            )
    return results


class GoogleBrowserSearchClient:
    def __init__(
        self,
        base_url: str = "https://www.google.com/search",
        language: str = "zh-CN",
        transport: SearchTransport | None = None,
    ):
        self.base_url = base_url
        self.language = language
        self.transport = transport or _urllib_get

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        url = self.base_url + "?" + urllib.parse.urlencode({"q": query, "hl": self.language})
        html_body = self.transport(
            url,
            {
                "User-Agent": "Mozilla/5.0 search-assistant/0.1",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": f"{self.language},zh;q=0.9,en;q=0.6",
            },
        )
        return self.parse_results(html_body, checked_at=datetime.now(UTC).isoformat())[:limit]

    @staticmethod
    def parse_results(html_body: str, checked_at: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        title_pattern = re.compile(
            r'<a[^>]*href="([^"]+)"[^>]*>.*?<h3[^>]*>(.*?)</h3>.*?</a>',
            flags=re.DOTALL,
        )
        matches = list(title_pattern.finditer(html_body))
        for index, title_match in enumerate(matches):
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(html_body)
            item_tail = html_body[title_match.end() : next_start]
            url = _decode_google_url(title_match.group(1))
            title = _clean_text(_strip_tags(title_match.group(2)))
            snippet_match = re.search(r'<div[^>]+class="[^"]*(?:VwiC3b|IsZvec)[^"]*"[^>]*>(.*?)</div>', item_tail, flags=re.DOTALL)
            snippet = _clean_text(_strip_tags(snippet_match.group(1))) if snippet_match else ""
            if title and url.startswith(("http://", "https://")):
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        provider="browser-google",
                        checked_at=checked_at,
                    )
                )
        return results


class BraveSearchClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.search.brave.com/res/v1/web/search",
        transport: SearchTransport | None = None,
    ):
        if not api_key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY is required for BraveSearchClient")
        self.api_key = api_key
        self.base_url = base_url
        self.transport = transport or _urllib_get

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        count = max(1, min(limit, 20))
        url = self.base_url + "?" + urllib.parse.urlencode({"q": query, "count": count})
        payload = self.transport(
            url,
            {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
        )
        return self.parse_results(payload, checked_at=datetime.now(UTC).isoformat())[:limit]

    @staticmethod
    def parse_results(payload: str, checked_at: str) -> list[SearchResult]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SearchProviderError("Brave search returned non-JSON response") from exc
        raw_results = data.get("web", {}).get("results", []) if isinstance(data, dict) else []
        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = _clean_text(str(item.get("title") or ""))
            url = str(item.get("url") or "").strip()
            snippet = _clean_text(str(item.get("description") or ""))
            if title and url:
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        provider="brave",
                        checked_at=checked_at,
                    )
                )
        return results


class SearxngSearchClient:
    """JSON search client for a self-hosted SearXNG instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/search",
        transport: SearchTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.transport = transport or _urllib_get

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        url = self.base_url + "?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "language": "auto"}
        )
        payload = self.transport(url, {"Accept": "application/json"})
        return self.parse_results(payload, checked_at=datetime.now(UTC).isoformat())[:limit]

    @staticmethod
    def parse_results(payload: str, checked_at: str) -> list[SearchResult]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SearchProviderError("SearXNG search returned non-JSON response") from exc
        raw_results = data.get("results", []) if isinstance(data, dict) else []
        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = _clean_text(str(item.get("title") or ""))
            url = str(item.get("url") or "").strip()
            snippet = _clean_text(str(item.get("content") or item.get("snippet") or ""))
            if title and url.startswith(("https://", "http://")):
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        provider="searxng",
                        checked_at=checked_at,
                    )
                )
        return results


class BilibiliPublicSearchClient:
    """No-login Bilibili video search for a route explicitly scoped to bilibili.com."""

    def __init__(
        self,
        base_url: str = "https://api.bilibili.com/x/web-interface/search/type",
        transport: SearchTransport | None = None,
        request_interval_seconds: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.transport = transport or _urllib_get
        self.request_interval_seconds = max(0.0, request_interval_seconds)
        self._request_lock = Lock()
        self._next_request_at = 0.0

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        keyword = _query_without_site_directives(query)
        if not keyword:
            return []
        url = self.base_url + "?" + urllib.parse.urlencode(
            {"search_type": "video", "keyword": keyword, "page": "1"}
        )
        last_error: SearchProviderError | None = None
        for headers in _bilibili_api_headers():
            try:
                payload = self._rate_limited_get(url, headers)
                return self.parse_results(payload, checked_at=datetime.now(UTC).isoformat())[:limit]
            except SearchProviderError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return []

    def _rate_limited_get(self, url: str, headers: dict[str, str]) -> str:
        if self.request_interval_seconds <= 0:
            return self.transport(url, headers)
        with self._request_lock:
            wait_seconds = self._next_request_at - time.monotonic()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            try:
                return self.transport(url, headers)
            finally:
                self._next_request_at = time.monotonic() + self.request_interval_seconds

    @staticmethod
    def parse_results(payload: str, checked_at: str) -> list[SearchResult]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SearchProviderError("Bilibili search returned non-JSON response") from exc
        if not isinstance(data, dict) or data.get("code") != 0:
            raise SearchProviderError(f"Bilibili search returned API code {data.get('code') if isinstance(data, dict) else 'unknown'}")

        raw_results = data.get("data", {}).get("result", [])
        if not isinstance(raw_results, list):
            return []

        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = _clean_text(_strip_tags(str(item.get("title") or "")))
            video_url = str(item.get("arcurl") or "").strip()
            if not video_url:
                bvid = str(item.get("bvid") or "").strip()
                if bvid:
                    video_url = f"https://www.bilibili.com/video/{bvid}"
            if video_url.startswith("http://"):
                video_url = "https://" + video_url.removeprefix("http://")
            snippet = _clean_text(_strip_tags(str(item.get("description") or "")))
            author = _clean_text(_strip_tags(str(item.get("author") or "")))
            if author:
                snippet = _clean_text(f"{snippet} 作者：{author}")
            if title and _matches_scoped_domains(video_url, ("bilibili.com",)):
                results.append(
                    SearchResult(
                        title=title,
                        url=video_url,
                        snippet=snippet,
                        provider="bilibili-public-api",
                        checked_at=checked_at,
                    )
                )
        return results


class ToutiaoPublicSearchClient:
    """No-login Toutiao public search page parser for routes explicitly scoped to toutiao.com."""

    def __init__(
        self,
        base_url: str = "https://so.toutiao.com/search",
        transport: SearchTransport | None = None,
        request_interval_seconds: float = 0.0,
    ):
        self.base_url = base_url
        self.transport = transport or _urllib_get
        self.request_interval_seconds = max(0.0, request_interval_seconds)
        self._request_lock = Lock()
        self._next_request_at = 0.0

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        keyword = _query_without_site_directives(query)
        if not keyword:
            return []
        url = self.base_url + "?" + urllib.parse.urlencode({"keyword": keyword})
        html_body = self._rate_limited_get(
            url,
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
            },
        )
        return self.parse_results(html_body, checked_at=datetime.now(UTC).isoformat())[:limit]

    def _rate_limited_get(self, url: str, headers: dict[str, str]) -> str:
        if self.request_interval_seconds <= 0:
            return self.transport(url, headers)
        with self._request_lock:
            wait_seconds = self._next_request_at - time.monotonic()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            try:
                return self.transport(url, headers)
            finally:
                self._next_request_at = time.monotonic() + self.request_interval_seconds

    @staticmethod
    def parse_results(html_body: str, checked_at: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        url_pattern = re.compile(
            r'(?:open_url|article_url)(?:&quot;|["\'])\s*:\s*(?:&quot;|["\'])(?P<url>.*?)(?:&quot;|["\'])',
            flags=re.DOTALL | re.IGNORECASE,
        )
        matches = list(url_pattern.finditer(html_body))
        for index, match in enumerate(matches):
            url = _decode_embedded_json_value(match.group("url"))
            url = html.unescape(url).replace("\\/", "/").strip()
            url = _canonical_toutiao_url(url)
            if not url:
                continue
            next_start = matches[index + 1].start() if index + 1 < len(matches) else min(len(html_body), match.end() + 6000)
            window = html_body[max(0, match.start() - 2500) : next_start]
            title = _extract_embedded_json_field(window, ("title", "display_title")) or url
            snippet = _extract_embedded_json_field(window, ("abstract", "summary", "hot_board_summary"))
            results.append(
                SearchResult(
                    title=_clean_text(_strip_tags(title)),
                    url=url,
                    snippet=_clean_text(_strip_tags(snippet)),
                    provider="toutiao-public-search",
                    checked_at=checked_at,
                )
            )
        return _dedupe_search_results(results)


class YouTubePublicSearchClient:
    """No-login YouTube search page parser for routes explicitly scoped to youtube.com."""

    def __init__(
        self,
        base_url: str = "https://www.youtube.com/results",
        transport: SearchTransport | None = None,
        request_interval_seconds: float = 0.0,
    ):
        self.base_url = base_url
        self.transport = transport or _urllib_get
        self.request_interval_seconds = max(0.0, request_interval_seconds)
        self._request_lock = Lock()
        self._next_request_at = 0.0

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        keyword = _query_without_site_directives(query)
        if not keyword:
            return []
        url = self.base_url + "?" + urllib.parse.urlencode({"search_query": keyword})
        html_body = self._rate_limited_get(
            url,
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9,zh;q=0.6",
            },
        )
        return self.parse_results(html_body, checked_at=datetime.now(UTC).isoformat())[:limit]

    def _rate_limited_get(self, url: str, headers: dict[str, str]) -> str:
        if self.request_interval_seconds <= 0:
            return self.transport(url, headers)
        with self._request_lock:
            wait_seconds = self._next_request_at - time.monotonic()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            try:
                return self.transport(url, headers)
            finally:
                self._next_request_at = time.monotonic() + self.request_interval_seconds

    @staticmethod
    def parse_results(html_body: str, checked_at: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        pattern = re.compile(
            r'"videoId":"(?P<video_id>[^"]+)".{0,5000}?"title":\{"runs":\[\{"text":"(?P<title>.*?)"\}',
            flags=re.DOTALL,
        )
        for match in pattern.finditer(html_body):
            video_id = match.group("video_id")
            title = _decode_embedded_json_value(match.group("title"))
            if not video_id or not title:
                continue
            results.append(
                SearchResult(
                    title=_clean_text(title),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    snippet="Public YouTube search result.",
                    provider="youtube-public-search",
                    checked_at=checked_at,
                )
            )
        return _dedupe_search_results(results)


class DuckDuckGoSearchClient:
    def __init__(self, transport: SearchTransport | None = None, timeout_seconds: float = 12.0):
        self.transport = transport or _urllib_get_with_timeout(timeout_seconds)

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        html_body = self.transport(
            url,
            {
                "User-Agent": "Mozilla/5.0 search-assistant/0.1",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        return self.parse_results(
            html_body,
            provider="duckduckgo",
            checked_at=datetime.now(UTC).isoformat(),
        )[:limit]

    @staticmethod
    def parse_results(html_body: str, provider: str, checked_at: str) -> list[SearchResult]:
        parser = _DuckDuckGoHtmlParser(provider=provider, checked_at=checked_at)
        parser.feed(html_body)
        return parser.results


class FakeSearchClient:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return []


def search_with_provider_events(client: SearchClient, query: str, limit: int = 5) -> SearchOutcome:
    traced_search = getattr(client, "search_with_events", None)
    if callable(traced_search):
        outcome = traced_search(query, limit=limit)
        return _screen_injection(outcome, client, query)

    provider = _client_provider_name(client)
    started = time.monotonic()
    try:
        results = client.search(query, limit=limit)
    except Exception as exc:
        return SearchOutcome(
            results=[],
            provider_events=[
                _provider_event(
                    provider=provider,
                    query=query,
                    status="error",
                    error=_safe_provider_error(exc),
                    elapsed_ms=_elapsed_ms(started),
                )
            ],
        )
    return _screen_injection(
        SearchOutcome(
            results=results,
            provider_events=[
                _provider_event(
                    provider=provider,
                    query=query,
                    status="success" if results else "empty",
                    result_count=len(results),
                    elapsed_ms=_elapsed_ms(started),
                )
            ],
        ),
        client,
        query,
    )


def _screen_injection(outcome: SearchOutcome, client: SearchClient, query: str) -> SearchOutcome:
    """Drop retrieved sources that carry prompt-injection patterns.

    Screened content is untrusted data, not instructions; matched sources are
    removed from what reaches the LLM context while a provider event keeps the
    audit trail.  Applies to every retrieval path (traced and plain).
    """
    if not outcome.results:
        return outcome
    from search_assistant.search.injection import strip_injected_sources

    clean, flagged = strip_injected_sources(outcome.results)
    if not flagged:
        return outcome
    provider_events = list(outcome.provider_events)
    provider_events.append(
        _provider_event(
            provider=_client_provider_name(client),
            query=query,
            status="filtered",
            result_count=len(flagged),
            reason="prompt_injection",
        )
    )
    return SearchOutcome(results=clean, provider_events=provider_events)


def search_client_from_settings(settings: Settings) -> SearchClient:
    provider = settings.search_provider.lower()
    if provider == "agent-reach":
        return _agent_reach_search_client_from_settings(settings)
    if provider == "browser":
        return _browser_search_client_from_settings(settings)
    if provider == "mcp":
        return _mcp_search_client_from_settings(settings)
    if provider == "hybrid":
        # MCP and browser first; Agent Reach last as a supplementary pass.
        # Agent Reach routes spawn a subprocess per query (e.g. mcporter for
        # Exa), so running them first starves the briefing search budget and
        # the browser engines never get to contribute.
        clients: list[SearchClient] = [
            _mcp_search_client_from_settings(settings),
            _browser_search_client_from_settings(settings),
        ]
        if settings.agent_reach_enabled:
            clients.append(_agent_reach_search_client_from_settings(settings, require_available=False))
        return CompositeSearchClient(
            clients,
            # Public MCP sources provide structured technical evidence, but a
            # small number of results is not enough to represent global and
            # Chinese web coverage. Merge the browser pass before ranking.
            primary_sufficient_results=None,
        )
    if provider == "brave":
        return BraveSearchClient(
            api_key=settings.brave_search_api_key,
            base_url=settings.brave_search_base_url,
        )
    if provider == "searxng":
        return SearxngSearchClient(
            base_url=settings.searxng_base_url,
            transport=_urllib_get_with_timeout(settings.searxng_timeout_seconds),
        )
    if provider == "duckduckgo":
        return DuckDuckGoSearchClient(timeout_seconds=settings.duckduckgo_timeout_seconds)
    if provider == "fake" and settings.allow_fake_runtime:
        return FakeSearchClient()
    if provider == "fake":
        raise RuntimeError("Fake search provider is disabled. Set SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME=true only for tests.")
    raise RuntimeError(f"Unsupported search provider: {settings.search_provider}")


def _agent_reach_search_client_from_settings(
    settings: Settings,
    require_available: bool = True,
) -> AgentReachSearchClient:
    return AgentReachSearchClient(
        command=settings.agent_reach_command,
        timeout_seconds=settings.agent_reach_timeout_seconds,
        doctor_cache_seconds=settings.agent_reach_doctor_cache_seconds,
        require_available=require_available,
    )


def _browser_search_client_from_settings(settings: Settings) -> BrowserSearchClient:
    return BrowserSearchClient(
        _browser_engines_from_settings(settings),
        enrich_content=True,
        content_transport=_urllib_get_with_timeout(settings.browser_content_timeout_seconds),
        engine_timeout_seconds=settings.browser_search_timeout_seconds,
        content_timeout_seconds=settings.browser_content_timeout_seconds,
        max_enriched_results=settings.browser_enrich_max_results,
        scoped_search_clients={
            "bilibili.com": BilibiliPublicSearchClient(
                base_url=settings.bilibili_search_base_url,
                transport=_urllib_get_with_timeout(settings.browser_search_timeout_seconds),
                request_interval_seconds=0.5,
            ),
            "toutiao.com": ToutiaoPublicSearchClient(
                transport=_urllib_get_with_timeout(settings.browser_search_timeout_seconds),
                request_interval_seconds=0.5,
            ),
            "youtube.com": YouTubePublicSearchClient(
                transport=_urllib_get_with_timeout(settings.browser_search_timeout_seconds),
                request_interval_seconds=0.5,
            ),
        },
        baidu_only_domains={
            "mp.weixin.qq.com",
            "weixin.qq.com",
            "toutiao.com",
            "xiaohongshu.com",
            "xhslink.com",
        },
    )


def _mcp_search_client_from_settings(settings: Settings) -> McpSearchClient:
    config_path = settings.mcp_search_config_path
    if config_path is None:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "public-sources.mcp.json"
    domestic_config_path = settings.domestic_rss_config_path
    if domestic_config_path is None:
        domestic_config_path = Path(__file__).resolve().parents[3] / "configs" / "domestic-rss.sources.json"
    return McpSearchClient(
        config_path=config_path,
        timeout_seconds=settings.mcp_search_timeout_seconds,
        server_environment={
            "RSSHUB_BASE_URL": settings.domestic_rss_base_url,
            "SEARCH_ASSISTANT_DOMESTIC_RSS_CONFIG": str(domestic_config_path.resolve()),
            "DOMESTIC_RSS_TIMEOUT_SECONDS": str(settings.domestic_rss_timeout_seconds),
        },
    )


def _browser_engines_from_settings(settings: Settings) -> list[SearchClient]:
    engines: list[SearchClient] = []
    transport = _urllib_get_with_timeout(settings.browser_search_timeout_seconds)
    for engine in settings.browser_search_engines:
        if engine == "bing":
            engines.append(
                BingBrowserSearchClient(
                    base_url=settings.browser_search_base_url,
                    market=settings.browser_search_market,
                    transport=transport,
                )
            )
        elif engine == "baidu":
            engines.append(
                BaiduBrowserSearchClient(
                    base_url=settings.baidu_search_base_url,
                    transport=transport,
                    request_interval_seconds=0.8,
                    challenge_cooldown_seconds=60.0,
                )
            )
        elif engine == "google":
            engines.append(
                GoogleBrowserSearchClient(
                    base_url=settings.google_search_base_url,
                    language=settings.browser_search_market,
                    transport=transport,
                )
            )
        elif engine == "duckduckgo":
            engines.append(DuckDuckGoSearchClient(transport=transport))
        else:
            raise RuntimeError(f"Unsupported browser search engine: {engine}")
    return engines


_AGENT_REACH_BROWSER_UA = "Mozilla/5.0 agent-reach/search-assistant"

_AGENT_REACH_DOMAIN_CHANNELS: dict[str, str] = {
    "github.com": "github",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "bilibili.com": "bilibili",
    "b23.tv": "bilibili",
    "reddit.com": "reddit",
    "x.com": "twitter",
    "twitter.com": "twitter",
    "xiaohongshu.com": "xiaohongshu",
    "xhslink.com": "xiaohongshu",
    "facebook.com": "facebook",
    "instagram.com": "instagram",
    "linkedin.com": "linkedin",
}


def _agent_reach_channels_for_domains(scoped_domains: tuple[str, ...]) -> list[str]:
    channels: list[str] = []
    for domain in scoped_domains:
        normalized = _normalize_domain(domain)
        for configured_domain, channel in _AGENT_REACH_DOMAIN_CHANNELS.items():
            if normalized == configured_domain or normalized.endswith(f".{configured_domain}"):
                _append_unique(channels, channel)
                break
    return channels


def _agent_reach_channel_available(doctor: Mapping[str, Any], channel: str) -> bool:
    raw_result = doctor.get(channel)
    if not isinstance(raw_result, dict):
        return False
    # Agent Reach may report warn when it deliberately avoids a live platform
    # read (for example Exa registered in mcporter or OpenCLI present but not
    # probed).  That is still a real Agent Reach route worth trying.
    return str(raw_result.get("status", "")).lower() in {"ok", "warn"}


def _agent_reach_active_backend(doctor: Mapping[str, Any], channel: str) -> str:
    raw_result = doctor.get(channel)
    if not isinstance(raw_result, dict):
        return ""
    return str(raw_result.get("active_backend") or "")


def _agent_reach_unavailable_message(query: str, doctor: Mapping[str, Any]) -> str:
    channels = _agent_reach_channels_for_domains(_site_domains(query)) or ["exa_search"]
    statuses: list[str] = []
    for channel in channels:
        result = doctor.get(channel)
        if isinstance(result, dict):
            status = result.get("status", "unknown")
            message = str(result.get("message") or "").splitlines()[0]
            statuses.append(f"{channel}={status}: {message}")
        else:
            statuses.append(f"{channel}=missing")
    return "Agent Reach has no usable route for this query. " + "; ".join(statuses)


def _agent_reach_bilibili_search_api_url(query: str) -> str:
    return (
        "https://api.bilibili.com/x/web-interface/search/all/v2?"
        + urllib.parse.urlencode({"keyword": query, "page": "1"})
    )


def _agent_reach_output_to_search_results(raw_output: str, provider: str, checked_at: str) -> list[SearchResult]:
    parsed = _json_loads_maybe(raw_output)
    results: list[SearchResult] = []
    if parsed is not None:
        results.extend(_agent_reach_json_to_search_results(parsed, provider, checked_at))
    if not results:
        results.extend(_agent_reach_text_to_search_results(raw_output, provider, checked_at))
    return _dedupe_search_results(results)


def _agent_reach_json_to_search_results(value: Any, provider: str, checked_at: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    if isinstance(value, list):
        for item in value:
            results.extend(_agent_reach_json_to_search_results(item, provider, checked_at))
        return results

    if isinstance(value, dict):
        title = _first_non_empty_string(
            value,
            (
                "title",
                "name",
                "fullName",
                "full_name",
                "repo",
                "repository",
                "author",
                "username",
            ),
        )
        url = _first_non_empty_string(
            value,
            (
                "url",
                "html_url",
                "htmlUrl",
                "link",
                "external_url",
                "externalUrl",
                "webpage_url",
                "webpageUrl",
                "arcurl",
                "short_link",
            ),
        )
        snippet = _first_non_empty_string(
            value,
            (
                "snippet",
                "description",
                "desc",
                "content",
                "text",
                "summary",
                "body",
                "dynamic",
            ),
        )
        if url and url.startswith(("http://", "https://")):
            results.append(
                SearchResult(
                    title=_clean_text(title or _title_from_url(url)),
                    url=html.unescape(url),
                    snippet=_clean_text(snippet),
                    provider=provider,
                    checked_at=checked_at,
                )
            )

        for raw_child in value.values():
            if isinstance(raw_child, str):
                child_json = _json_loads_maybe(raw_child)
                if child_json is not None:
                    results.extend(_agent_reach_json_to_search_results(child_json, provider, checked_at))
                else:
                    results.extend(_agent_reach_text_to_search_results(raw_child, provider, checked_at))
            elif isinstance(raw_child, (dict, list)):
                results.extend(_agent_reach_json_to_search_results(raw_child, provider, checked_at))
        return results

    if isinstance(value, str):
        nested = _json_loads_maybe(value)
        if nested is not None:
            return _agent_reach_json_to_search_results(nested, provider, checked_at)
        return _agent_reach_text_to_search_results(value, provider, checked_at)

    return results


def _agent_reach_text_to_search_results(raw_output: str, provider: str, checked_at: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    for raw_line in raw_output.splitlines():
        line = _clean_text(raw_line)
        if not line:
            continue
        for url in _urls_from_text(line):
            title = _title_from_text_line(line, url)
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=line,
                    provider=provider,
                    checked_at=checked_at,
                )
            )
        for bvid in re.findall(r"\bBV[0-9A-Za-z]{8,}\b", line):
            url = f"https://www.bilibili.com/video/{bvid}"
            results.append(
                SearchResult(
                    title=_title_from_text_line(line, bvid),
                    url=url,
                    snippet=line,
                    provider=provider,
                    checked_at=checked_at,
                )
            )
    return results


def _json_loads_maybe(value: str) -> Any | None:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _first_non_empty_string(values: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int | float):
            return str(value)
    return ""


def _urls_from_text(value: str) -> list[str]:
    urls: list[str] = []
    for match in re.finditer(r"https?://[^\s\]\)\"'<>，。；、]+", value):
        url = html.unescape(match.group(0)).rstrip(".,;:!?)】》")
        if url.startswith(("http://", "https://")):
            urls.append(url)
    return urls


def _title_from_text_line(line: str, marker: str) -> str:
    before_marker = line.split(marker, 1)[0].strip(" -|:：[]()")
    if before_marker:
        return _clean_text(before_marker)[-120:]
    if marker.startswith("http"):
        return _title_from_url(marker)
    return marker


def _title_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = _normalize_domain(parsed.hostname or "source")
    path = urllib.parse.unquote(parsed.path.strip("/").split("/")[-1] if parsed.path else "")
    return _clean_text(path or host)
def _provider_event(
    *,
    provider: str,
    query: str,
    status: str,
    result_count: int = 0,
    reason: str | None = None,
    error: str | None = None,
    elapsed_ms: float | None = None,
) -> ProviderTraceEvent:
    return ProviderTraceEvent(
        provider=provider,
        query=query,
        status=status,  # type: ignore[arg-type]
        result_count=result_count,
        reason=reason,
        error=error,
        elapsed_ms=elapsed_ms,
        checked_at=datetime.now(UTC).isoformat(),
    )


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 3)


def _safe_provider_error(exc: Exception, max_chars: int = 240) -> str:
    text = f"{type(exc).__name__}: {exc}"
    for marker in ("api_key", "token", "authorization", "secret", "cookie", "key="):
        text = re.sub(marker, "[redacted]", text, flags=re.IGNORECASE)
    return text[:max_chars]


def _client_provider_name(client: SearchClient) -> str:
    if isinstance(client, CompositeSearchClient):
        return "composite"
    if isinstance(client, BrowserSearchClient):
        return "browser"
    if isinstance(client, McpSearchClient):
        return "mcp"
    return _browser_engine_name(client)


def _browser_engine_name(engine: SearchClient) -> str:
    if isinstance(engine, BingBrowserSearchClient):
        return "bing"
    if isinstance(engine, BaiduBrowserSearchClient):
        return "baidu"
    if isinstance(engine, GoogleBrowserSearchClient):
        return "google"
    if isinstance(engine, DuckDuckGoSearchClient):
        return "duckduckgo"
    if isinstance(engine, BraveSearchClient):
        return "brave"
    if isinstance(engine, SearxngSearchClient):
        return "searxng"
    if isinstance(engine, BilibiliPublicSearchClient):
        return "bilibili-public-api"
    if isinstance(engine, FakeSearchClient):
        return "fake"
    return type(engine).__name__


class _DuckDuckGoHtmlParser(HTMLParser):
    def __init__(self, provider: str, checked_at: str):
        super().__init__(convert_charrefs=True)
        self.provider = provider
        self.checked_at = checked_at
        self.results: list[SearchResult] = []
        self._link_href: str | None = None
        self._link_parts: list[str] = []
        self._snippet_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {name: value or "" for name, value in attrs}
        classes = set(attrs_dict.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._link_href = attrs_dict.get("href", "")
            self._link_parts = []
        if "result__snippet" in classes:
            self._snippet_parts = []

    def handle_data(self, data: str) -> None:
        if self._link_href is not None:
            self._link_parts.append(data)
        if self._snippet_parts is not None:
            self._snippet_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._link_href is not None:
            title = _clean_text(" ".join(self._link_parts))
            url = _decode_duckduckgo_url(self._link_href)
            if title and url:
                self.results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet="",
                        provider=self.provider,
                        checked_at=self.checked_at,
                    )
                )
            self._link_href = None
            self._link_parts = []
        if self._snippet_parts is not None and tag in {"a", "div"}:
            snippet = _clean_text(" ".join(self._snippet_parts))
            if snippet and self.results:
                last = self.results[-1]
                if not last.snippet:
                    last.snippet = snippet
            self._snippet_parts = None


def _clean_text(value: str) -> str:
    return " ".join(html.unescape(value).split())


def _market_for_bing_query(query: str, configured_market: str) -> str:
    if configured_market.lower() == "zh-cn" and re.search(r"[A-Za-z]", query) and not re.search(r"[\u4e00-\u9fff]", query):
        return "en-US"
    return configured_market


def _accept_language_for_market(market: str) -> str:
    if market.lower().startswith("en"):
        return f"{market},en;q=0.9"
    return f"{market},zh;q=0.9,en;q=0.6"


def _browser_page_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 search-assistant/0.1",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9,zh;q=0.6",
    }


def _scoped_query_fallbacks(query: str, scoped_domains: tuple[str, ...]) -> list[str]:
    clean_query = _query_without_site_directives(query)
    if not clean_query or not scoped_domains:
        return []

    fallbacks: list[str] = []
    for domain in scoped_domains:
        _append_unique(fallbacks, f"{clean_query} {domain}")
    return [candidate for candidate in fallbacks if candidate != query]


def _append_unique(values: list[str], value: str) -> None:
    normalized = " ".join(value.split())
    if normalized and normalized.lower() not in {item.lower() for item in values}:
        values.append(normalized)


def _unique_urls(urls: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = url.strip().rstrip("/")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _baidu_search_url(base_url: str, query: str) -> str:
    host = _normalize_domain(urllib.parse.urlparse(base_url).hostname or "")
    parameters = {"word": query} if host == "m.baidu.com" else {"wd": query, "ie": "utf-8"}
    return base_url.rstrip("/") + "?" + urllib.parse.urlencode(parameters)


def _baidu_page_headers(base_url: str) -> dict[str, str]:
    host = _normalize_domain(urllib.parse.urlparse(base_url).hostname or "")
    if host == "m.baidu.com":
        return {
            "User-Agent": (
                "Mozilla/5.0 (Linux; Android 14; Pixel 7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Mobile Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
        }
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
    }


def _bilibili_api_headers() -> list[dict[str, str]]:
    desktop_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
        "Origin": "https://search.bilibili.com",
        "Referer": "https://search.bilibili.com/",
    }
    return [
        {
            "User-Agent": "Mozilla/5.0 search-assistant/0.1",
            "Accept": "application/json",
            "Referer": "https://search.bilibili.com/",
        },
        desktop_headers,
    ]


def _extract_embedded_json_field(window: str, field_names: tuple[str, ...]) -> str:
    for field_name in field_names:
        patterns = (
            rf'{re.escape(field_name)}(?:&quot;|["\'])\s*:\s*(?:&quot;|["\'])(.*?)(?:&quot;|["\'])',
            rf'{re.escape(field_name)}\\?"\s*:\s*\\?"(.*?)(?:\\?"|")',
        )
        for pattern in patterns:
            match = re.search(pattern, window, flags=re.DOTALL | re.IGNORECASE)
            if match:
                value = _decode_embedded_json_value(match.group(1))
                if value:
                    return value
    return ""


def _decode_embedded_json_value(value: str) -> str:
    unescaped = html.unescape(value).replace("\\/", "/")
    try:
        return str(json.loads(f'"{unescaped}"'))
    except json.JSONDecodeError:
        return _decode_javascript_escapes(unescaped)


def _dedupe_search_results(results: list[SearchResult]) -> list[SearchResult]:
    unique: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        key = _dedupe_key(result.url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def _canonical_toutiao_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = _normalize_domain(parsed.hostname or "")
    if not _matches_scoped_domains(url, ("toutiao.com",)):
        return ""
    if host == "article.zlink.toutiao.com":
        return ""
    match = re.match(r"^/(?:group|article)/(\d+)/?", parsed.path)
    if not match:
        return ""
    return f"https://toutiao.com/group/{match.group(1)}"


def _extract_relevant_page_excerpt(html_body: str, query: str, max_chars: int = 1200) -> str:
    text = _html_to_text(html_body)
    escaped_text = _escaped_html_to_text(html_body)
    if escaped_text and len(escaped_text) > len(text):
        text = _clean_text(f"{text} {escaped_text}")
    if not text:
        return ""

    terms = _query_terms(query)
    if not terms:
        return text[:max_chars]

    keywords = set(terms)
    if any(term in {"dgx", "gb10", "deepseek", "model"} for term in terms):
        keywords.update({"memory", "unified", "superchip", "parameters", "tokens"})
    if "cxl" in terms:
        keywords.update({"memory", "accelerators", "cache", "coherent", "bandwidth"})
    if any(term in {"gr00t", "genie", "world", "physical", "robotics"} for term in terms):
        keywords.update({"robot", "robots", "world", "model", "interactive", "foundation"})

    lowered = text.lower()
    priority_terms = _priority_excerpt_terms(terms, lowered)
    priority_positions: list[int] = []
    for term in priority_terms:
        priority_positions.extend(match.start() for match in re.finditer(re.escape(term), lowered))

    positions: list[int] = []
    for term in terms:
        positions.extend(match.start() for match in re.finditer(re.escape(term), lowered))
    positions = priority_positions + [position for position in positions if position not in set(priority_positions)]
    if not positions:
        return text[:max_chars]

    best_excerpt = ""
    best_score = -1
    window_size = max_chars
    for position in positions[:80]:
        start = max(0, position - window_size // 3)
        end = min(len(text), start + window_size)
        window = text[start:end].strip()
        lowered_window = window.lower()
        score = sum(1 for keyword in keywords if keyword in lowered_window)
        score += 10 * sum(1 for priority_term in priority_terms if priority_term in lowered_window)
        if score > best_score:
            best_score = score
            best_excerpt = window
    return _trim_leading_excerpt_noise(best_excerpt, priority_terms)[:max_chars]


def _trim_leading_excerpt_noise(excerpt: str, priority_terms: list[str]) -> str:
    if not excerpt or not priority_terms:
        return excerpt
    lowered = excerpt.lower()
    positions = [lowered.find(term) for term in priority_terms if term in lowered]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return excerpt
    first_priority = min(positions)
    if first_priority <= 160:
        return excerpt
    context_start = max(0, first_priority - 120)
    leading_context = excerpt[context_start:first_priority]
    if _looks_like_navigation_noise(leading_context):
        return excerpt[first_priority:].lstrip()
    return excerpt[context_start:].lstrip()


def _looks_like_navigation_noise(text: str) -> bool:
    lowered = text.lower()
    navigation_markers = (
        "navigation home",
        "deployment api reference",
        "skip to content",
        "table of contents",
        "ctrl + k",
    )
    return any(marker in lowered for marker in navigation_markers)


def _priority_excerpt_terms(terms: list[str], lowered_text: str) -> list[str]:
    priority = [
        "671b",
        "37b",
        "activated",
        "total parameters",
        "connectx-7",
        "connectx",
        "200 gbps",
        "200gbps",
        "nic",
        "unified memory",
    ]
    query_terms = set(terms)
    if query_terms & {
        "distributed",
        "distributed inference",
        "inference",
        "model parallelism",
        "tensor parallelism",
        "pipeline parallelism",
        "expert parallelism",
        "parallelism",
    }:
        priority.extend(
            [
                "distributed inference",
                "single-model replica",
                "tensor parallelism",
                "pipeline parallelism",
                "expert parallelism",
                "all-to-all",
                "communication across nodes",
                "multi-node",
            ]
        )
    if query_terms & {"cxl", "compute express link"}:
        priority.extend(
            [
                "compute express link",
                "open industry-standard interconnect",
                "coherent memory",
                "memory pooling",
                "accelerators",
                "artificial intelligence",
            ]
        )
    return [
        term
        for term in priority
        if term in lowered_text
        and (
            term in query_terms
            or "deepseek" in query_terms
            or "dgx" in query_terms
            or "gb10" in query_terms
            or "connectx" in query_terms
        )
    ]


def _query_terms(query: str) -> list[str]:
    stopwords = {"the", "and", "for", "with", "from", "that", "this", "what", "how", "where"}
    terms: list[str] = []
    lowered_query = query.lower()
    terms.extend(_query_alias_terms(lowered_query, query))
    for token in re.findall(r"[A-Za-z0-9]+", query.lower()):
        if len(token) < 3 or token in stopwords:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _query_alias_terms(lowered_query: str, original_query: str) -> list[str]:
    aliases: list[str] = []
    alias_rules = (
        (
            ("分布式", "多节点", "distributed", "multi-node", "multinode"),
            ["distributed", "distributed inference", "multi-node"],
        ),
        (
            ("大模型", "llm", "large model", "large language model"),
            ["llm", "large model", "single-model replica"],
        ),
        (
            ("推理", "inference", "serving", "服务"),
            ["inference", "distributed inference", "serving"],
        ),
        (
            ("模型并行", "model parallel"),
            ["model parallelism", "parallelism"],
        ),
        (
            ("张量并行", "tensor parallel"),
            ["tensor parallelism", "parallelism"],
        ),
        (
            ("流水线并行", "pipeline parallel"),
            ["pipeline parallelism", "parallelism"],
        ),
        (
            ("专家并行", "expert parallel", "moe"),
            ["expert parallelism", "all-to-all", "parallelism"],
        ),
        (
            ("节点", "互联", "通信", "interconnect", "communication"),
            ["nodes", "interconnect", "communication across nodes"],
        ),
        (
            ("cxl", "compute express link"),
            ["cxl", "compute express link", "coherent memory", "memory pooling", "accelerators"],
        ),
    )
    for markers, expanded_terms in alias_rules:
        if any(marker in lowered_query or marker in original_query for marker in markers):
            for term in expanded_terms:
                if term not in aliases:
                    aliases.append(term)
    return aliases


def _html_to_text(html_body: str) -> str:
    html_body = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", html_body, flags=re.DOTALL | re.IGNORECASE)
    return _clean_text(_strip_tags(html_body))


def _escaped_html_to_text(html_body: str) -> str:
    decoded = _decode_javascript_escapes(html_body)
    if decoded == html_body:
        return ""
    return _clean_text(_strip_tags(decoded))


def _decode_javascript_escapes(value: str) -> str:
    decoded = re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        value,
    )
    return (
        decoded.replace("\\n", " ")
        .replace("\\r", " ")
        .replace("\\t", " ")
        .replace('\\"', '"')
        .replace("\\/", "/")
    )


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value)


def _dedupe_key(url: str) -> str:
    parsed = urllib.parse.urlparse(html.unescape(url).strip())
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", parsed.query, ""))


def _mcp_toolset_from_config(
    name: str,
    server: dict[str, object],
    config_directory: Path,
    timeout_seconds: float,
) -> Any:
    try:
        from fastmcp.client.transports import StdioTransport
        from pydantic_ai.mcp import MCPToolset
    except ModuleNotFoundError as exc:
        return _UnavailableMcpToolset(f"MCP dependency is missing: {exc.name}")

    expanded_server = _expand_mcp_environment(server)
    raw_url = expanded_server.get("url")
    if isinstance(raw_url, str) and raw_url.strip():
        parsed_url = urllib.parse.urlparse(raw_url)
        if parsed_url.scheme not in {"http", "https"}:
            raise RuntimeError(f"MCP server '{name}' URL must use http or https")
        raw_headers = expanded_server.get("headers", {})
        headers = {str(key): str(value) for key, value in raw_headers.items()} if isinstance(raw_headers, dict) else None
        return MCPToolset(
            raw_url,
            id=name,
            headers=headers,
            init_timeout=timeout_seconds,
            read_timeout=timeout_seconds,
        )

    command = expanded_server.get("command")
    raw_args = expanded_server.get("args", [])
    if not isinstance(command, str) or not command.strip() or not isinstance(raw_args, list):
        raise RuntimeError(f"MCP server '{name}' requires either a URL or a command with args")
    raw_environment = expanded_server.get("env", {})
    if raw_environment is not None and not isinstance(raw_environment, dict):
        raise RuntimeError(f"MCP server '{name}' env must be an object")
    env = {**os.environ, **{str(key): str(value) for key, value in (raw_environment or {}).items()}}
    raw_cwd = expanded_server.get("cwd")
    cwd = None
    if isinstance(raw_cwd, str) and raw_cwd.strip():
        candidate = Path(raw_cwd)
        cwd = str((config_directory / candidate).resolve()) if not candidate.is_absolute() else str(candidate)
    transport = StdioTransport(
        command=command,
        args=[str(arg) for arg in raw_args],
        env=env,
        cwd=cwd,
    )
    return MCPToolset(
        transport,
        id=name,
        init_timeout=timeout_seconds,
        read_timeout=timeout_seconds,
    )


class _UnavailableMcpToolset:
    def __init__(self, reason: str):
        self.reason = reason

    async def direct_call_tool(self, name: str, args: dict[str, object]) -> object:
        raise SearchProviderError(self.reason)


def _expand_mcp_environment(value: object) -> object:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            variable = match.group(1)
            default = match.group(2)
            resolved = os.environ.get(variable, default)
            if resolved is None:
                raise RuntimeError(f"MCP configuration requires environment variable {variable}")
            return resolved

        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", replace, value)
    if isinstance(value, list):
        return [_expand_mcp_environment(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_mcp_environment(item) for key, item in value.items()}
    return value


def _materialize_mcp_arguments(template: Mapping[str, object], query: str, limit: int) -> dict[str, object]:
    def materialize(value: object) -> object:
        if value == "{query}":
            return query
        if value == "{limit}":
            return limit
        if isinstance(value, str):
            return value.replace("{query}", query).replace("{limit}", str(limit))
        if isinstance(value, list):
            return [materialize(item) for item in value]
        if isinstance(value, dict):
            return {str(key): materialize(item) for key, item in value.items()}
        return value

    return {str(key): materialize(value) for key, value in template.items()}


def _mcp_results_to_search_results(raw_result: object, server_name: str) -> list[SearchResult]:
    raw_items = _mcp_result_items(raw_result)
    results: list[SearchResult] = []
    checked_at = datetime.now(UTC).isoformat()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("link") or item.get("html_url") or "").strip()
        title = _clean_text(str(item.get("title") or item.get("name") or url))
        snippet = _clean_text(str(item.get("snippet") or item.get("description") or item.get("content") or ""))
        source_provider = _clean_text(str(item.get("provider") or "search"))
        item_checked_at = str(item.get("checked_at") or checked_at).strip()
        if title and url.startswith(("http://", "https://")):
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    provider=f"mcp:{server_name}:{source_provider}",
                    checked_at=item_checked_at,
                )
            )
    return results


def _mcp_result_items(raw_result: object) -> list[object]:
    if isinstance(raw_result, str):
        try:
            return _mcp_result_items(json.loads(raw_result))
        except json.JSONDecodeError:
            return []
    if isinstance(raw_result, list):
        return raw_result
    if not isinstance(raw_result, dict):
        return []
    for key in ("results", "items", "data", "result"):
        value = raw_result.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _mcp_result_items(value)
            if nested:
                return nested
    return [raw_result] if any(key in raw_result for key in ("url", "link", "html_url")) else []


def _run_async_from_sync(factory: Callable[[], Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(factory())).result()


def _normalize_domain(value: str) -> str:
    return value.lower().strip().strip(".").removeprefix("www.")


def _site_domains(query: str) -> tuple[str, ...]:
    domains: list[str] = []
    for raw_domain in re.findall(r"(?<!\S)site:([^\s]+)", query, flags=re.IGNORECASE):
        domain = _normalize_domain(raw_domain)
        if domain and domain not in domains:
            domains.append(domain)
    return tuple(domains)


def _matches_scoped_domains(url: str, scoped_domains: tuple[str, ...]) -> bool:
    if not scoped_domains:
        return True
    host = _normalize_domain(urllib.parse.urlparse(url).hostname or "")
    return any(host == domain or host.endswith(f".{domain}") for domain in scoped_domains)


def _binding_supports_scoped_domains(binding: McpSearchBinding, scoped_domains: tuple[str, ...]) -> bool:
    if not binding.site_domains:
        return False
    return any(
        configured == requested
        or configured.endswith(f".{requested}")
        or requested.endswith(f".{configured}")
        for configured in binding.site_domains
        for requested in scoped_domains
    )


def _query_without_site_directives(query: str) -> str:
    without_directives = re.sub(r"(?<!\S)site:[^\s]+", " ", query, flags=re.IGNORECASE)
    return " ".join(part for part in without_directives.split() if part.upper() != "OR")


def _decode_bing_url(value: str) -> str:
    unescaped = html.unescape(value)
    parsed = urllib.parse.urlparse(unescaped)
    query = urllib.parse.parse_qs(parsed.query)
    if "u" not in query or not query["u"]:
        return unescaped

    encoded = query["u"][0]
    if encoded.startswith("a1"):
        encoded = encoded[2:]
    encoded += "=" * (-len(encoded) % 4)
    try:
        import base64

        return base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return unescaped


def _decode_google_url(value: str) -> str:
    unescaped = html.unescape(value)
    parsed = urllib.parse.urlparse(unescaped)
    if parsed.path == "/url":
        query = urllib.parse.parse_qs(parsed.query)
        if "q" in query and query["q"]:
            return query["q"][0]
    if parsed.scheme and parsed.netloc:
        return unescaped
    return urllib.parse.urljoin("https://www.google.com", unescaped)


def _baidu_original_result_url(html_body: str, title_start: int, href: str) -> str:
    """Prefer Baidu's public result `mu` target over its click-tracking URL."""
    prefix = html_body[max(0, title_start - 2400) : title_start]
    original_targets = re.findall(r'\bmu="([^"]+)"', prefix, flags=re.IGNORECASE)
    if original_targets:
        original_url = html.unescape(original_targets[-1]).strip()
        if original_url.startswith(("http://", "https://")):
            return original_url
    return html.unescape(href).strip()


def _is_baidu_redirect_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return _normalize_domain(parsed.hostname or "") == "baidu.com" and parsed.path.startswith("/link")


def _is_baidu_result_url(value: str) -> bool:
    host = _normalize_domain(urllib.parse.urlparse(value).hostname or "")
    return host == "baidu.com" or host.endswith(".baidu.com")


def _is_baidu_challenge_page(html_body: str) -> bool:
    return (
        "百度安全验证" in html_body
        or "网络不给力，请稍后重试" in html_body
        or ("timeout-title" in html_body and "mkdjump" in html_body)
    )


def _decode_duckduckgo_url(value: str) -> str:
    unescaped = html.unescape(value)
    parsed = urllib.parse.urlparse(unescaped)
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return query["uddg"][0]
    if parsed.scheme and parsed.netloc:
        return unescaped
    return urllib.parse.urljoin("https://duckduckgo.com", unescaped)


def _urllib_get(url: str, headers: dict[str, str]) -> str:
    return _urllib_get_with_timeout(20.0)(url, headers)


def _urllib_get_with_timeout(timeout_seconds: float) -> SearchTransport:
    def transport(url: str, headers: dict[str, str]) -> str:
        return _urllib_get_once(url, headers, timeout_seconds)

    return transport


def _urllib_get_once(url: str, headers: dict[str, str], timeout_seconds: float) -> str:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
            content_type = response.headers.get_content_charset() or "utf-8"
            return raw.decode(content_type, errors="replace")
    except OSError as exc:
        raise SearchProviderError(f"Search request failed: {exc}") from exc
