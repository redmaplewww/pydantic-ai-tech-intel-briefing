from __future__ import annotations

from collections.abc import Callable
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from search_assistant.config import Settings
from search_assistant.memory.store import MemoryStore
from search_assistant.search.source_registry import source_contracts_as_dicts
from search_assistant.skills.service import SkillDraftService


class ReadinessService:
    def __init__(
        self,
        store: MemoryStore,
        settings: Settings,
        data_dir: str | Path,
        process_checker: Callable[[int], bool] | None = None,
        process_command_resolver: Callable[[int], str | None] | None = None,
    ):
        self.store = store
        self.settings = settings
        self.data_dir = Path(data_dir)
        self.process_checker = process_checker or _is_process_alive
        self.process_command_resolver = process_command_resolver or _process_command_line

    def run(self) -> dict[str, Any]:
        checks = [
            self._model_runtime_check(),
            self._browser_search_check(),
            self._source_contract_check(),
            self._provider_trace_check(),
            self._feishu_credentials_check(),
            self._runtime_session_check(),
            self._reply_attempt_check(),
            self._learning_evidence_check(),
            self._verification_evidence_check(),
            self._skill_evidence_check(),
            self._evaluation_report_check(),
        ]
        hints = self._hints(checks)
        return {
            "ok": all(check["ok"] for check in checks),
            "checks": checks,
            "counts": self.store.diagnostic_counts(),
            "hints": hints,
        }

    def _model_runtime_check(self) -> dict[str, Any]:
        provider = self.settings.model_provider.lower()
        if provider not in {"deepseek", "glm"}:
            latest_runtime = self._latest_runtime_model_config()
            if latest_runtime:
                return {
                    "name": "model_runtime",
                    "ok": True,
                    "detail": "latest runtime session was started with a configured model",
                    "data": latest_runtime,
                }
            return {
                "name": "model_runtime",
                "ok": False,
                "detail": "production requires SEARCH_ASSISTANT_MODEL_PROVIDER=glm or deepseek",
            }
        api_key = self.settings.glm_api_key if provider == "glm" else self.settings.deepseek_api_key
        model = self.settings.glm_model if provider == "glm" else self.settings.deepseek_model
        if not api_key:
            latest_runtime = self._latest_runtime_model_config()
            if latest_runtime:
                return {
                    "name": "model_runtime",
                    "ok": True,
                    "detail": "latest runtime session was started with a configured model",
                    "data": latest_runtime,
                }
            missing_key = "GLM_API_KEY" if provider == "glm" else "DEEPSEEK_API_KEY"
            return {"name": "model_runtime", "ok": False, "detail": f"{missing_key} is missing"}
        if self.settings.allow_fake_runtime:
            latest_runtime = self._latest_runtime_model_config()
            if latest_runtime:
                return {
                    "name": "model_runtime",
                    "ok": True,
                    "detail": "latest runtime session was started with fake runtime disabled",
                    "data": latest_runtime,
                }
            return {
                "name": "model_runtime",
                "ok": False,
                "detail": "SEARCH_ASSISTANT_ALLOW_FAKE_RUNTIME should be false for production",
            }
        return {
            "name": "model_runtime",
            "ok": True,
            "detail": f"{provider} configured with model {model}",
            "data": {"provider": provider, "model": model},
        }

    def _browser_search_check(self) -> dict[str, Any]:
        provider = self.settings.search_provider.lower()
        if provider in {"mcp", "hybrid"}:
            config_path = self.settings.mcp_search_config_path
            if config_path is None:
                config_path = Path(__file__).resolve().parents[3] / "configs" / "public-sources.mcp.json"
            if not config_path.exists():
                return {
                    "name": "browser_search",
                    "ok": False,
                    "detail": "MCP search provider requires an existing MCP search configuration",
                    "data": {"provider": provider, "mcp_config_path": str(config_path)},
                }
            if provider == "mcp":
                return {
                    "name": "browser_search",
                    "ok": True,
                    "detail": "MCP search configuration is present",
                    "data": {"provider": provider, "mcp_config_path": str(config_path)},
                }
            return {
                "name": "browser_search",
                "ok": True,
                "detail": "hybrid search has an MCP configuration and browser fallback",
                "data": {
                    "provider": provider,
                    "mcp_config_path": str(config_path),
                    "configured_engines": self.settings.browser_search_engines,
                },
            }
        if provider == "searxng":
            return {
                "name": "browser_search",
                "ok": True,
                "detail": "SearXNG search endpoint is configured; run a live health check before production use",
                "data": {"provider": provider, "base_url": self.settings.searxng_base_url},
            }
        if provider != "browser":
            return {
                "name": "browser_search",
                "ok": False,
                "detail": f"unsupported search provider: {self.settings.search_provider}",
            }
        required = {"bing", "baidu", "google"}
        configured = set(self.settings.browser_search_engines)
        if not required.issubset(configured):
            return {
                "name": "browser_search",
                "ok": False,
                "detail": "browser search should include bing, baidu, and google",
                "data": {"configured_engines": sorted(configured)},
            }
        return {
            "name": "browser_search",
            "ok": True,
            "detail": "browser search engines include bing, baidu, and google",
            "data": {"configured_engines": self.settings.browser_search_engines},
        }

    def _source_contract_check(self) -> dict[str, Any]:
        contracts = source_contracts_as_dicts()
        unsafe_contracts = [
            contract["slug"]
            for contract in contracts
            if contract.get("requires_login") or "write" not in contract.get("unsupported_actions", [])
        ]
        return {
            "name": "source_contracts",
            "ok": not unsafe_contracts,
            "detail": "public-source contracts are registered and keep login/write actions unsupported"
            if not unsafe_contracts
            else "one or more source contracts violate the public-source boundary",
            "data": {
                "contract_count": len(contracts),
                "unsafe_contracts": unsafe_contracts,
            },
        }

    def _provider_trace_check(self) -> dict[str, Any]:
        counts = self.store.diagnostic_counts()
        events = counts.get("provider_trace_events", 0)
        briefings = counts.get("daily_briefings", 0)
        ok = events > 0 or briefings == 0
        return {
            "name": "provider_trace",
            "ok": ok,
            "detail": "provider trace events are present for briefing/search replay"
            if events > 0
            else "no provider trace is present yet; run brief-run once to generate it",
            "data": {
                "provider_trace_events": events,
                "daily_briefings": briefings,
            },
        }

    def _feishu_credentials_check(self) -> dict[str, Any]:
        ok = bool(self.settings.feishu_app_id and self.settings.feishu_app_secret)
        if not ok:
            latest_runtime = self._latest_runtime_feishu_config()
            if latest_runtime:
                return {
                    "name": "feishu_credentials",
                    "ok": True,
                    "detail": "latest runtime session was started with Feishu credentials configured",
                    "data": latest_runtime,
                }
        return {
            "name": "feishu_credentials",
            "ok": ok,
            "detail": "FEISHU_APP_ID and FEISHU_APP_SECRET are configured"
            if ok
            else "FEISHU_APP_ID or FEISHU_APP_SECRET is missing",
        }

    def _runtime_session_check(self) -> dict[str, Any]:
        sessions = self._runtime_sessions(kind="feishu-long-connection")
        if not sessions:
            return {
                "name": "runtime_session",
                "ok": False,
                "detail": "no runtime session has been recorded after restart",
            }
        latest = sessions[0]
        process_status = self._session_process_status(latest)
        process_alive = bool(process_status["process_alive"])
        process_identity_ok = bool(process_status["process_identity_ok"])
        ok = latest["status"] == "started" and process_alive and process_identity_ok
        if latest["status"] != "started":
            detail = f"latest runtime session status is {latest['status']}"
        elif not process_alive:
            detail = f"latest runtime session process {latest['process_id']} is not running"
        elif not process_identity_ok:
            detail = f"latest runtime session process {latest['process_id']} does not match {process_status['expected_process']}"
        else:
            detail = f"latest runtime session status is {latest['status']} and process identity is verified"
        return {
            "name": "runtime_session",
            "ok": ok,
            "detail": detail,
            "data": {
                "kind": latest["kind"],
                "process_id": latest["process_id"],
                "process_alive": process_alive,
                "process_identity_ok": process_identity_ok,
                "expected_process": process_status["expected_process"],
                "created_at": latest["created_at"],
            },
        }

    def _latest_runtime_metadata(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        sessions = self._runtime_sessions(kind="feishu-long-connection")
        if not sessions:
            return None
        latest = sessions[0]
        if latest["status"] != "started":
            return None
        if not self._session_process_verified(latest):
            return None
        try:
            metadata = json.loads(str(latest.get("metadata_json") or "{}"))
        except json.JSONDecodeError:
            return None
        if not isinstance(metadata, dict):
            return None
        return latest, metadata

    def _runtime_sessions(self, kind: str | None = None) -> list[dict[str, Any]]:
        sessions = self.store.list_runtime_sessions()
        if kind is None:
            return sessions
        return [session for session in sessions if str(session.get("kind")) == kind]

    def _latest_runtime_model_config(self) -> dict[str, Any] | None:
        latest_metadata = self._latest_runtime_metadata()
        if latest_metadata is None:
            return None
        latest, metadata = latest_metadata
        provider = str(metadata.get("model_provider") or "").lower()
        configured_key = (
            metadata.get("glm_api_key_configured")
            if provider == "glm"
            else metadata.get("deepseek_api_key_configured")
            if provider == "deepseek"
            else False
        )
        if provider in {"glm", "deepseek"} and configured_key is True and metadata.get("allow_fake_runtime") is False:
            return {
                "source": "latest_runtime_session",
                "kind": latest["kind"],
                "process_id": latest["process_id"],
                "created_at": latest["created_at"],
                "provider": provider,
                "model": metadata.get("glm_model") if provider == "glm" else metadata.get("deepseek_model"),
            }
        return None

    def _latest_runtime_feishu_config(self) -> dict[str, Any] | None:
        latest_metadata = self._latest_runtime_metadata()
        if latest_metadata is None:
            return None
        latest, metadata = latest_metadata
        if metadata.get("feishu_app_id_configured") is True and metadata.get("feishu_app_secret_configured") is True:
            return {
                "source": "latest_runtime_session",
                "kind": latest["kind"],
                "process_id": latest["process_id"],
                "created_at": latest["created_at"],
            }
        return None

    def _reply_attempt_check(self) -> dict[str, Any]:
        attempts = self.store.list_reply_attempts(limit=1)
        if not attempts:
            return {"name": "reply_attempt", "ok": False, "detail": "no Feishu reply attempt recorded"}
        automatic_attempts = [
            attempt
            for attempt in self.store.list_reply_attempts()
            if attempt["channel"] in {"feishu-long-connection", "feishu-polling"}
        ]
        if not automatic_attempts:
            latest = attempts[0]
            return {
                "name": "reply_attempt",
                "ok": False,
                "detail": "no automatic Feishu reply attempt recorded",
                "data": {
                    "latest_message_id": latest["message_id"],
                    "latest_channel": latest["channel"],
                    "latest_status": latest["status"],
                },
            }
        latest_runtime_metadata = self._latest_runtime_metadata()
        if latest_runtime_metadata is not None:
            latest_runtime, _ = latest_runtime_metadata
            runtime_created_at = str(latest_runtime["created_at"])
            fresh_attempts = [
                attempt
                for attempt in automatic_attempts
                if _created_at_is_at_or_after(str(attempt.get("created_at", "")), runtime_created_at)
            ]
            if not fresh_attempts:
                latest = automatic_attempts[0]
                return {
                    "name": "reply_attempt",
                    "ok": False,
                    "detail": "no automatic Feishu reply attempt recorded after latest runtime session",
                    "data": {
                        "latest_message_id": latest["message_id"],
                        "latest_channel": latest["channel"],
                        "latest_status": latest["status"],
                        "latest_attempt_created_at": latest["created_at"],
                        "latest_runtime_created_at": runtime_created_at,
                    },
                }
            latest = fresh_attempts[0]
            return {
                "name": "reply_attempt",
                "ok": latest["status"] == "success",
                "detail": f"latest reply attempt after latest runtime session status is {latest['status']}",
                "data": {
                    "message_id": latest["message_id"],
                    "channel": latest["channel"],
                    "created_at": latest["created_at"],
                    "latest_runtime_created_at": runtime_created_at,
                },
            }
        latest = automatic_attempts[0]
        return {
            "name": "reply_attempt",
            "ok": latest["status"] == "success",
            "detail": f"latest reply attempt status is {latest['status']}",
            "data": {"message_id": latest["message_id"], "channel": latest["channel"]},
        }

    def _learning_evidence_check(self) -> dict[str, Any]:
        counts = self.store.diagnostic_counts()
        ok = (
            counts["answers"] > 0
            and counts["profile_snapshots"] > 0
            and counts["experience_items"] > 0
            and counts["learning_reports"] > 0
        )
        return {
            "name": "learning_evidence",
            "ok": ok,
            "detail": "answers, profile snapshots, experience items, and learning reports are present"
            if ok
            else "answers, profile snapshots, experience items, or learning reports are missing",
            "data": {
                "answers": counts["answers"],
                "profile_snapshots": counts["profile_snapshots"],
                "experience_items": counts["experience_items"],
                "learning_reports": counts["learning_reports"],
            },
        }

    def _verification_evidence_check(self) -> dict[str, Any]:
        counts = self.store.diagnostic_counts()
        answers = self.store.list_answers()
        source_backed_answers = sum(1 for answer in answers if answer.sources)
        verified_answer_packages = sum(1 for answer in answers if answer.verified_claims)
        ok = counts["evidence"] > 0
        if ok:
            detail = "verified claim evidence rows are present"
        elif source_backed_answers > 0:
            detail = "source-backed answers exist but no verified claim evidence rows are present"
        else:
            detail = "no verified claim evidence rows are present yet"
        return {
            "name": "verification_evidence",
            "ok": ok,
            "detail": detail,
            "data": {
                "evidence": counts["evidence"],
                "source_backed_answers": source_backed_answers,
                "answers_with_verified_claims": verified_answer_packages,
            },
        }

    def _skill_evidence_check(self) -> dict[str, Any]:
        status = SkillDraftService(
            self.store,
            drafts_dir=self.data_dir / "skills" / "drafts",
            active_dir=self.data_dir / "skills" / "active",
        ).list_review_status()
        counts = status["counts"]
        count = int(counts["total"])
        missing_active_files = int(counts["missing_files"])
        ok = count > 0 and missing_active_files == 0
        if count == 0:
            detail = "no skill drafts have been generated"
        elif missing_active_files > 0:
            detail = "one or more promoted active skill files are missing"
        elif int(counts["promoted"]) > 0:
            detail = "skill drafts and promoted active skills are present"
        else:
            detail = "skill drafts are present and await review"
        return {
            "name": "skill_evidence",
            "ok": ok,
            "detail": detail,
            "data": {
                "skill_drafts": count,
                "draft_skills": int(counts["draft"]),
                "promoted_skills": int(counts["promoted"]),
                "active_skill_files": int(counts["active_files"]),
                "missing_active_skill_files": missing_active_files,
            },
        }

    def _evaluation_report_check(self) -> dict[str, Any]:
        path = self.data_dir / "evaluations" / "evaluation-report.json"
        if not path.exists():
            return {
                "name": "evaluation_report",
                "ok": False,
                "detail": "evaluation report has not been generated",
                "data": {"path": str(path)},
            }
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "name": "evaluation_report",
                "ok": False,
                "detail": "evaluation report is not readable JSON",
                "data": {"path": str(path)},
            }
        missing_fields = _evaluation_report_missing_quality_fields(report)
        if missing_fields:
            return {
                "name": "evaluation_report",
                "ok": False,
                "detail": "evaluation report is missing quality audit fields; rerun eval-suite",
                "data": {"path": str(path), "missing_fields": missing_fields},
            }
        return {
            "name": "evaluation_report",
            "ok": True,
            "detail": "evaluation report exists with quality audit fields",
            "data": {"path": str(path)},
        }

    def _hints(self, checks: list[dict[str, Any]]) -> list[str]:
        hints: list[str] = []
        failed_names = {check["name"] for check in checks if not check["ok"]}
        if "model_runtime" in failed_names:
            hints.append("Set SEARCH_ASSISTANT_MODEL_PROVIDER=glm with GLM_API_KEY, or select deepseek with DEEPSEEK_API_KEY.")
        if "browser_search" in failed_names:
            hints.append("Set SEARCH_ASSISTANT_SEARCH_PROVIDER=hybrid and configure a valid MCP search config, or explicitly use browser/searxng.")
        if "runtime_session" in failed_names:
            hints.append("Restart feishu-long-connection after code changes, then run doctor again.")
        if "reply_attempt" in failed_names:
            hints.append("Send a new Feishu bot message after restart and inspect reply_attempts.")
        if "learning_evidence" in failed_names or "evaluation_report" in failed_names:
            hints.append("Run eval-suite or several real questions to generate learning evidence.")
        if "verification_evidence" in failed_names:
            hints.append(
                "Run evidence-backfill after verifier changes, or ask a verifiable sourced question, "
                "so verified claims are persisted in evidence."
            )
        if "skill_evidence" in failed_names:
            skill_check = next((check for check in checks if check["name"] == "skill_evidence"), None)
            skill_data = (skill_check or {}).get("data") or {}
            if skill_data.get("missing_active_skill_files"):
                hints.append("Inspect promoted skills and restore the missing active skill file or rerun skill-promote.")
            else:
                hints.append("Run evolve or skill-draft after profile snapshots exist.")
        skill_check = next((check for check in checks if check["name"] == "skill_evidence"), None)
        skill_data = (skill_check or {}).get("data") or {}
        if skill_check and skill_check.get("ok") and skill_data.get("skill_drafts", 0) > 0 and skill_data.get("promoted_skills", 0) == 0:
            hints.append("Review generated skill drafts and run skill-promote for approved skills.")
        if any(
            (check.get("data") or {}).get("source") == "latest_runtime_session"
            for check in checks
            if check["name"] in {"model_runtime", "feishu_credentials"}
        ):
            hints.append(
                "Current shell is missing credentials; the running long-connection session is configured, "
                "but set env vars or .env.local before restarting it."
            )
        return hints

    def _session_process_alive(self, session: dict[str, Any]) -> bool:
        try:
            process_id = int(session["process_id"])
        except (KeyError, TypeError, ValueError):
            return False
        try:
            return bool(self.process_checker(process_id))
        except OSError:
            return False

    def _session_process_verified(self, session: dict[str, Any]) -> bool:
        status = self._session_process_status(session)
        return bool(status["process_alive"] and status["process_identity_ok"])

    def _session_process_status(self, session: dict[str, Any]) -> dict[str, object]:
        process_alive = self._session_process_alive(session)
        expected_process = _expected_process_label(str(session.get("kind", "")))
        if not process_alive:
            return {
                "process_alive": False,
                "process_identity_ok": False,
                "expected_process": expected_process,
            }
        return {
            "process_alive": True,
            "process_identity_ok": self._session_process_identity_matches(session),
            "expected_process": expected_process,
        }

    def _session_process_identity_matches(self, session: dict[str, Any]) -> bool:
        kind = str(session.get("kind", ""))
        if not _requires_process_identity(kind):
            return True
        try:
            process_id = int(session["process_id"])
        except (KeyError, TypeError, ValueError):
            return False
        try:
            command_line = self.process_command_resolver(process_id)
        except OSError:
            return False
        if not command_line:
            return False
        return _command_line_matches_runtime_kind(kind, command_line)


def run_readiness_diagnostics(
    store: MemoryStore,
    data_dir: str | Path,
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved_settings = settings or Settings.from_env()
    return ReadinessService(store=store, settings=resolved_settings, data_dir=data_dir).run()


def _is_process_alive(process_id: int) -> bool:
    if process_id <= 0:
        return False
    if os.name == "nt":
        return _is_windows_process_alive(process_id)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_windows_process_alive(process_id: int) -> bool:
    process_query_limited_information = 0x1000
    error_access_denied = 5
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(process_query_limited_information, False, process_id)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return ctypes.get_last_error() == error_access_denied


def _process_command_line(process_id: int) -> str | None:
    if process_id <= 0:
        return None
    if os.name == "nt":
        return _windows_process_command_line(process_id)
    return _posix_process_command_line(process_id)


def _windows_process_command_line(process_id: int) -> str | None:
    command = (
        f"(Get-CimInstance Win32_Process -Filter \"ProcessId = {process_id}\").CommandLine"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _posix_process_command_line(process_id: int) -> str | None:
    path = Path("/proc") / str(process_id) / "cmdline"
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip() or None


def _requires_process_identity(kind: str) -> bool:
    return kind == "feishu-long-connection"


def _expected_process_label(kind: str) -> str:
    if kind == "feishu-long-connection":
        return "feishu-long-connection"
    return "not_required"


def _command_line_matches_runtime_kind(kind: str, command_line: str) -> bool:
    if kind != "feishu-long-connection":
        return True
    lowered = command_line.lower()
    return "feishu-long-connection" in lowered and "search_assistant" in lowered


def _created_at_is_at_or_after(created_at: str, threshold: str) -> bool:
    try:
        created = _parse_iso_datetime(created_at)
        minimum = _parse_iso_datetime(threshold)
    except ValueError:
        return False
    return created >= minimum


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    return datetime.fromisoformat(normalized)


def _evaluation_report_missing_quality_fields(report: object) -> list[str]:
    if not isinstance(report, dict):
        return ["report"]
    missing: list[str] = []
    items = report.get("items")
    if not isinstance(items, list) or not items:
        missing.append("items")
    else:
        required_item_fields = {
            "answer_excerpt",
            "search_record",
            "search_record_structured",
            "source_urls",
            "review_approved",
            "review_issues",
            "uncertainty_assessment",
            "quality_flags",
        }
        first_item = items[0]
        if not isinstance(first_item, dict):
            missing.append("items[0]")
        else:
            for field in sorted(required_item_fields):
                if field not in first_item:
                    missing.append(f"items[0].{field}")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        missing.append("summary")
    else:
        for field in ("flagged_answers", "review_rejected_answers"):
            if field not in summary:
                missing.append(f"summary.{field}")
    return missing
