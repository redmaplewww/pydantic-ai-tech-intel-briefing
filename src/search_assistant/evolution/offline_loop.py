"""Offline evolution loop: turn immutable trajectories into verified learning signals.

This is the offline half of the online/offline double loop:

1. **Aggregate** pending immutable trajectories (those without a persisted
   trajectory evaluation).
2. **Verify** each with the three-layer verifier
   (:mod:`search_assistant.evolution.verification`) — result, process, and
   rubric quality layers, each carrying evidence locations.
3. **Diagnose** failures into reviewable update carriers (prompt/skill,
   program/harness) without mutating prompts or code.
4. **Persist** trajectory evaluations and route failures to experience items.
5. **Distill** domain-knowledge candidates: re-run validation gates so claims
   that reappeared in an independent trajectory can accumulate
   cross-trajectory support.
6. **Monitor release**: check the latest eval report gate and surface
   regressed candidates for rollback.  Release remains human-gated; the loop
   never auto-promotes or auto-deprecates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from search_assistant.evolution.service import DomainKnowledgeCandidateService, EvolutionDiagnosisService
from search_assistant.evolution.verification import RULE_JUDGE_NAME, QualityJudge, TrajectoryVerifier
from search_assistant.memory.store import MemoryStore


class OfflineEvolutionLoop:
    def __init__(
        self,
        store: MemoryStore,
        data_dir: str | Path,
        verifier: TrajectoryVerifier | None = None,
        judge: QualityJudge | None = None,
        candidate_service: DomainKnowledgeCandidateService | None = None,
        diagnosis_service: EvolutionDiagnosisService | None = None,
    ):
        self.store = store
        self.data_dir = Path(data_dir)
        self.verifier = verifier or TrajectoryVerifier(judge=judge)
        self.candidate_service = candidate_service or DomainKnowledgeCandidateService(store)
        self.diagnosis_service = diagnosis_service or EvolutionDiagnosisService()

    def run(self, max_trajectories: int | None = None) -> dict[str, Any]:
        trajectories = self.store.list_trajectory_logs()
        evaluated_ids = {str(item["trajectory_id"]) for item in self.store.list_trajectory_evaluations()}
        pending = [trajectory for trajectory in trajectories if str(trajectory["id"]) not in evaluated_ids]
        if max_trajectories is not None:
            pending = pending[-max_trajectories:]

        items: list[dict[str, Any]] = []
        failure_count = 0
        for trajectory in pending:
            verification = self.verifier.verify(trajectory)
            diagnosis_input = self._diagnosis_input(verification, trajectory)
            diagnosis = self.diagnosis_service.diagnose(diagnosis_input, str(trajectory["id"]))
            evaluation_id = self.store.add_trajectory_evaluation(
                trajectory_id=str(trajectory["id"]),
                question_id=str((trajectory.get("payload") or {}).get("question_id", "")),
                result_verification=verification["result_verification"],
                process_verification=verification["process_verification"],
                quality_verification=verification["quality_verification"],
                diagnosis=diagnosis,
            )
            item = {
                "trajectory_id": str(trajectory["id"]),
                "question": verification["question"],
                "evaluation_id": evaluation_id,
                "result_flags": verification["result_verification"]["flags"],
                "process_flags": verification["process_verification"]["flags"],
                "quality_flags": verification["quality_verification"]["flags"],
                "diagnosis": diagnosis,
            }
            items.append(item)
            if diagnosis["root_causes"]:
                failure_count += 1
                self.store.add_experience_item(
                    title="Offline evolution quality issue",
                    body=self._experience_body(verification, diagnosis),
                    source_ids=[str((trajectory.get("payload") or {}).get("question_id", ""))],
                    user_id=str(trajectory.get("user_id") or "system"),
                    chat_id=str(trajectory.get("chat_id") or "system"),
                )

        distillation = self.candidate_service.distill_candidates()
        release_monitor = self._release_monitor()
        rollback_monitor = self._rollback_monitor()
        judge_name = getattr(self.verifier.judge, "name", RULE_JUDGE_NAME)

        report = {
            "ok": failure_count == 0,
            "judge": judge_name,
            "aggregated_trajectories": len(trajectories),
            "evaluated_trajectories": len(evaluated_ids),
            "pending_trajectories": len(pending),
            "verified": len(items),
            "failures": failure_count,
            "items": items,
            "distillation": distillation,
            "release_monitor": release_monitor,
            "rollback_monitor": rollback_monitor,
            "report_path": str(self.data_dir / "evolution" / "offline-evolution-report.json"),
        }
        output_path = self.data_dir / "evolution"
        output_path.mkdir(parents=True, exist_ok=True)
        report_file = output_path / "offline-evolution-report.json"
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        gate_id = self.store.add_gate_record(
            gate_type="evolution_offline_run",
            subject_type="evolution_run",
            subject_id=str(report_file),
            result="passed" if failure_count == 0 else "failed",
            reason=(
                "offline trajectory verification found no issues"
                if failure_count == 0
                else f"offline trajectory verification found {failure_count} trajectories with diagnosed failures"
            ),
            evidence_refs=[str(report_file), *[str(item["evaluation_id"]) for item in items[:10]]],
            metadata={
                "judge": judge_name,
                "aggregated_trajectories": len(trajectories),
                "verified": len(items),
                "failures": failure_count,
                "distillation": {
                    "considered": distillation["considered"],
                    "validated": distillation["validated"],
                    "weak_signal": distillation["weak_signal"],
                },
                "release_monitor_passed": release_monitor.get("passed"),
            },
        )
        self.store.add_project_ledger_entry(
            entry_type="offline_evolution_run",
            subject=str(report_file),
            status="completed" if failure_count == 0 else "failed",
            summary=(
                f"Verified {len(items)} trajectories with the {judge_name} quality judge and "
                f"re-ran {distillation['considered']} knowledge-candidate gates."
            ),
            evidence_refs=[gate_id, str(report_file)],
            risk="Verification flags and candidates remain reviewable artifacts; no prompt, skill, or code was changed by this run.",
            rollback="Delete the offline evolution report and any experience items added by this run from an isolated data directory.",
            metadata={
                "judge": judge_name,
                "failures": failure_count,
                "release_monitor_passed": release_monitor.get("passed"),
                "rollback_candidates": len(rollback_monitor.get("regressed_candidates", [])),
            },
        )
        return report

    @staticmethod
    def _diagnosis_input(verification: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
        payload = trajectory.get("payload") or {}
        source_urls: list[str] = []
        search_record = payload.get("search_record") or {}
        for item in search_record.get("sources") or []:
            if isinstance(item, dict) and str(item.get("url", "")).startswith(("http://", "https://")):
                source_urls.append(str(item["url"]))
        return {
            "result_verification": verification["result_verification"],
            "process_verification": verification["process_verification"],
            "quality_verification": verification["quality_verification"],
            "question_id": str(payload.get("question_id", "")),
            "source_urls": source_urls,
        }

    @staticmethod
    def _experience_body(verification: dict[str, Any], diagnosis: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"trajectory_id={verification['trajectory_id']}",
                f"question={verification['question']}",
                f"result_flags={', '.join(verification['result_verification']['flags'])}",
                f"process_flags={', '.join(verification['process_verification']['flags'])}",
                f"quality_flags={', '.join(verification['quality_verification']['flags'])}",
                f"root_causes={', '.join(diagnosis.get('root_causes', []))}",
                f"recommended_update_carrier={diagnosis.get('recommended_update_carrier', 'none')}",
                f"target={diagnosis.get('target', '')}",
                "future_rule=Use offline trajectory verification failures as self-evolution input; never treat a "
                "verification flag as proven fact without reopening the original evidence.",
            ]
        )

    def _release_monitor(self) -> dict[str, Any]:
        """Check the latest eval report before any knowledge release happens."""
        report_path = self.data_dir / "evaluations" / "evaluation-report.json"
        if not report_path.exists():
            return {
                "checked": False,
                "passed": None,
                "reason": "no evaluation report; run eval-suite before releasing knowledge candidates",
                "report_path": str(report_path),
            }
        try:
            report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError) as exc:
            return {
                "checked": True,
                "passed": False,
                "reason": f"evaluation report is not readable: {exc}",
                "report_path": str(report_path),
            }
        summary = report.get("summary") or {}
        blocking = {
            "flagged_answers": int(summary.get("flagged_answers", 0) or 0),
            "review_rejected_answers": int(summary.get("review_rejected_answers", 0) or 0),
            "result_failed_answers": int(summary.get("result_failed_answers", 0) or 0),
            "process_flagged_answers": int(summary.get("process_flagged_answers", 0) or 0),
        }
        passed = all(value == 0 for value in blocking.values())
        return {
            "checked": True,
            "passed": passed,
            "reason": "evaluation report passed release gate" if passed else "evaluation report has blocking findings",
            "report_path": str(report_path),
            "summary": blocking,
        }

    def _rollback_monitor(self) -> dict[str, Any]:
        """Surface candidates whose cross-trajectory support has regressed.

        Non-destructive: the loop never auto-deprecates.  A candidate that was
        validated but no longer holds at least two independent trajectories
        (e.g. after a data-dir restore or a manual source edit) is listed as a
        rollback recommendation.
        """
        regressed: list[dict[str, Any]] = []
        for candidate in self.store.list_domain_knowledge_candidates(status="validated"):
            trajectory_count = self.candidate_service.independent_trajectory_count(candidate)
            if trajectory_count < 2:
                regressed.append(
                    {
                        "candidate_id": str(candidate["id"]),
                        "topic": str(candidate.get("topic", "")),
                        "independent_trajectory_count": trajectory_count,
                        "recommendation": "deprecate and re-approve from fresh independent evidence",
                    }
                )
        return {"regressed_candidates": regressed}
