from __future__ import annotations

from pathlib import Path

from search_assistant.memory.store import MemoryStore


class ReportService:
    def __init__(self, store: MemoryStore, output_dir: str | Path = "reports"):
        self.store = store
        self.output_dir = Path(output_dir)

    def generate_markdown(self, start: str | None = None, end: str | None = None) -> str:
        answers = self.store.list_answers()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "learning-report.md"
        lines = [
            "# Learning Report",
            "",
            f"Question volume: {len(answers)}",
            "",
            "## Learning Direction",
        ]
        topics = self._learning_topics()
        if topics:
            lines.extend(f"- {topic}" for topic in topics)
        else:
            lines.append("- No learning direction detected yet.")
        lines.extend(
            [
                "",
                "## Recommended Next Learning Actions",
            ]
        )
        lines.extend(f"- {action}" for action in self._recommended_actions(topics))
        lines.extend(
            [
                "",
                "## Important Answers",
            ]
        )
        for answer in answers:
            lines.append(f"- ({answer.classification}, confidence {answer.confidence}) {answer.answer_text}")
        lines.extend(
            [
                "",
                "## Experience Notes",
            ]
        )
        experiences = self.store.list_experience_items()
        if experiences:
            for item in experiences:
                lines.append(f"- {item['title']}: {item['body']}")
        else:
            lines.append("- No experience notes recorded yet.")
        lines.extend(
            [
                "",
                "## Trajectory Evaluation Summary",
            ]
        )
        lines.extend(self._trajectory_summary_lines())
        lines.extend(
            [
                "",
                "## Domain Knowledge Candidates",
            ]
        )
        lines.extend(self._domain_knowledge_candidate_lines())
        lines.extend(
            [
                "",
                "## AgentOps Infrastructure Readiness",
            ]
        )
        lines.extend(self._agentops_infrastructure_lines())
        lines.extend(
            [
                "",
                "## Unresolved Or Weakly Verified Areas",
            ]
        )
        weak_claims = [claim for answer in answers for claim in answer.unverified_claims]
        if weak_claims:
            lines.extend(f"- {claim}" for claim in weak_claims)
        else:
            lines.append("- No weakly verified areas recorded.")
        markdown = "\n".join(lines) + "\n"
        report_path.write_text(markdown, encoding="utf-8")
        self.store.add_learning_report(markdown, str(report_path))
        return markdown

    def _trajectory_summary_lines(self) -> list[str]:
        trajectories = self.store.list_trajectory_logs()
        evaluations = self.store.list_trajectory_evaluations()
        result_failures = sum(
            1 for item in evaluations if not item["result_verification"].get("passed", False)
        )
        process_warnings = sum(
            1 for item in evaluations if not item["process_verification"].get("passed", False)
        )
        quality_warnings = sum(
            1 for item in evaluations if not item["quality_verification"].get("passed", False)
        )
        return [
            f"- Immutable trajectories: {len(trajectories)}",
            f"- Structured evaluations: {len(evaluations)}",
            f"- Result failures: {result_failures}",
            f"- Process warnings: {process_warnings}",
            f"- Quality warnings: {quality_warnings}",
        ]

    def _domain_knowledge_candidate_lines(self) -> list[str]:
        candidates = self.store.list_domain_knowledge_candidates()
        if not candidates:
            return ["- No search-derived domain knowledge candidates recorded yet."]
        links_by_candidate: dict[str, int] = {}
        for link in self.store.list_domain_candidate_graph_links():
            candidate_id = str(link["candidate_id"])
            links_by_candidate[candidate_id] = links_by_candidate.get(candidate_id, 0) + 1
        return [
            (
                f"- [{item['status']}; confidence {item['confidence']}] {item['topic']}: "
                f"{item['claim']} (evidence: {len(item['evidence'])}; "
                f"graph links: {links_by_candidate.get(str(item['id']), 0)})"
            )
            for item in candidates
        ]

    def _agentops_infrastructure_lines(self) -> list[str]:
        counts = self.store.diagnostic_counts()
        gates = self.store.list_gate_records()
        gate_status = {
            result: sum(1 for item in gates if item["result"] == result)
            for result in ("passed", "failed", "waived")
        }
        provider_summary = self.store.search_provider_health_summary()
        return [
            f"- Project ledger entries: {counts['project_ledger_entries']}",
            (
                "- Self-evolution gate records: "
                f"{counts['agentops_gate_records']} "
                f"(passed {gate_status['passed']}, failed {gate_status['failed']}, waived {gate_status['waived']})"
            ),
            f"- Trace events: {counts['agentops_trace_events']}",
            (
                "- Search provider health records: "
                f"{counts['search_provider_health']} across {len(provider_summary['platforms'])} requested platforms"
            ),
        ]

    def _learning_topics(self) -> list[str]:
        seen: set[str] = set()
        topics: list[str] = []
        for item in self.store.list_memory_items():
            content = item["content"]
            if content not in seen:
                seen.add(content)
                topics.append(content)
        return topics

    def _recommended_actions(self, topics: list[str]) -> list[str]:
        actions: list[str] = []
        topic_set = set(topics)
        if "Feishu integration" in topic_set:
            actions.append("Build a Feishu bot callback and reply checklist")
        if "Microsoft Agent Framework" in topic_set:
            actions.append("Map the Agent Framework workflow, runtime, and tool boundaries")
        if "Verification practice" in topic_set or "API reliability" in topic_set:
            actions.append("Create a key-data verification checklist before answers")
        if "Learning direction planning" in topic_set:
            actions.append("Turn repeated questions into a weekly learning plan")
        if "AI infrastructure and model deployment" in topic_set:
            actions.append("Build a model deployment feasibility checklist")
        if "AI memory architecture" in topic_set:
            actions.append("Map CXL, memory pooling, and accelerator memory tiers")
        if "Physical AI and world models" in topic_set:
            actions.append("Track world-model and embodied-AI releases against primary sources")
        if not actions:
            actions.append("Ask three focused questions in one topic so the assistant can identify a learning direction")
        return actions
