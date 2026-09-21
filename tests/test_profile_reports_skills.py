from pathlib import Path

from search_assistant.contracts import AnswerPackage
from search_assistant.memory.store import MemoryStore
from search_assistant.profile.service import ProfileService
from search_assistant.reports.service import ReportService
from search_assistant.skills.service import SkillDraftService
from search_assistant.contracts import IncomingMessage
from search_assistant.runtime import FakeAgentRuntime
from search_assistant.workflow.service import SearchAssistantWorkflow


def test_profile_report_and_skill_draft_are_persisted(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    package = AnswerPackage(
        question_id="q-1",
        answer_text="You ask about Feishu and agent workflows.",
        classification="research",
        confidence="medium",
        verified_claims=[],
        unverified_claims=["Feishu API detail"],
        calibration={"ran": True, "critique": "Needs source", "revision": "Marked unverified"},
        memory_updates=[{"kind": "topic", "content": "Feishu agent workflow"}],
    )
    store.record_answer(package)

    profile_id = ProfileService(store).update_from_answer(package)
    report = ReportService(store, output_dir=tmp_path / "reports").generate_markdown()
    draft_path = SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts").create_from_experience(
        title="Feishu Workflow Answers",
        source_ids=[profile_id],
    )
    draft_body = (tmp_path / "skills" / "drafts" / "feishu-workflow-answers" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "Feishu" in report
    assert draft_path.endswith("SKILL.md")
    assert "Feishu agent workflow" in draft_body
    assert "## Evidence Summary" in draft_body
    assert store.list_profile_snapshots()[0]["id"] == profile_id
    assert store.list_skill_drafts()[0]["review_status"] == "draft"


def test_learning_report_summarizes_direction_and_suggestions(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=FakeAgentRuntime(answer_text="需要围绕飞书 bot、Agent Framework、核验和工作流持续学习。"),
    )
    profile_service = ProfileService(store)
    questions = [
        "我想系统学习 Microsoft Agent Framework 和飞书 bot 集成，请根据我的问题判断学习方向。",
        "请比较飞书机器人事件订阅和消息回复 API 的风险，并告诉我下一步该学什么。",
        "How should I verify current API version answers before replying in Feishu?",
    ]
    for index, question in enumerate(questions, start=1):
        package = workflow.answer(
            IncomingMessage(
                message_id=f"learn-{index}",
                event_id=f"learn-{index}",
                user_id="u-1",
                chat_id="c-1",
                text=question,
                source="cli",
            )
        )
        profile_service.update_from_answer(package)

    report = ReportService(store, output_dir=tmp_path / "reports").generate_markdown()

    assert "## Learning Direction" in report
    assert "Feishu integration" in report
    assert "Microsoft Agent Framework" in report
    assert "## Recommended Next Learning Actions" in report
    assert "Build a Feishu bot callback and reply checklist" in report


def test_learning_report_recommends_actions_for_ai_infrastructure_topics(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.add_memory_item("topic", "AI infrastructure and model deployment", "q-gb10")
    store.add_memory_item("topic", "AI memory architecture", "q-cxl")
    store.add_memory_item("topic", "Physical AI and world models", "q-world")

    report = ReportService(store, output_dir=tmp_path / "reports").generate_markdown()

    assert "AI infrastructure and model deployment" in report
    assert "Build a model deployment feasibility checklist" in report
    assert "Map CXL, memory pooling, and accelerator memory tiers" in report
    assert "Track world-model and embodied-AI releases against primary sources" in report


def test_skill_draft_auto_creation_uses_profile_topics_and_avoids_duplicates(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    profile_id = store.add_profile_snapshot(
        {
            "recurring_topics": ["Feishu integration", "API reliability"],
            "preferred_answer_style": "precise answers with verification notes",
        }
    )
    service = SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts")

    paths = service.auto_create_from_experience(max_drafts=2)
    second_run_paths = service.auto_create_from_experience(max_drafts=2)

    assert len(paths) == 2
    assert second_run_paths == []
    assert len(store.list_skill_drafts()) == 2
    first_body = (tmp_path / "skills" / "drafts" / "feishu-integration-practice" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert profile_id in first_body
    assert "Feishu integration" in first_body
    assert "## Evidence Summary" in first_body


def test_skill_draft_contains_topic_specific_guidance_for_ai_infrastructure(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    profile_id = store.add_profile_snapshot(
        {
            "recurring_topics": ["AI infrastructure and model deployment"],
            "preferred_answer_style": "precise answers with verification notes",
        }
    )
    service = SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts")

    paths = service.auto_create_from_experience(max_drafts=1)

    body = Path(paths[0]).read_text(encoding="utf-8")
    assert profile_id in body
    assert "## Search Discipline" in body
    assert "ModelScope/魔搭/魔塔" in body
    assert "Hugging Face" in body
    assert "GitHub" in body
    assert "NVIDIA official" in body
    assert "ConnectX-7" in body
    assert "Do not present tok/s" in body


def test_auto_creation_refreshes_old_review_only_draft_templates(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.add_profile_snapshot(
        {
            "recurring_topics": ["AI infrastructure and model deployment"],
            "preferred_answer_style": "precise answers with verification notes",
        }
    )
    service = SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts")
    paths = service.auto_create_from_experience(max_drafts=1)
    draft_path = Path(paths[0])
    draft_path.write_text(
        (
            "---\n"
            "name: ai-infrastructure-and-model-deployment-practice\n"
            "description: old generic draft\n"
            "---\n\n"
            "## Instructions\n\n"
            "Answer similar future questions with explicit assumptions, verification notes, and a short final recommendation.\n"
        ),
        encoding="utf-8",
    )

    second_run_paths = service.auto_create_from_experience(max_drafts=1)

    refreshed = draft_path.read_text(encoding="utf-8")
    assert second_run_paths == []
    assert "old generic draft" not in refreshed
    assert "ModelScope/魔搭/魔塔" in refreshed
    assert "Do not present tok/s" in refreshed


def test_report_and_skill_draft_include_experience_items(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    experience_id = store.add_experience_item(
        title="Verification habit",
        body="Search before answering current API questions, then calibrate the draft.",
        source_ids=["q-exp"],
    )

    report = ReportService(store, output_dir=tmp_path / "reports").generate_markdown()
    draft_path = SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts").create_from_experience(
        title="Verification Habit",
        source_ids=[experience_id],
    )
    draft_body = (tmp_path / "skills" / "drafts" / "verification-habit" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "## Experience Notes" in report
    assert "Verification habit" in report
    assert draft_path.endswith("SKILL.md")
    assert "Search before answering current API questions" in draft_body


def test_skill_draft_auto_creation_can_use_experience_items_without_profiles(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    experience_id = store.add_experience_item(
        title="CXL Answering Pattern",
        body="Explain CXL with memory pooling, GPU connectivity, and source verification.",
        source_ids=["q-cxl"],
    )
    service = SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts")

    paths = service.auto_create_from_experience(max_drafts=1)

    assert len(paths) == 1
    assert len(store.list_skill_drafts()) == 1
    body = (tmp_path / "skills" / "drafts" / "cxl-answering-pattern" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert experience_id in body
    assert "memory pooling" in body


def test_skill_draft_auto_creation_specializes_evaluation_quality_issues(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    review_experience_id = store.add_experience_item(
        title="Evaluation quality issue",
        body=(
            "question=What is the current Feishu message reply API behavior?\n"
            "quality_flags=low_confidence, review_rejected, blocked_answer\n"
            "review_issues=unsupported model-memory claims\n"
            "future_rule=Use evaluation failures as self-evolution input for search planning, answer style, calibration, and skill drafts."
        ),
        source_ids=["q-review"],
    )
    search_experience_id = store.add_experience_item(
        title="Evaluation quality issue",
        body=(
            "question=What is CXL?\n"
            "quality_flags=no_sources, missing_search_record\n"
            "review_issues=\n"
            "future_rule=Use evaluation failures as self-evolution input for search planning, answer style, calibration, and skill drafts."
        ),
        source_ids=["q-search"],
    )
    service = SkillDraftService(store, drafts_dir=tmp_path / "skills" / "drafts")

    paths = service.auto_create_from_experience(max_drafts=3)

    drafts = store.list_skill_drafts()
    names = [draft["name"] for draft in drafts]
    assert len(paths) == 2
    assert "Evaluation Review Rejection Recovery Practice" in names
    assert "Evaluation Search Coverage Recovery Practice" in names
    assert all(name != "Evaluation quality issue" for name in names)
    review_body = (tmp_path / "skills" / "drafts" / "evaluation-review-rejection-recovery-practice" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    search_body = (tmp_path / "skills" / "drafts" / "evaluation-search-coverage-recovery-practice" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert review_experience_id in review_body
    assert search_experience_id in search_body
    assert "unsupported model-memory claims" in review_body
    assert "missing_search_record" in search_body


def test_skill_draft_promote_copies_reviewed_draft_to_active_dir_and_updates_status(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = SkillDraftService(
        store,
        drafts_dir=tmp_path / "skills" / "drafts",
        active_dir=tmp_path / "skills" / "active",
    )
    draft_path = service.create_from_experience("Verification Habit", source_ids=["manual"])

    promoted_path = service.promote("verification-habit")

    active_path = tmp_path / "skills" / "active" / "verification-habit" / "SKILL.md"
    active_body = active_path.read_text(encoding="utf-8")
    draft_body = Path(draft_path).read_text(encoding="utf-8")
    assert promoted_path == str(active_path)
    assert active_body != draft_body
    assert "Use this draft only after human review" not in active_body
    assert "Draft skill generated" not in active_body
    assert "This skill has been reviewed and promoted as procedural guidance." in active_body
    assert "Do not treat it as factual evidence." in active_body
    assert "Source ids reviewed before promotion:" in active_body
    assert "- manual" in active_body
    assert "## Search Discipline" in active_body
    assert "## Answer Discipline" in active_body
    promoted = store.list_skill_drafts()[0]
    assert promoted["review_status"] == "promoted"
    assert promoted["path"] == str(active_path)


def test_skill_draft_promote_refuses_to_overwrite_active_skill(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = SkillDraftService(
        store,
        drafts_dir=tmp_path / "skills" / "drafts",
        active_dir=tmp_path / "skills" / "active",
    )
    service.create_from_experience("Verification Habit", source_ids=["manual"])
    active_path = tmp_path / "skills" / "active" / "verification-habit" / "SKILL.md"
    active_path.parent.mkdir(parents=True)
    active_path.write_text("already reviewed", encoding="utf-8")

    try:
        service.promote("Verification Habit")
    except FileExistsError:
        pass
    else:
        raise AssertionError("promote should not overwrite an existing active skill")

    assert active_path.read_text(encoding="utf-8") == "already reviewed"
    assert store.list_skill_drafts()[0]["review_status"] == "draft"


def test_skill_draft_service_lists_review_status_and_file_health(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = SkillDraftService(
        store,
        drafts_dir=tmp_path / "skills" / "drafts",
        active_dir=tmp_path / "skills" / "active",
    )
    service.create_from_experience("Draft Only Guidance", source_ids=["manual"])
    service.create_from_experience("Verification Habit", source_ids=["manual"])
    active_path = service.promote("verification-habit")

    result = service.list_review_status()

    assert result["counts"] == {
        "total": 2,
        "draft": 1,
        "promoted": 1,
        "active_files": 1,
        "missing_files": 0,
    }
    assert [item["slug"] for item in result["skills"]] == ["draft-only-guidance", "verification-habit"]
    assert result["skills"][0]["review_status"] == "draft"
    assert result["skills"][0]["file_exists"] is True
    assert result["skills"][1]["review_status"] == "promoted"
    assert result["skills"][1]["path"] == active_path
    assert result["skills"][1]["active"] is True
