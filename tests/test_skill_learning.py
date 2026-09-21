"""Tests for the learning trigger (Path B): bad case -> LLM template -> staging draft."""

from __future__ import annotations

from search_assistant.skills.learning import DRAFT_INSTRUCTIONS, SkillLearningTrigger
from search_assistant.skills.library import VersionedSkillLibrary


def _fake_llm(instructions: str, payload: dict):
    assert "Bad Case 信息" in instructions
    assert "review-rejection" in payload["bad_case"]["id"]
    return (
        "---\n"
        "name: review-rejection-recovery\n"
        "version: 1\n"
        "description: Recover from a review-rejected answer.\n"
        "related:\n"
        "  - search-discipline\n"
        "---\n\n"
        "## Search Discipline\n\n"
        "- Re-search with concrete entities when the first pass lacks evidence.\n\n"
        "## Answer Discipline\n\n"
        "- Disclose evidence gaps and do not invent facts.\n"
    )


def test_learn_creates_staging_draft_with_llm_output(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")
    trigger = SkillLearningTrigger(library, llm_runner=_fake_llm)
    bad_case = {
        "id": "review-rejection-1",
        "question": "What is the current Feishu reply API?",
        "tier": "hard",
        "quality_flags": ["review_rejected", "blocked_answer"],
        "root_causes": ["task_blocked"],
        "attribution_layer": "prompt/skill",
    }
    result = trigger.learn(bad_case, fix_note="retry review parse failures")
    assert result.fixed is True
    assert result.bad_case_id == "review-rejection-1"
    skill = result.skill
    assert skill.state == "staging"
    assert skill.slug == "review-rejection-recovery"
    assert skill.version == 1
    assert "## Search Discipline" in skill.body
    assert library.find_staging("review-rejection-recovery") is not None


def test_learn_falls_back_to_bad_case_name_when_llm_omits_frontmatter(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")
    trigger = SkillLearningTrigger(
        library,
        llm_runner=lambda instructions, payload: "## Answer Discipline\n\n- Be careful.\n",
    )
    bad_case = {"id": "bc-2", "question": "Is CXL relevant to AI servers?"}
    skill = trigger.learn(bad_case).skill
    assert skill.state == "staging"
    assert "## Answer Discipline" in skill.body
    assert skill.slug


def test_learn_records_source_bad_case_in_frontmatter(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")

    def fake_llm_for_bc3(instructions, payload):
        assert payload["bad_case"]["id"] == "bc-3"
        return (
            "---\nname: review-rejection-recovery\nversion: 1\n---\n\n"
            "## Search Discipline\n\n- rule\n\n## Answer Discipline\n\n- rule\n"
        )

    trigger = SkillLearningTrigger(library, llm_runner=fake_llm_for_bc3)
    trigger.learn({"id": "bc-3", "question": "q"})
    record = library.find_staging("review-rejection-recovery")
    assert record is not None
    text = record.path.read_text(encoding="utf-8")
    assert "source_bad_case: bc-3" in text


def test_draft_instructions_template_mentions_disciplines():
    assert "## Search Discipline" in DRAFT_INSTRUCTIONS
    assert "## Answer Discipline" in DRAFT_INSTRUCTIONS
    assert "frontmatter" in DRAFT_INSTRUCTIONS
