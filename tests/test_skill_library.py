"""Tests for the versioned skill library (Path B): frontmatter, versioning,
staging -> merge/reject flows, and archive-not-delete semantics."""

from __future__ import annotations

import pytest

from search_assistant.skills.library import (
    VersionedSkillLibrary,
    parse_frontmatter,
    render_skill_document,
    slugify,
)


def test_slugify_normalizes_names():
    assert slugify("Review Rejection Recovery") == "review-rejection-recovery"
    assert slugify("   ") == "skill"
    assert slugify("搜索纪律 Practice") == "practice"


def test_parse_frontmatter_roundtrip():
    text = (
        "---\n"
        "name: review-rejection-recovery\n"
        "version: 3\n"
        "description: Recovery guidance\n"
        "related:\n"
        "  - search-discipline\n"
        "  - answer-discipline\n"
        "---\n\n"
        "# Body\n\nSome guidance.\n"
    )
    meta, body = parse_frontmatter(text)
    assert meta["name"] == "review-rejection-recovery"
    assert meta["version"] == 3
    assert meta["related"] == ["search-discipline", "answer-discipline"]
    assert "Some guidance." in body
    rendered = render_skill_document(meta, body)
    assert rendered.startswith("---\nname: review-rejection-recovery")
    assert "version: 3" in rendered


def test_parse_frontmatter_without_frontmatter():
    meta, body = parse_frontmatter("# Legacy\n\nno meta")
    assert meta == {}
    assert "no meta" in body


def test_create_draft_and_version_bump_on_rewrite(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")
    draft = library.create_draft("Recovery Practice", "body v1", description="d")
    assert draft.state == "staging"
    assert draft.version == 1
    assert draft.path.exists()

    revised = library.revise_draft("recovery-practice", "body v2")
    assert revised.version == 2
    assert revised.body.strip() == "body v2"


def test_create_draft_duplicate_bumps_version(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")
    library.create_draft("Recovery Practice", "body v1")
    second = library.create_draft("Recovery Practice", "body v1 again")
    assert second.version == 2


def test_promote_moves_staging_to_active_and_bumps_over_existing(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")
    library.create_draft("Recovery Practice", "draft body", version=1)
    active = library.promote("recovery-practice")
    assert active.state == "active"
    assert active.version == 1
    assert library.find_staging("recovery-practice") is None
    assert active.path.exists()

    # A second revision becomes version 2 over the existing active skill.
    library.create_draft("Recovery Practice", "draft body v2")
    active2 = library.promote("recovery-practice")
    assert active2.version == 2
    assert active2.path == active.path  # same active file, overwritten in place
    assert "draft body v2" in active2.path.read_text(encoding="utf-8")
    assert "version: 2" in active2.path.read_text(encoding="utf-8")


def test_reject_moves_staging_to_archive_never_deletes(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")
    library.create_draft("Rejected Practice", "draft body")
    rejected = library.reject("rejected-practice", reason="p not significant")
    assert rejected.state == "archive"
    assert rejected.path.exists()
    assert library.find_staging("rejected-practice") is None
    archived_text = rejected.path.read_text(encoding="utf-8")
    assert "archived: rejected" in archived_text
    assert "archive_reason: p not significant" in archived_text


def test_archive_deprecates_active_skill(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")
    library.create_draft("Old Practice", "body")
    library.promote("old-practice")
    archived = library.archive("old-practice", reason="superseded")
    assert archived.state == "archive"
    assert archived.path.exists()
    assert library.find_active("old-practice") is None
    assert len(library.list_archive()) == 1


def test_list_states_are_separated(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")
    library.create_draft("Staged A", "a")
    library.create_draft("Staged B", "b")
    library.promote("staged-a")
    library.reject("staged-b", reason="no")
    library.create_draft("Active C", "c")
    library.promote("active-c")
    library.archive("active-c", reason="retired")

    active = [record.slug for record in library.list_active()]
    staging = [record.slug for record in library.list_staging()]
    archive = [record.slug for record in library.list_archive()]
    # staged-a was promoted and never archived; active-c was promoted then
    # archived; staged-b was rejected straight into the archive.
    assert active == ["staged-a"]
    assert staging == []
    assert archive == ["active-c", "staged-b"]


def test_find_any_and_unknown(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")
    library.create_draft("Finder Practice", "body")
    assert library.find_any("finder-practice") is not None
    assert library.find_any("does-not-exist") is None


def test_promote_unknown_raises(tmp_path):
    library = VersionedSkillLibrary(tmp_path / "skills")
    with pytest.raises(FileNotFoundError):
        library.promote("missing-skill")
