"""Versioned skill library for the learning/evaluation loop (Path B).

Skills live under ``skills/<slug>/SKILL.md`` with YAML-like frontmatter
(``name``, ``version``, ``related``, ``description``).  Drafts are written to
``skills/staging/<slug>/SKILL.md`` first and only become active after a
statistically significant regression check.  Deprecated skills are moved to
``skills/.archive/<slug>/SKILL.md`` instead of being deleted.

The frontmatter parser is deliberately minimal (no PyYAML dependency); it
supports scalar values and ``- item`` list blocks for ``related``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ACTIVE_DIR_NAME = "active"
STAGING_DIR_NAME = "staging"
ARCHIVE_DIR_NAME = ".archive"
SKILL_FILE_NAME = "SKILL.md"


@dataclass(frozen=True)
class SkillRecord:
    """A skill file on disk with its parsed metadata."""

    name: str
    slug: str
    version: int
    description: str
    related: list[str]
    path: Path
    state: str  # "active" | "staging" | "archive"
    body: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "slug": self.slug,
            "version": self.version,
            "description": self.description,
            "related": list(self.related),
            "path": str(self.path),
            "state": self.state,
        }


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "skill"


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse ``---``-delimited frontmatter; returns ``(meta, body)``.

    Unknown keys are kept as strings; ``version`` is coerced to int when
    possible; ``related`` accepts either a comma-separated scalar or a list
    block of ``- item`` lines.
    """
    text = text.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end]
    body = text[end + 4 :].lstrip("\n")
    meta: dict[str, Any] = {}
    current_key: str | None = None
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            if current_key is not None:
                items = meta.setdefault(current_key, [])
                if not isinstance(items, list):
                    items = [items] if str(items).strip() else []
                    meta[current_key] = items
                items.append(stripped[2:].strip())
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if key == "version":
            try:
                meta[key] = int(value)
            except ValueError:
                meta[key] = value
        else:
            meta[key] = value
        current_key = key
    return meta, body


def render_skill_document(meta: dict[str, Any], body: str) -> str:
    """Render frontmatter + body back into a SKILL.md string."""
    lines = ["---"]
    for key, value in meta.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {item}" for item in value if str(item).strip())
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    body = body.strip()
    if body:
        lines.append(body)
    return "\n".join(lines).rstrip() + "\n"


def default_meta(name: str, version: int = 1, description: str = "", related: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": name.strip() or slugify(name),
        "version": max(1, int(version)),
        "description": description,
        "related": list(related or []),
    }


class VersionedSkillLibrary:
    """File-backed versioned skill library with staging and archive states.

    Directory layout under ``root`` (default ``skills``)::

        skills/<slug>/SKILL.md        active skill (versioned)
        skills/staging/<slug>/SKILL.md  draft, not yet effective
        skills/.archive/<slug>/SKILL.md deprecated/rejected skill (never deleted)
    """

    def __init__(self, root: str | Path = "skills") -> None:
        self.root = Path(root)
        self.active_dir = self.root / ACTIVE_DIR_NAME
        self.staging_dir = self.root / STAGING_DIR_NAME
        self.archive_dir = self.root / ARCHIVE_DIR_NAME
        for directory in (self.active_dir, self.staging_dir, self.archive_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # -- read ----------------------------------------------------------------

    def list_active(self) -> list[SkillRecord]:
        return self._list_state(self.active_dir, "active")

    def list_staging(self) -> list[SkillRecord]:
        return self._list_state(self.staging_dir, "staging")

    def list_archive(self) -> list[SkillRecord]:
        return self._list_state(self.archive_dir, "archive")

    def find_active(self, name_or_slug: str) -> SkillRecord | None:
        return self._find(self.active_dir, "active", name_or_slug)

    def find_staging(self, name_or_slug: str) -> SkillRecord | None:
        return self._find(self.staging_dir, "staging", name_or_slug)

    def find_archive(self, name_or_slug: str) -> SkillRecord | None:
        return self._find(self.archive_dir, "archive", name_or_slug)

    def find_any(self, name_or_slug: str) -> SkillRecord | None:
        for finder in (self.find_active, self.find_staging, self.find_archive):
            record = finder(name_or_slug)
            if record is not None:
                return record
        return None

    # -- write ---------------------------------------------------------------

    def create_draft(
        self,
        name: str,
        body: str,
        description: str = "",
        related: list[str] | None = None,
        version: int = 1,
        source_bad_case: str = "",
    ) -> SkillRecord:
        """Write a draft to staging.  Version defaults to 1; if a draft with
        the same slug already exists its version is bumped by one instead."""
        slug = slugify(name)
        existing = self.find_staging(slug)
        if existing is not None:
            version = existing.version + 1
        meta = default_meta(name=name, version=version, description=description, related=related)
        if source_bad_case:
            meta["source_bad_case"] = source_bad_case
        path = self.staging_dir / slug / SKILL_FILE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_skill_document(meta, body), encoding="utf-8")
        return self._read(path, "staging")

    def revise_draft(self, name_or_slug: str, body: str, description: str | None = None) -> SkillRecord:
        """Increment the draft version and rewrite its body (staging only)."""
        draft = self.find_staging(name_or_slug)
        if draft is None:
            raise FileNotFoundError(f"No staging skill draft found for {name_or_slug}")
        meta, _ = parse_frontmatter(draft.path.read_text(encoding="utf-8"))
        meta["version"] = int(meta.get("version", 1)) + 1
        if description is not None:
            meta["description"] = description
        draft.path.write_text(render_skill_document(meta, body), encoding="utf-8")
        return self._read(draft.path, "staging")

    def promote(self, name_or_slug: str) -> SkillRecord:
        """Move a staging draft into active, bumping the version over any
        existing active version.  The staging file is removed afterwards."""
        draft = self.find_staging(name_or_slug)
        if draft is None:
            raise FileNotFoundError(f"No staging skill draft found for {name_or_slug}")
        active = self.find_active(draft.slug)
        version = (active.version + 1) if active is not None else draft.version
        meta, body = parse_frontmatter(draft.path.read_text(encoding="utf-8"))
        meta["name"] = draft.name
        meta["version"] = version
        target = self.active_dir / draft.slug / SKILL_FILE_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_skill_document(meta, body), encoding="utf-8")
        draft.path.unlink()
        self._prune_empty_dir(draft.path.parent)
        return self._read(target, "active")

    def reject(self, name_or_slug: str, reason: str = "") -> SkillRecord:
        """Reject a staging draft: move it into the archive (never deleted)."""
        draft = self.find_staging(name_or_slug)
        if draft is None:
            raise FileNotFoundError(f"No staging skill draft found for {name_or_slug}")
        return self._move_to_archive(draft, reason=reason, state="rejected")

    def archive(self, name_or_slug: str, reason: str = "") -> SkillRecord:
        """Deprecate an active skill: move it into the archive (never deleted)."""
        active = self.find_active(name_or_slug)
        if active is None:
            raise FileNotFoundError(f"No active skill found for {name_or_slug}")
        return self._move_to_archive(active, reason=reason, state="archived")

    # -- helpers -------------------------------------------------------------

    def _move_to_archive(self, record: SkillRecord, reason: str, state: str) -> SkillRecord:
        meta, body = parse_frontmatter(record.path.read_text(encoding="utf-8"))
        meta["archived"] = state
        if reason:
            meta["archive_reason"] = reason
        target = self.archive_dir / record.slug / SKILL_FILE_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_skill_document(meta, body), encoding="utf-8")
        record.path.unlink()
        self._prune_empty_dir(record.path.parent)
        return self._read(target, "archive")

    def _list_state(self, directory: Path, state: str) -> list[SkillRecord]:
        records: list[SkillRecord] = []
        if not directory.exists():
            return records
        for skill_dir in sorted(directory.iterdir()):
            if not skill_dir.is_dir():
                continue
            path = skill_dir / SKILL_FILE_NAME
            if path.exists():
                records.append(self._read(path, state))
        return sorted(records, key=lambda record: (record.slug, record.version))

    def _find(self, directory: Path, state: str, name_or_slug: str) -> SkillRecord | None:
        slug = slugify(name_or_slug)
        for record in self._list_state(directory, state):
            if name_or_slug in {record.slug, record.name, str(record.path)} or slug in {record.slug, record.name}:
                return record
        return None

    @staticmethod
    def _read(path: Path, state: str) -> SkillRecord:
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        name = str(meta.get("name") or path.parent.name or "skill")
        version = int(meta.get("version", 1) or 1)
        related = meta.get("related", [])
        if not isinstance(related, list):
            related = [str(item) for item in str(related).split(",") if str(item).strip()]
        description = str(meta.get("description", ""))
        return SkillRecord(
            name=name,
            slug=path.parent.name,
            version=version,
            description=description,
            related=[str(item) for item in related],
            path=path,
            state=state,
            body=body,
        )

    @staticmethod
    def _prune_empty_dir(directory: Path) -> None:
        try:
            directory.rmdir()
        except OSError:
            pass
