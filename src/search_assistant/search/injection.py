"""Prompt-injection screening for retrieved web content.

Retrieved snippets and titles are untrusted data: a hostile page can embed
instructions ("ignore previous instructions", "泄露你的 system prompt") that
compete with the agent's own prompt.  This module detects the common injection
patterns before content reaches an LLM context.  Matched sources are dropped
from retrieval output (the URL is still recorded in provider events), and the
runtime prompts additionally state that retrieved content is data, not
instructions.
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # English: instruction-override and prompt-extraction language.
    re.compile(r"ignore (?:all |any )?(?:previous|prior|above|earlier) instructions", re.IGNORECASE),
    re.compile(r"disregard (?:all |any )?(?:previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"do not follow (?:any )?(?:previous|prior|above) (?:instructions|rules|prompts)", re.IGNORECASE),
    re.compile(r"(?:you are|act as|pretend to be) (?:now )?(?:an? )?(?:chatgpt|gpt|assistant|agent)", re.IGNORECASE),
    re.compile(r"(?:new|updated|override|redefine) (?:system )?(?:prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"print (?:your|the) (?:system )?(?:prompt|instructions)", re.IGNORECASE),
    re.compile(r"reveal (?:your|the|hidden) (?:system )?(?:prompt|instructions|chain[ -]of[ -]thought)", re.IGNORECASE),
    re.compile(r"tell me (?:your|the) (?:system )?instructions", re.IGNORECASE),
    re.compile(r"repeat (?:your|the) (?:system )?(?:prompt|instructions)", re.IGNORECASE),
    re.compile(r"\bignore previous\b", re.IGNORECASE),
    # Chinese: instruction-override and prompt-extraction language.
    re.compile(r"忽略(?:掉|所有)?(?:之前|以上|前面)?的?(?:指令|提示|要求|规则)", re.IGNORECASE),
    re.compile(r"无视(?:之前|以上|前面)?的?(?:指令|提示|要求|规则)", re.IGNORECASE),
    re.compile(r"不要(?:遵循|遵守|理会)(?:之前|以上|前面)?的?(?:指令|提示|要求|规则)", re.IGNORECASE),
    re.compile(r"(?:你现在|从现在起)(?:必须)?(?:扮演|作为|是)(?:一个)?(?:系统|我的|助手)"),
    re.compile(r"泄露(?:你的|系统)(?:提示词|指令|system ?prompt)"),
    re.compile(r"输出(?:你的|系统)(?:提示词|指令|system ?prompt)"),
    re.compile(r"打印(?:你的|系统)(?:提示词|指令)"),
    re.compile(r"忽略(?:系统)?提示词"),
    re.compile(r"假装(?:你是|你是我的)"),
)


def injection_flag(text: str) -> str | None:
    """Return the first matched injection pattern, or None when clean."""
    lowered = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(lowered):
            return pattern.pattern[:60]
    return None


def strip_injected_sources(sources: list[object]) -> tuple[list[object], list[object]]:
    """Split retrieved sources into (clean, flagged).

    Both the title and the snippet are screened; a match on either drops the
    source.  The flagged list keeps the original objects so callers can record
    provider events without losing the audit trail.
    """
    clean: list[object] = []
    flagged: list[object] = []
    for source in sources:
        title = str(getattr(source, "title", "") or "")
        snippet = str(getattr(source, "snippet", "") or "")
        if injection_flag(title) is not None or injection_flag(snippet) is not None:
            flagged.append(source)
        else:
            clean.append(source)
    return clean, flagged
