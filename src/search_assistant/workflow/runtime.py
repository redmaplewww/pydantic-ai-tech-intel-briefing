"""Compatibility imports for the runtime package.

New code should import from ``search_assistant.runtime``.  This module keeps the
old import path for runtime protocols and Pydantic AI provider adapters; removed
Agent Framework adapters are not re-exported.
"""

from search_assistant.runtime import (
    AgentRuntime,
    BriefingRuntime,
    DeepSeekChatRuntime,
    FakeAgentRuntime,
    GLMPydanticAIRuntime,
    ModelRuntimeError,
    PydanticAIModelRuntime,
    TextModelRuntime,
)
from search_assistant.runtime.pydantic_ai import (
    _parse_answer_review,
    _parse_briefing_synthesis,
    _parse_search_query_plan,
    runtime_from_settings,
)

__all__ = [
    "AgentRuntime",
    "BriefingRuntime",
    "DeepSeekChatRuntime",
    "FakeAgentRuntime",
    "GLMPydanticAIRuntime",
    "ModelRuntimeError",
    "PydanticAIModelRuntime",
    "TextModelRuntime",
    "_parse_answer_review",
    "_parse_briefing_synthesis",
    "_parse_search_query_plan",
    "runtime_from_settings",
]
