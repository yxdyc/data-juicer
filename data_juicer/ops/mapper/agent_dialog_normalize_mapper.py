# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Normalize agent interaction format (messages + choices) to DJ fields for
# dialog/text ops. Supports multi-platform, multi-agent tool formats.

import re
from typing import Any, List, Optional, Tuple

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = "agent_dialog_normalize_mapper"

# Default labels for flattened dialog (i18n / multi-platform)
DEFAULT_USER_LABEL = "用户"
DEFAULT_ASSISTANT_LABEL = "助手"


def _content_to_text(content: Any) -> str:
    """Extract plain text from message.content.
    Supports: str, list of {type, text} (OpenAI multimodal), list of str.
    """  # noqa: E501
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append((block.get("text") or "").strip())
                elif block.get("type") == "input_text" or "text" in block:
                    parts.append((block.get("text") or "").strip())
            elif isinstance(block, str):
                parts.append(block.strip())
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def _get_tool_name_from_call(tc: dict) -> Optional[str]:
    """Get tool/function name from one tool_call item.
    Supports OpenAI (function.name), Anthropic (name), generic.
    """
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function") or tc.get("function_call")
    if isinstance(fn, dict) and fn.get("name"):
        return fn["name"]
    if tc.get("name"):
        return tc["name"]
    return None


def _tool_calls_summary(
    tool_calls: Any,
    max_names: int = 10,
) -> str:
    """Summarize tool_calls for history; multi-format."""
    if not tool_calls or not isinstance(tool_calls, list):
        return ""
    names = []
    for tc in tool_calls:
        n = _get_tool_name_from_call(tc)
        if n and n not in names:
            names.append(n)
    if not names:
        return ""
    display = names[:max_names]
    if len(names) > max_names:
        display.append(f"...+{len(names) - max_names}")
    return "[Tool calls: " + ", ".join(display) + "]"


def _extract_tool_types(messages: List[dict]) -> List[str]:
    """Collect unique tool/function names from messages (multi-format)."""
    out = []
    seen = set()
    for m in messages:
        for tc in m.get("tool_calls") or m.get("tool_use") or []:
            name = _get_tool_name_from_call(tc)
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


# Skill patterns: .../skill_name/SKILL.md, ## skill_name
_SKILL_PATTERNS = [
    re.compile(r"[/\\](\w+)[/\\]SKILL\.md", re.IGNORECASE),
    re.compile(r"##\s+(\w+)\s*\n"),
    re.compile(r"Check\s+[\"'].*?(\w+)/SKILL\.md", re.IGNORECASE),  # noqa: E501
]


def _extract_skill_types(messages: List[dict]) -> List[str]:
    """Extract skill names from system/content (multi-pattern)."""
    out = []
    seen = set()
    for m in messages:
        text = _content_to_text(m.get("content"))
        for pat in _SKILL_PATTERNS:
            for mo in pat.finditer(text):
                name = (mo.group(1) or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    out.append(name)
    return out


def _messages_to_history(
    messages: List[dict],
    include_system_in_first_user: bool = False,
) -> List[Tuple[str, str]]:
    """Convert messages to [(query, response), ...]. User/assistant only."""
    history = []
    pending_system = []

    for m in messages:
        role = (m.get("role") or "").lower()
        content = _content_to_text(m.get("content"))
        tool_calls = m.get("tool_calls") or m.get("tool_use") or []

        if role == "system":
            if include_system_in_first_user and content:
                pending_system.append(content)
            continue
        if role == "user":
            if pending_system:
                content = "\n\n".join(pending_system + [content]).strip()
                pending_system = []
            history.append((content, ""))
            continue
        if role == "assistant":
            if not content and tool_calls:
                content = _tool_calls_summary(tool_calls)
            if history:
                history[-1] = (history[-1][0], content)
            else:
                history.append(("", content))
            continue
        if role == "tool":
            # Append tool result to last assistant response for context
            if history and history[-1][1]:
                history[-1] = (
                    history[-1][0],
                    history[-1][1] + "\n[Tool result]\n" + content[:500],
                )
            continue
    return history


def _choices_to_text(choices: Any) -> str:
    """Extract reply text from choices (OpenAI / Anthropic / generic)."""
    if not choices or not isinstance(choices, list):
        return ""
    for c in choices:
        if not isinstance(c, dict):
            continue
        msg = c.get("message") or c.get("delta") or c
        text = msg.get("content")
        if text is None:
            continue
        if isinstance(text, str) and text.strip():
            return text.strip()
        if isinstance(text, list):
            t = _content_to_text(text)
            if t:
                return t
    return ""


def _flatten_history_to_text(
    history: List[Tuple[str, str]],
    user_label: str = DEFAULT_USER_LABEL,
    assistant_label: str = DEFAULT_ASSISTANT_LABEL,
) -> str:
    """Flatten history to one text for text-based ops."""
    lines = []
    for q, r in history:
        if q:
            lines.append(f"{user_label}：{q}")
        if r:
            lines.append(f"{assistant_label}：{r}")
    return "\n\n".join(lines)


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentDialogNormalizeMapper(Mapper):
    """Normalize agent format (messages + choices) to DJ fields.

    Outputs: text, dialog_history, query, response; optionally meta tags
    agent_tool_types, agent_skill_types, agent_turn_count.
    Supports multi-format tool_calls (e.g. tool_calls[].function.name as in
    OpenAI / demos/local/demo-agent-data-content.json) and configurable
    user/assistant labels.
    """

    def __init__(
        self,
        messages_key: str = "messages",
        choices_key: str = "choices",
        text_key: str = "text",
        history_key: str = "dialog_history",
        query_key: str = "query",
        response_key: str = "response",
        extract_tool_skill_tags: bool = True,
        include_system_in_first_user: bool = False,
        user_label: str = DEFAULT_USER_LABEL,
        assistant_label: str = DEFAULT_ASSISTANT_LABEL,
        **kwargs,
    ):
        super().__init__(text_key=text_key, **kwargs)
        self.messages_key = messages_key
        self.choices_key = choices_key
        self.history_key = history_key
        self.query_key = query_key
        self.response_key = response_key
        self.extract_tool_skill_tags = extract_tool_skill_tags
        self.include_system_in_first_user = include_system_in_first_user
        self.user_label = user_label
        self.assistant_label = assistant_label

    def process_single(self, sample):
        messages = sample.get(self.messages_key) or []
        choices = sample.get(self.choices_key) or []

        if not isinstance(messages, list):
            messages = []

        history = _messages_to_history(
            messages,
            include_system_in_first_user=self.include_system_in_first_user,
        )
        flat_text = _flatten_history_to_text(
            history,
            user_label=self.user_label,
            assistant_label=self.assistant_label,
        )
        last_query = history[-1][0] if history else ""
        last_response = history[-1][1] if history else ""
        if not last_response and choices:
            last_response = _choices_to_text(choices)

        sample[self.text_key] = flat_text
        sample[self.history_key] = history
        sample[self.query_key] = last_query
        sample[self.response_key] = last_response

        if Fields.meta not in sample:
            sample[Fields.meta] = {}
        meta = sample[Fields.meta]
        meta[MetaKeys.agent_turn_count] = len(history)
        if self.extract_tool_skill_tags:
            meta[MetaKeys.agent_tool_types] = _extract_tool_types(messages)
            meta[MetaKeys.agent_skill_types] = _extract_skill_types(messages)

        return sample
