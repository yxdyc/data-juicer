# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for dialog LLM mappers (intent / topic / sentiment / intensity)."""

from __future__ import annotations

from typing import Tuple


def clip_text_for_dialog_prompt(
    text: str,
    max_chars: int,
    note: str = "truncated",
) -> str:
    """Truncate long ``text`` for API prompts when ``max_chars`` > 0.

    Agent traces often concatenate tool outputs into ``response``; formatter
    limits elsewhere do not apply to these mappers' ``history_key`` payloads.
    """
    if max_chars is None or max_chars <= 0:
        return text
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    suffix = f"\n…[{note}]…"
    take = max_chars - len(suffix)
    if take <= 0:
        return suffix.strip()
    return text[:take] + suffix


def clip_query_response_pair(
    q: object,
    r: object,
    max_query_chars: int,
    max_response_chars: int,
) -> Tuple[str, str]:
    qs = "" if q is None else str(q)
    rs = "" if r is None else str(r)
    return (
        clip_text_for_dialog_prompt(qs, max_query_chars, "query truncated"),
        clip_text_for_dialog_prompt(
            rs,
            max_response_chars,
            "response truncated",
        ),
    )
