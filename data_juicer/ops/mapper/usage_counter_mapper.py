# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Extract token usage from agent response (choices, usage, response_metadata).
# Multi-format: OpenAI, Anthropic, generic usage objects.

from typing import Any, List

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = "usage_counter_mapper"


def _get_usage_from_obj(obj: Any) -> dict:
    """Extract prompt_tokens, completion_tokens, total_tokens from a dict.
    Supports: nested obj.usage / obj.usage_metadata, or obj as the usage dict.
    """
    if not isinstance(obj, dict):
        return {}
    usage = obj.get("usage") or obj.get("usage_metadata") or {}
    if not isinstance(usage, dict):
        usage = {}
    # Top-level usage (e.g. response_usage with prompt_tokens directly)
    if usage and (usage.get("prompt_tokens") is not None or usage.get("completion_tokens") is not None):
        pass
    elif obj.get("prompt_tokens") is not None or obj.get("completion_tokens") is not None:
        usage = obj
    p = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
    c = usage.get("completion_tokens") or usage.get("output_tokens", 0)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": usage.get("total_tokens")}


def _aggregate_usage(usages: List[dict]) -> tuple:
    """Sum prompt/completion; total if present else prompt+completion."""
    p = sum(u.get("prompt_tokens") or 0 for u in usages)
    c = sum(u.get("completion_tokens") or 0 for u in usages)
    totals = [u.get("total_tokens") for u in usages if u.get("total_tokens") is not None]
    t = totals[0] if totals else None
    if t is None and (p or c):
        t = p + c
    return p, c, t


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class UsageCounterMapper(Mapper):
    """Write token usage to meta from choices/usage (OpenAI/Anthropic-style)."""

    def __init__(
        self,
        choices_key: str = "choices",
        usage_key: str = "usage",
        response_metadata_key: str = "response_metadata",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.choices_key = choices_key
        self.usage_key = usage_key
        self.response_metadata_key = response_metadata_key

    def process_single(self, sample):
        usages = []

        # Top-level usage (e.g. usage / response_usage)
        if self.usage_key in sample:
            u = _get_usage_from_obj(sample.get(self.usage_key)) or _get_usage_from_obj(sample)
            if u:
                usages.append(u)

        # response_metadata.usage
        meta = sample.get(self.response_metadata_key) or {}
        if isinstance(meta, dict):
            u = _get_usage_from_obj(meta)
            if u:
                usages.append(u)

        # choices[].usage or choices[].message.usage
        choices = sample.get(self.choices_key) or []
        if isinstance(choices, list):
            for c in choices:
                if not isinstance(c, dict):
                    continue
                u = _get_usage_from_obj(c)
                if u:
                    usages.append(u)
                msg = c.get("message") or c.get("delta")
                if isinstance(msg, dict):
                    u = _get_usage_from_obj(msg)
                    if u:
                        usages.append(u)

        if Fields.meta not in sample:
            sample[Fields.meta] = {}
        meta = sample[Fields.meta]
        p, c, t = _aggregate_usage(usages)
        meta[MetaKeys.prompt_tokens] = p
        meta[MetaKeys.completion_tokens] = c
        meta[MetaKeys.total_tokens] = t
        return sample
