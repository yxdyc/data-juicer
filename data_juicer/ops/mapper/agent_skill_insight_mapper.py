# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LLM-based summarization of agent_tool_types + agent_skill_types into
# high-level capability insights (agent_skill_insights).

from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "agent_skill_insight_mapper"

DEFAULT_SYSTEM_PROMPT = (
    "你是一个对话能力分析助手。根据给定的工具列表和技能列表，归纳为3～5个高层能力类别。"
    "每个类别用简短中文标签（2～6字），例如：文件与编辑、搜索与记忆、执行与调度、沟通协作、信息检索。"
    "仅输出逗号分隔的标签，不要编号、不要解释、不要换行。"
)


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentSkillInsightMapper(Mapper):
    """Summarize agent_tool_types and agent_skill_types into insights via LLM.

    Reads meta[agent_tool_types] and meta[agent_skill_types] (from
    agent_dialog_normalize_mapper), calls API for 3～5 capability categories,
    stores in meta[agent_skill_insights]. Use after normalize for better
    skill tagging than raw regex-extracted names.
    """

    def __init__(
        self,
        api_model: str = "gpt-4o",
        *,
        tool_types_key: str = MetaKeys.agent_tool_types,
        skill_types_key: str = MetaKeys.agent_skill_types,
        insights_key: str = MetaKeys.agent_skill_insights,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        try_num: PositiveInt = 2,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tool_types_key = tool_types_key
        self.skill_types_key = skill_types_key
        self.insights_key = insights_key
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.try_num = try_num
        self.sampling_params = sampling_params or {}
        self.model_key = prepare_model(
            model_type="api",
            model=api_model,
            endpoint=api_endpoint,
            response_path=response_path,
            **model_params,
        )

    def process_single(self, sample, rank=None):
        meta = sample.get(Fields.meta)
        if not isinstance(meta, dict):
            return sample
        if self.insights_key in meta:
            return sample

        tools = meta.get(self.tool_types_key) or []
        skills = meta.get(self.skill_types_key) or []
        if not isinstance(tools, list):
            tools = [tools] if tools else []
        if not isinstance(skills, list):
            skills = [skills] if skills else []

        if not tools and not skills:
            meta[self.insights_key] = []
            return sample

        tools_str = "、".join(str(x) for x in tools[:30])
        skills_str = "、".join(str(x) for x in skills[:30])
        user_content = f"工具：{tools_str}\n技能：{skills_str}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        raw = ""
        for _ in range(self.try_num):
            try:
                client = get_model(self.model_key, rank=rank)
                raw = client(messages, **self.sampling_params)
                if raw and isinstance(raw, str) and raw.strip():
                    break
            except Exception as e:
                logger.warning("agent_skill_insight_mapper: %s", e)

        if not raw or not isinstance(raw, str):
            meta[self.insights_key] = []
            return sample
        # Parse comma-separated labels, strip, dedupe order-preserving
        labels = [s.strip() for s in raw.split(",") if s.strip()]
        meta[self.insights_key] = list(dict.fromkeys(labels))
        return sample
