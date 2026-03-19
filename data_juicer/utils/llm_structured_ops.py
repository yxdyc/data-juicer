"""LLM structured ops: user-configurable extract/condition helpers.

Shared by llm_extract_mapper, llm_condition_filter; reusable for DataFrame/SQL/DB
by adapting input to (text, schema/condition). Aligns with existing llm_* naming.
"""

import json
import logging
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class InferenceStrategy(Enum):
    DIRECT = "direct"  # Fast, no reasoning
    COT = "cot"  # Chain-of-Thought reasoning
    FEW_SHOT = "few_shot"  # Direct with examples
    COT_SHOT = "cot_shot"  # Reasoning with examples


# ---- Default prompts for user-configurable schema/condition ----

DEFAULT_EXTRACT_SYSTEM = (
    "You are a precise information extraction assistant. "
    "Given the input text, extract the requested fields and return ONLY a valid "
    "JSON object. Use the exact output keys provided. "
    "If a value cannot be determined, use null."
)  # noqa: E501

DEFAULT_EXTRACT_USER_TEMPLATE = (
    "# Input\n{input_text}\n\n"
    "# Instructions (output keys and what to extract)\n"
    "{output_instructions}\n\n"
    "# Output\nReturn a single JSON object with the above keys. No explanation."
)

DEFAULT_EXTRACT_COT_TEMPLATE = (
    "# Input\n{input_text}\n\n"
    "# Instructions (output keys and what to extract)\n"
    "{output_instructions}\n\n"
    "# Output\nFirst, think step by step about how to extract each field. "
    "Then return a single JSON object with the above keys."
)

DEFAULT_EXTRACT_FEW_SHOT_TEMPLATE = (
    "# Examples\n{examples}\n\n"
    "# Input\n{input_text}\n\n"
    "# Instructions (output keys and what to extract)\n"
    "{output_instructions}\n\n"
    "# Output\nReturn a single JSON object with the above keys. No explanation."
)

DEFAULT_EXTRACT_COT_SHOT_TEMPLATE = (
    "# Examples\n{examples}\n\n"
    "# Input\n{input_text}\n\n"
    "# Instructions (output keys and what to extract)\n"
    "{output_instructions}\n\n"
    "# Output\nFirst, think step by step about how to extract each field based on the examples. "
    "Then return a single JSON object with the above keys."
)

DEFAULT_CONDITION_SYSTEM = "You are a binary classifier. Answer only 'yes' or 'no', nothing else."

DEFAULT_CONDITION_USER_TEMPLATE = (
    "# Text\n{text}\n\n# Condition\n{condition}\n\n" "Does the text satisfy the condition? Answer yes or no."
)  # noqa: E501

DEFAULT_CONDITION_COT_TEMPLATE = (
    "# Text\n{text}\n\n# Condition\n{condition}\n\n"
    "Think step by step: analyze the text and condition, then determine if the text satisfies the condition. "
    "Answer yes or no."
)

DEFAULT_CONDITION_FEW_SHOT_TEMPLATE = (
    "# Examples\n{examples}\n\n"
    "# Text\n{text}\n\n# Condition\n{condition}\n\n"
    "Does the text satisfy the condition? Answer yes or no."
)

DEFAULT_CONDITION_COT_SHOT_TEMPLATE = (
    "# Examples\n{examples}\n\n"
    "# Text\n{text}\n\n# Condition\n{condition}\n\n"
    "Think step by step: analyze the text and condition based on the examples, "
    "then determine if the text satisfies the condition. Answer yes or no."
)


def get_extract_prompt(
    input_text: str,
    output_schema: Dict[str, str],
    knowledge_grounding: Optional[str] = None,
    strategy: Optional[InferenceStrategy] = None,
    examples: Optional[str] = None,
) -> str:
    """Build user prompt for extraction. output_schema: {key: instruction}."""
    instructions = "\n".join(f"- {k}: {v}" for k, v in output_schema.items())
    strategy = strategy or InferenceStrategy.DIRECT
    
    # Map strategies to their corresponding templates
    template_map = {
        InferenceStrategy.DIRECT: lambda: DEFAULT_EXTRACT_USER_TEMPLATE.format(
            input_text=input_text,
            output_instructions=instructions,
        ),
        InferenceStrategy.COT: lambda: DEFAULT_EXTRACT_COT_TEMPLATE.format(
            input_text=input_text,
            output_instructions=instructions,
        ),
        InferenceStrategy.FEW_SHOT: lambda: DEFAULT_EXTRACT_FEW_SHOT_TEMPLATE.format(
            examples=examples or "",
            input_text=input_text,
            output_instructions=instructions,
        ),
        InferenceStrategy.COT_SHOT: lambda: DEFAULT_EXTRACT_COT_SHOT_TEMPLATE.format(
            examples=examples or "",
            input_text=input_text,
            output_instructions=instructions,
        ),
    }
    
    body = template_map.get(strategy, template_map[InferenceStrategy.DIRECT])()
    
    if knowledge_grounding:
        body = f"# Background / grounding\n{knowledge_grounding}\n\n" + body
    return body


def get_condition_prompt(
    text: str,
    condition: str,
    knowledge_grounding: Optional[str] = None,
    strategy: Optional[InferenceStrategy] = None,
    examples: Optional[str] = None,
) -> str:
    """Build user prompt for LLM condition filter (yes/no)."""
    strategy = strategy or InferenceStrategy.DIRECT
    
    # Map strategies to their corresponding templates
    template_map = {
        InferenceStrategy.DIRECT: lambda: DEFAULT_CONDITION_USER_TEMPLATE.format(
            text=text, condition=condition
        ),
        InferenceStrategy.COT: lambda: DEFAULT_CONDITION_COT_TEMPLATE.format(
            text=text, condition=condition
        ),
        InferenceStrategy.FEW_SHOT: lambda: DEFAULT_CONDITION_FEW_SHOT_TEMPLATE.format(
            examples=examples or "",
            text=text,
            condition=condition,
        ),
        InferenceStrategy.COT_SHOT: lambda: DEFAULT_CONDITION_COT_SHOT_TEMPLATE.format(
            examples=examples or "",
            text=text,
            condition=condition,
        ),
    }
    
    body = template_map.get(strategy, template_map[InferenceStrategy.DIRECT])()
    
    if knowledge_grounding:
        body = f"# Background\n{knowledge_grounding}\n\n" + body
    return body


def call_llm_sync(
    model: Any,
    messages: list,
    *,
    enable_vllm: bool = False,
    is_hf_model: bool = False,
    sampling_params: Optional[Dict] = None,
) -> str:
    """Call LLM synchronously; return content. Compatible with DJ model_utils."""
    sampling_params = sampling_params or {}
    try:
        if enable_vllm:
            from data_juicer.utils.lazy_loader import LazyLoader

            vllm_mod = LazyLoader("vllm")
            sp = vllm_mod.SamplingParams(**sampling_params) if isinstance(sampling_params, dict) else sampling_params
            response = model.chat(messages, sp)
            return (response[0].outputs[0].text or "").strip()
        if is_hf_model:
            out = model(messages, return_full_text=False, **sampling_params)
            return (out[0].get("generated_text") or "").strip()
        return (model(messages, **sampling_params) or "").strip()
    except Exception as e:
        logger.warning("LLM structured ops call failed: %s", e)
        return ""


def extract_one(
    input_text: str,
    output_schema: Dict[str, str],
    model: Any,
    *,
    system_prompt: Optional[str] = None,
    knowledge_grounding: Optional[str] = None,
    strategy: Optional[InferenceStrategy] = None,
    examples: Optional[str] = None,
    enable_vllm: bool = False,
    is_hf_model: bool = False,
    sampling_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Extract structured fields from input_text using the model. Returns dict."""
    user_content = get_extract_prompt(
        input_text, output_schema, knowledge_grounding, strategy, examples
    )
    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_EXTRACT_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    raw = call_llm_sync(
        model,
        messages,
        enable_vllm=enable_vllm,
        is_hf_model=is_hf_model,
        sampling_params=sampling_params,
    )
    out_keys = list(output_schema.keys())
    if not raw:
        return {k: None for k in out_keys}
    raw = raw.strip()
    start_obj = raw.find("{")
    end_obj = raw.rfind("}") + 1
    start_arr = raw.find("[")
    end_arr = raw.rfind("]") + 1
    has_obj = start_obj >= 0 and end_obj > start_obj
    has_arr = start_arr >= 0 and end_arr > start_arr
    try:
        if has_obj and (not has_arr or start_obj <= start_arr):
            js = json.loads(raw[start_obj:end_obj])
            if isinstance(js, dict):
                return {k: js.get(k) for k in out_keys}
        if has_arr and len(out_keys) == 1:
            js = json.loads(raw[start_arr:end_arr])
            if isinstance(js, list):
                return {out_keys[0]: js}
    except json.JSONDecodeError:
        pass
    return {k: None for k in out_keys}


def condition_filter_one(
    text: str,
    condition: str,
    model: Any,
    *,
    knowledge_grounding: Optional[str] = None,
    strategy: Optional[InferenceStrategy] = None,
    examples: Optional[str] = None,
    enable_vllm: bool = False,
    is_hf_model: bool = False,
    sampling_params: Optional[Dict] = None,
) -> bool:
    """True iff the model says the text satisfies the condition (yes/no)."""
    user_content = get_condition_prompt(text, condition, knowledge_grounding, strategy, examples)
    messages = [
        {"role": "system", "content": DEFAULT_CONDITION_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    raw = call_llm_sync(
        model,
        messages,
        enable_vllm=enable_vllm,
        is_hf_model=is_hf_model,
        sampling_params=sampling_params,
    )
    if not raw:
        return False
    low = raw.strip().lower()
    return low.startswith("yes") or low == "y"
