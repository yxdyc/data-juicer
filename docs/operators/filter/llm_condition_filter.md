# llm_condition_filter

Filter by user-given natural language condition (LLM yes/no). Part of the llm_* semantic ops family.

This operator uses an LLM to decide whether each sample satisfies a free-form condition string (e.g. "contains a question", "is in formal tone"). It writes the result to `stats.llm_condition_filter_result` and keeps samples where the result is True. Token/cost usage is recorded in `stats.llm_semantic_usage`.

按用户给定的自然语言条件进行过滤（LLM 回答是/否）；满足条件的样本保留，并在 stats 中记录用量。

Type 算子类型: **filter**

Tags 标签: gpu, vllm, hf, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `text_key` | str | `'text'` | Sample key for the text to evaluate. |
| `condition` | str | `''` | Natural language condition (e.g. "contains a question"). |
| `api_or_hf_model` | str | `'gpt-4o'` | Model name. |
| `knowledge_grounding_key` | str, optional | `None` | Optional sample key for per-sample grounding. |
| `knowledge_grounding_fixed` | str, optional | `None` | Optional fixed grounding string. |
| `is_hf_model` | bool | `False` | If true, use HuggingFace. |
| `enable_vllm` | bool | `False` | If true, use vLLM. |
| `api_endpoint` | str, optional | `None` | API endpoint. |
| `response_path` | str, optional | `None` | Path to extract content from API response. |
| `try_num` | int | `3` | Retries on API/parse failure; treat as False after all fail. |
| `model_params` | dict | `{}` | Model init params. |
| `sampling_params` | dict | `{}` | Sampling params. |

## 🌐 DashScope / OpenAI-compatible 环境变量

与 `llm_extract_mapper` 相同：设置 `OPENAI_BASE_URL`（或 `OPENAI_API_URL`）为 DashScope 兼容地址，以及 `OPENAI_API_KEY` 或 `DASHSCOPE_API_KEY`。检测到 DashScope 兼容端点时，默认的 `gpt-4o` 会映射为 `qwen-plus`，可用 `DASHSCOPE_DEFAULT_MODEL` 覆盖。

## 📊 Cost / usage 成本与用量

Each sample gets `stats.llm_semantic_usage` with:
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `cost_estimate` (optional)

## 🔗 Related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/llm_condition_filter.py)
- [semantic ops 语义工具](../../../data_juicer/utils/llm_semantic_ops.py)
- [unit test 单元测试](../../../tests/ops/filter/test_llm_condition_filter.py) (`TestLLMConditionFilter`)
- [Return operator list 返回算子列表](../../Operators.md)
