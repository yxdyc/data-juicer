# llm_extract_mapper

Extract structured fields from text using an LLM; write results to meta. Part of the llm_* semantic ops family.

This operator uses an LLM to extract user-defined fields from each sample's text (or multiple input keys). You provide an `output_schema` (key → extraction instruction). Results are written to `meta[meta_output_key]` or to individual meta keys. Supports structured (JSON) and unstructured (e.g. plain text, jsonl) input. Token/cost usage is recorded in `meta[llm_semantic_usage]` (prompt_tokens, completion_tokens, total_tokens, optional cost_estimate).

使用 LLM 从文本中提取用户定义的结构化字段；结果写入 meta。支持结构化与非结构化输入，并记录 token/cost 用量。

Type 算子类型: **mapper**

Tags 标签: gpu, vllm, hf, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `input_keys` | list | required | Sample keys to build input text (e.g. `["text"]` or `["query","response"]`). |
| `output_schema` | dict | required | `{output_key: "extraction instruction"}`. |
| `api_or_hf_model` | str | `'gpt-4o'` | Model name for API or HuggingFace. |
| `meta_output_key` | str, optional | `'llm_extract'` | If set, write full result to `meta[meta_output_key]`. |
| `knowledge_grounding_key` | str, optional | `None` | Optional sample key for per-sample grounding. |
| `knowledge_grounding_fixed` | str, optional | `None` | Optional fixed grounding string. |
| `is_hf_model` | bool | `False` | If true, use HuggingFace/Transformers. |
| `enable_vllm` | bool | `False` | If true, use vLLM backend. |
| `api_endpoint` | str, optional | `None` | URL endpoint for the API. |
| `response_path` | str, optional | `None` | Path to extract content from API response. |
| `system_prompt` | str, optional | `None` | Override default extraction system prompt. |
| `try_num` | int | `3` | Retries on parse/API failure. |
| `model_params` | dict | `{}` | Parameters for model init. |
| `sampling_params` | dict | `{}` | Sampling params (e.g. temperature, top_p). |

## 🌐 DashScope / OpenAI-compatible 环境变量

使用阿里云 DashScope **OpenAI 兼容模式**（REST）时，可只配环境变量，无需在 recipe 里写 key：

```bash
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENAI_API_KEY=<你的 DashScope API Key>
# 或使用：export DASHSCOPE_API_KEY=<同上>
```

默认算子模型名为 `gpt-4o`，在 DashScope 上不可用。若检测到上述兼容 Base URL，会自动改用 **`qwen-plus`**（可通过 `DASHSCOPE_DEFAULT_MODEL` 或 `OPENAI_DEFAULT_MODEL` 覆盖），或在配置里显式设置 `api_or_hf_model: qwen-plus`。

`OPENAI_API_URL` 与 `OPENAI_BASE_URL` 等价（任选其一）。

## 📊 Cost / usage 成本与用量

Each sample gets `meta[llm_semantic_usage]` with:
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `cost_estimate` (optional, when pricing is available)

## 🔗 Related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/llm_extract_mapper.py)
- [semantic ops 语义工具](../../../data_juicer/utils/llm_semantic_ops.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_llm_extract_mapper.py) (`TestLLMExtractMapper`)
- [Return operator list 返回算子列表](../../Operators.md)
