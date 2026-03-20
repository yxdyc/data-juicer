# Agent 交互数据：Bad case 洞察方法论（precision-oriented）

**从哪里读起**：命令与排障 → [`QUICKSTART_BAD_CASE.md`](QUICKSTART_BAD_CASE.md)；仅 HTML 报告 → [`BAD_CASE_REPORT.md`](BAD_CASE_REPORT.md)；脚本参数 → [`scripts/README.md`](scripts/README.md)。总索引见 [`README.md`](README.md)。

---

本文配合 `agent_interaction_quality_analysis.yaml`：

- **第 9 步** `agent_bad_case_signal_mapper`：多源**确定性**信号 + 保守分层。  
- **第 10 步** `agent_insight_llm_mapper`（可选）：把数值 stats 与各类 LLM 评估文本**二次综合**成可解释 JSON（便于看板与人工质检）。

目标：为「某个 `request_model`（`meta.agent_request_model`）」产出**可复核**线索，优先 **precision > recall**。

## Agent 对话形态：多段 assistant 与 tool（必读）

智能体数据**不是**严格的「user → assistant → user → assistant」一轮一轮交替：

- 常见模式：`user` → 若干次 `(assistant + tool_calls) → tool → assistant → tool → …` → 最终 assistant 文本或下一轮 `user`。
- **`agent_dialog_normalize_mapper`** 将 `messages` 压成 `dialog_history` / `query` / `response` / `text` 时，对**同一 user 回合内多段 assistant** 必须**拼接保留**（含 tool summary）；可选 **`history_*_max_chars`** 对 tool/助手侧做**首尾保留 + 明确的中段省略标记**写回（与仅 prompt 截断不同），并打 **`meta.agent_dialog_history_compressed`**。否则下游 **`llm_quality_score_filter` / `llm_analysis_filter`** 只看到最后一次助手片段，容易误判「没干活」或「偏题」。
- **`tool_success_tagger_mapper`** 按 **每条 `role=tool` 消息**计数；`failed` / `No such file` 等会记 **fail**。在探索性策略里单次失败可能随后被纠正——第 9 步提供 **`min_tool_fail_count_for_signal`**（默认 1；菜谱可调到 2+）再抬 `tool_message_error_pattern`，并与 **`meta.tool_unknown_count`**（无法-regex 分类的 tool 行）一起看。
- **Dialog 打标算子**仍用 `dialog_history` 的 (q,r) 列表与 `max_round`；历史拼接正确后，多轮语义更一致。

## 数据血缘字段（normalize 自动写入）

| 原始字段 | `meta` 键 | 用途 |
|---------|-------------|------|
| `request_model` | `agent_request_model` | 按模型 cohort 分层、A/B |
| `pt` | `agent_pt` | 同一时间桶 / 版本窗对比 |
| `total_cost_time` | `agent_total_cost_time_ms` | 与 token 等一起看「贵不贵」 |

## Upstream 信号覆盖（第 9 步尽量吃满 pipeline）

`agent_bad_case_signal_mapper` 在样本上读取（**有则参与，无则跳过**）：

| 来源 | 字段 | 默认行为 |
|------|------|----------|
| `tool_success_tagger_mapper` | `tool_fail_count` | `high`：`tool_message_error_pattern`（需 fail ≥ `min_tool_fail_count_for_signal`） |
| 同上 | `tool_unknown_count` | 不参与 ratio；供排查「工具返回非典型文本/JSON」 |
| 同上 | `tool_success_ratio` + 轮次数 | `medium`：比例过低（≥`min_tool_rounds_for_ratio_signal` 轮；分母不含 unknown） |
| `usage_counter_mapper` | `total_tokens` 等 | 可选绝对阈值 → `medium` |
| normalize 血缘 | `agent_total_cost_time_ms` | 可选绝对阈值 → `medium` |
| `llm_analysis_filter` | `llm_analysis_score` + `llm_analysis_record.recommendation` | discard + 低分 → `high`/`medium`（可 `*_discard_must_be_strict`） |
| `llm_quality_score_filter` | `llm_quality_score` + `llm_quality_record` | 同上，信号码 `llm_reply_quality_eval_low` |
| `dialog_sentiment_detection_mapper` | `dialog_sentiment_labels` | `signal_on_negative_sentiment_hint` 开启时 → `medium`（易噪，报告里归「附录」类） |
| `perplexity_filter` + `stats.perplexity` | `stats.perplexity` | **可选**：KenLM；macOS 上 pip 常编译失败，默认菜谱可关掉 filter + `signal_on_high_perplexity: false`（见 **`KENLM_MACOS.md`**） |
| `llm_difficulty_score_filter` + quality | difficulty + `llm_quality_score` | **默认关**；`signal_hard_query_poor_reply` → `medium` |
| 文本 | `query` / `response` | 长 query + 极短回复 → `medium` |

分层：

- **`meta.agent_bad_case_signals`**：`{code, detail, weight}`。  
- **`meta.agent_bad_case_tier`**（机器枚举，jq 仍用英文）：`high_precision` | `watchlist` | `none`。  
  - 报告中译为 **强怀疑（主证据）** / **待观察（弱证据）** / **未标记**；**`high_precision` 不是「模型精度高」**，而是「强怀疑、建议优先复核」。
  - 长 agent 轨迹易出现 **多次 tool 返回含 error 模式**：若 **`high_precision_on_tool_fail_alone: false`**（全量菜谱默认），仅凭 tool 计数**不会**单独升强怀疑档；可调 **`min_tool_fail_count_for_signal`** 控制何时打出 tool 信号。

YAML 中请将 **`llm_analysis_discard_must_be_strict`** / **`llm_text_quality_discard_must_be_strict`** 保持为 `true`，避免把 `review` 当坏样本。

## 第 10 步：`agent_insight_llm_mapper`（auto-analyst）

把**一条样本**打包为 JSON（token、工具、意图/主题/情感标签、三路 LLM eval 摘要、`agent_bad_case_*`、query/response 截断预览），调用 API 模型输出**严格 JSON**：

- `headline`：一句话中文总览（适合卡片）  
- `root_causes`：`factor` + `confidence` + `cited_fields`（必须来自输入 JSON 的键路径）+ `rationale_one_line`  
- `narrative_alignment`：数值与文字 rationales 是否一致  
- `human_review_priority`：`P0`–`P3`  
- `viz_facets`：建议切片维度  

写入 **`meta.agent_insight_llm`**；解析失败时仍有占位结构，原文在 **`meta.agent_insight_llm_raw`**。

**成本控制**：菜谱默认 `run_for_tiers: [high_precision, watchlist]`；要对全量跑可删掉该字段（等价于 `null` = 全部）。

## 为何不强依赖「用户不满意」单一视角

1. **硬证据优先**：工具 error 模式、`tool_fail_count`。  
2. **LLM 评估**：discard + 极低分才抬 `high_precision`。  
3. **成本 / 长尾**：用 `demos/agent/scripts/compute_percentile_thresholds.py` 算 P90/P95；可 **`--write-calibration`** 生成 JSON，在 `agent_bad_case_signal_mapper` 中设 **`auto_calibrate_thresholds: true`** 与 **`calibration_json_path`**，按 `meta.agent_request_model` 自动合并 `default` 与 `by_request_model` 的阈值（token / 延迟 / perplexity）。默认 **`calibration_manual_overrides_auto: true`**：YAML 里显式写的 `max_*` / perplexity 仍优先于文件；设为 `false` 则优先用校准文件。

## 深挖思路（按 model / pt）

- **跨模型**：同一意图桶、同一工具族，对比 `llm_analysis_score`、`llm_quality_score`、token、tool 失败率。  
- **跨 `pt`**：同一模型下看 P50/P90 与 `high_precision` 占比漂移。  
- **组合**：优先 **AND** 多条独立证据再对外宣称「坏」。

## Pipeline 之后的分析脚本

见 **`demos/agent/scripts/README.md`**：

- `compute_percentile_thresholds.py`：分位数报告；`--write-calibration` → 自动阈值 JSON。  
- `analyze_bad_case_cohorts.py`：按 model / pt / tier 汇总并可选 CSV。  
- `slice_export_by_tier.py`：按 tier（及可选 model）导出子集 jsonl。

## jq 快速筛选

Data-Juicer 导出里 **`meta` 字段名为 `__dj__meta__`**（见 `data_juicer.utils.constant.Fields`）。若配置 **`keep_stats_in_res_ds: true`**（本仓库 agent 菜谱已默认），主 `processed.jsonl` 每行自带该字段；否则 meta 仅在同级 **`processed_stats.jsonl`**，可用 `demos/agent/scripts/dj_export_row.py` 的合并逻辑或后处理脚本（`verify` / `analyze` / `slice` / `compute_percentile` 已自动按行合并 `_stats` 文件）。

```bash
jq 'select(."__dj__meta__".agent_bad_case_tier=="high_precision" and ."__dj__meta__".agent_request_model=="qwen3-max")' processed.jsonl | head
jq -c '."__dj__meta__".agent_insight_llm.headline, ."__dj__meta__".agent_bad_case_signals' processed.jsonl | head
```

## 相关算子

- `tool_success_tagger_mapper`、`usage_counter_mapper`  
- `llm_analysis_filter`、`llm_quality_score_filter`、`llm_difficulty_score_filter`  
- `agent_bad_case_signal_mapper`、`agent_insight_llm_mapper`  
- `extract_entity_attribute_mapper`、`relation_identity_mapper`（字段空泛 / `role_relation` 等见 **`ENTITY_RELATION_TUNING.md`**）

若需 **按 session 聚合** 再判坏，请在数据中带 `session_id` 并在分析脚本中 groupby；当前算子为**逐条样本**。
