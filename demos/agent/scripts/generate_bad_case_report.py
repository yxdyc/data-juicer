#!/usr/bin/env python3
"""HTML 报告：分档统计、信号图、队列表、典型样例（可展开）、可选 LLM 页首导读。

- 图表使用中文字体配置，避免中文显示为方框。
- 典型样例默认页内最多 50 条，全量写入同目录 ``*_drilldown_full.jsonl``。
- ``--llm-summary``：调用 OpenAI 兼容接口（默认读 ``DASHSCOPE_API_KEY`` / ``OPENAI_API_KEY``）。

示例::

  python demos/agent/scripts/generate_bad_case_report.py \\
    --input ./outputs/agent_quality/processed.jsonl \\
    --output ./outputs/agent_quality/bad_case_report.html \\
    --llm-summary
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from analyze_bad_case_cohorts import aggregate_cohort_stdlib, load_merged_rows  # noqa: E402
from bad_case_signal_support import SIGNAL_SUPPORT_ROWS  # noqa: E402
from dj_export_row import get_dj_meta, get_dj_stats  # noqa: E402

# 机器枚举（jsonl / jq）不变；页面与图例用中文降低误解（原 high_precision ≠「高精度模型」）
TIER_LABEL_ZH: Dict[str, str] = {
    "high_precision": "强怀疑（主证据）",
    "watchlist": "待观察（弱证据）",
    "none": "未标记",
}


def _tier_zh(machine: str) -> str:
    return TIER_LABEL_ZH.get(str(machine), str(machine))


def _fmt_evidence_val(val: object, maxlen: int = 200) -> str:
    if val is None:
        return "—"
    s = str(val).strip()
    if len(s) > maxlen:
        return s[:maxlen] + "…"
    return s


def _evidence_rows_for_signal(code: str, row: dict, meta: dict, stats: dict) -> List[Tuple[str, str]]:
    """Map signal code → (field_path, display_value) for report tables."""
    rows: List[Tuple[str, str]] = []

    def add(k: str, v: object) -> None:
        rows.append((k, _fmt_evidence_val(v)))

    if code == "tool_message_error_pattern":
        add("meta.tool_success_count", meta.get("tool_success_count"))
        add("meta.tool_fail_count", meta.get("tool_fail_count"))
        add("meta.tool_unknown_count", meta.get("tool_unknown_count"))
        add("meta.tool_success_ratio", meta.get("tool_success_ratio"))
        add("算子", "tool_success_tagger_mapper（regex 扫 role=tool）")
    elif code == "low_tool_success_ratio":
        add("meta.tool_success_ratio", meta.get("tool_success_ratio"))
        add("meta.tool_success_count", meta.get("tool_success_count"))
        add("meta.tool_fail_count", meta.get("tool_fail_count"))
    elif code == "llm_agent_analysis_eval_low":
        add("stats.llm_analysis_score", stats.get("llm_analysis_score"))
        rec = stats.get("llm_analysis_record")
        if isinstance(rec, dict):
            add("stats.llm_analysis_record.recommendation", rec.get("recommendation"))
        add("算子", "llm_analysis_filter")
    elif code == "llm_reply_quality_eval_low":
        add("stats.llm_quality_score", stats.get("llm_quality_score"))
        add("算子", "llm_quality_score_filter")
    elif code == "suspect_empty_or_trivial_final_response":
        add("sample.query 长度", len(row.get("query") or ""))
        add("sample.response 长度", len(row.get("response") or ""))
        add("说明", "agent_dialog_normalize 后的末轮 query/response")
    elif code == "high_token_usage":
        add("meta.total_tokens", meta.get("total_tokens"))
        add("算子", "usage_counter_mapper")
    elif code == "high_latency_ms":
        add("meta.agent_total_cost_time_ms", meta.get("agent_total_cost_time_ms"))
        add("算子", "agent_dialog_normalize_mapper.copy_lineage_fields")
    elif code == "negative_sentiment_label_hint":
        add("meta.dialog_sentiment_labels", meta.get("dialog_sentiment_labels"))
        add("算子", "dialog_sentiment_detection_mapper")
    elif code == "high_perplexity":
        add("stats.perplexity", stats.get("perplexity"))
        add("算子", "perplexity_filter")
    elif code == "hard_query_low_reply_quality_conjunction":
        add("stats.llm_difficulty_score", stats.get("llm_difficulty_score"))
        add("stats.llm_quality_score", stats.get("llm_quality_score"))
        add("算子", "llm_difficulty_score_filter ∩ llm_quality_score_filter")
    else:
        add("(见归因总表)", "本信号上游字段未逐项绑定，可查 meta / stats 全文")
    # de-dup keys keeping first
    seen = set()
    out: List[Tuple[str, str]] = []
    for k, v in rows:
        if k in seen:
            continue
        seen.add(k)
        out.append((k, v))
    return out


def _signal_evidence_tables_html(signals: List[dict], row: dict) -> str:
    """Per-signal small tables: code + upstream fields + values."""
    meta = get_dj_meta(row)
    stats = get_dj_stats(row)
    blocks = []
    for sig in signals:
        if not isinstance(sig, dict):
            continue
        code = str(sig.get("code") or "")
        if not code:
            continue
        w = html.escape(str(sig.get("weight") or ""))
        det = html.escape(_fmt_evidence_val(sig.get("detail"), 300))
        erows = _evidence_rows_for_signal(code, row, meta, stats)
        body = "".join(
            f"<tr><td><code>{html.escape(k)}</code></td><td>{html.escape(v)}</td></tr>"
            for k, v in erows
        )
        blocks.append(
            f"<div class='sig-evidence'><div class='sig-evidence-h'>"
            f"<code>{html.escape(code)}</code> "
            f"<span class='wtag'>weight={w}</span> "
            f"<span class='det'>{det}</span></div>"
            f"<table class='inner'><tbody>{body}</tbody></table></div>"
        )
    if not blocks:
        return "<p class='note'>本样本无结构化信号。</p>"
    return "<div class='sig-evidence-wrap'>" + "".join(blocks) + "</div>"


def _global_evidence_snapshot_html(row: dict) -> str:
    """Snapshot of key meta/stats for quick sanity check."""
    meta = get_dj_meta(row)
    stats = get_dj_stats(row)
    pairs = [
        ("meta.tool_success_count", meta.get("tool_success_count")),
        ("meta.tool_fail_count", meta.get("tool_fail_count")),
        ("meta.tool_unknown_count", meta.get("tool_unknown_count")),
        ("meta.tool_success_ratio", meta.get("tool_success_ratio")),
        ("meta.total_tokens", meta.get("total_tokens")),
        ("meta.agent_turn_count", meta.get("agent_turn_count")),
        ("stats.llm_analysis_score", stats.get("llm_analysis_score")),
        ("stats.llm_quality_score", stats.get("llm_quality_score")),
        ("stats.llm_difficulty_score", stats.get("llm_difficulty_score")),
    ]
    body = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td><td>{html.escape(_fmt_evidence_val(v))}</td></tr>"
        for k, v in pairs
    )
    return (
        "<section class='snap'><h4>关键 meta / stats 快照</h4>"
        "<p class='note'>与归因链对照；缺失项为 <code>—</code>（可能未跑对应算子或未写入导出）。</p>"
        f"<table class='inner'><tbody>{body}</tbody></table></section>"
    )


def _configure_matplotlib_cjk(plt_mod) -> None:
    """Avoid CJK in chart titles/labels rendering as tofu (□) in embedded PNGs."""
    plt_mod.rcParams.update(
        {
            "font.sans-serif": [
                "PingFang SC",
                "Hiragino Sans GB",
                "Heiti SC",
                "Songti SC",
                "STHeiti",
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Noto Sans CJK JP",
                "Source Han Sans SC",
                "WenQuanYi Zen Hei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def _get_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _configure_matplotlib_cjk(plt)
        return plt
    except ImportError:  # pragma: no cover
        return None


def _fig_to_data_uri(fig, plt_mod) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt_mod.close(fig)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _tier_counts(rows: List[dict]) -> Counter:
    c: Counter = Counter()
    for row in rows:
        meta = get_dj_meta(row)
        c[str(meta.get("agent_bad_case_tier", "none"))] += 1
    return c


def _signal_counts_by_weight(rows: List[dict]) -> Tuple[Counter, Counter]:
    high_c: Counter = Counter()
    med_c: Counter = Counter()
    for row in rows:
        meta = get_dj_meta(row)
        for s in meta.get("agent_bad_case_signals") or []:
            if not isinstance(s, dict) or not s.get("code"):
                continue
            code = str(s["code"])
            w = s.get("weight")
            if w == "high":
                high_c[code] += 1
            elif w == "medium":
                med_c[code] += 1
    return high_c, med_c


def _attribution_table_html() -> str:
    parts = []
    for r in SIGNAL_SUPPORT_ROWS:
        role = "主证据" if r["role"] == "primary" else "附录·启发式"
        parts.append(
            "<tr>"
            f"<td><code>{html.escape(r['code'])}</code></td>"
            f"<td>{html.escape(role)}</td>"
            f"<td>{html.escape(str(r['weight_hint']))}</td>"
            f"<td>{html.escape(str(r['upstream']))}</td>"
            "</tr>"
        )
    thead = (
        "<thead><tr><th>信号代码</th><th>证据角色</th><th>典型权重</th>"
        "<th>上游字段与算子</th></tr></thead>"
    )
    return f"<table>{thead}<tbody>{''.join(parts)}</tbody></table>"


def _model_tier_matrix(rows: List[dict]) -> Dict[str, Counter]:
    m: DefaultDict[str, Counter] = defaultdict(Counter)
    for row in rows:
        meta = get_dj_meta(row)
        model = str(meta.get("agent_request_model") or "_unknown")
        tier = str(meta.get("agent_bad_case_tier", "none"))
        m[model][tier] += 1
    return dict(m)


def _json_pretty(obj: object) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except TypeError:  # pragma: no cover
        return str(obj)


def _iter_bad_case_drill_rows(rows: List[dict]):
    """Yield rows in tier order: high_precision first, then watchlist (stable by row index)."""
    tier_rank = {"high_precision": 0, "watchlist": 1}
    scored: List[Tuple[int, int, dict]] = []
    for i, row in enumerate(rows):
        meta = get_dj_meta(row)
        tier = str(meta.get("agent_bad_case_tier", "none"))
        if tier not in tier_rank:
            continue
        scored.append((tier_rank[tier], i, row))
    scored.sort(key=lambda x: (x[0], x[1]))
    for _tr, _i, row in scored:
        yield row


def _row_to_drill_entry(row: dict) -> dict:
    meta = get_dj_meta(row)
    tier = str(meta.get("agent_bad_case_tier", ""))
    rid = (
        meta.get("agent_request_id")
        or row.get("request_id")
        or row.get("trace_id")
        or row.get("id")
    )
    rid_s = str(rid).strip() if rid is not None else ""
    u_idx = meta.get("agent_last_user_msg_idx")
    a_idx = meta.get("agent_last_assistant_msg_idx")
    signals = meta.get("agent_bad_case_signals") or []
    insight = meta.get("agent_insight_llm") or {}
    meta_subset = {
        k: meta[k]
        for k in (
            "agent_request_id",
            "agent_last_user_msg_idx",
            "agent_last_assistant_msg_idx",
            "agent_request_model",
            "agent_pt",
            "agent_bad_case_tier",
            "agent_turn_count",
            "agent_total_cost_time_ms",
        )
        if k in meta
    }
    return {
        "tier": tier,
        "tier_label_zh": _tier_zh(tier),
        "request_id": rid_s,
        "u_idx": u_idx,
        "a_idx": a_idx,
        "model": str(meta.get("agent_request_model") or ""),
        "pt": str(meta.get("agent_pt") or ""),
        "query": row.get("query") or "",
        "response": row.get("response") or "",
        "signals_json": _json_pretty(signals),
        "insight_json": _json_pretty(insight),
        "meta_json": _json_pretty(meta_subset),
        "evidence_snapshot_html": _global_evidence_snapshot_html(row),
        "signal_evidence_html": _signal_evidence_tables_html(signals, row),
    }


def _collect_drilldown(rows: List[dict], limit: Optional[int]) -> List[dict]:
    """Bad-case rows (high_precision, watchlist) for the report UI; ``limit`` None = all."""
    out: List[dict] = []
    for row in _iter_bad_case_drill_rows(rows):
        out.append(_row_to_drill_entry(row))
        if limit is not None and len(out) >= limit:
            break
    return out


def _drill_export_payload(d: dict) -> dict:
    """JSONL-friendly row (drop pre-rendered HTML)."""
    skip = frozenset({"evidence_snapshot_html", "signal_evidence_html"})
    return {k: v for k, v in d.items() if k not in skip}


def _write_drilldown_jsonl(path: Path, entries: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in entries:
            f.write(json.dumps(_drill_export_payload(d), ensure_ascii=False, default=str) + "\n")


def _idx_badge(u_idx: object, a_idx: object) -> str:
    parts = []
    if u_idx is not None:
        parts.append(f"user_idx={u_idx}")
    if a_idx is not None:
        parts.append(f"asst_idx={a_idx}")
    return " · ".join(parts) if parts else "—"


def _drilldown_section_html(
    drill: List[dict],
    *,
    total_count: int,
    export_rel: Optional[str] = None,
) -> str:
    """典型案例清单：页内仅展示前几条，全量可另存 jsonl。"""
    title = "典型案例（强怀疑 / 待观察，可展开详情）"
    if not drill and total_count == 0:
        return (
            f"<h2 id='sec-cases'>{html.escape(title)}</h2>"
            "<p class='note'>本批没有 <code>high_precision</code>（强怀疑）或 "
            "<code>watchlist</code>（待观察）命中样本；或已在命令行关闭本段。</p>"
        )
    shown = len(drill)
    extra_note = ""
    if total_count > shown:
        extra_note = (
            f"<p class='note'><strong>页内展示 {shown} 条</strong>（按分档优先级排序），"
            f"本批同条件共 <strong>{total_count}</strong> 条。"
        )
        if export_rel:
            extra_note += (
                f" 全量请下载/用脚本打开："
                f"<a href='{html.escape(export_rel)}'><code>{html.escape(export_rel)}</code></a> "
                f"（JSON Lines，一行一例，便于 jq / pandas）。"
            )
        extra_note += "</p>"
    elif export_rel and total_count > 0:
        extra_note = (
            f"<p class='note'>本批共 <strong>{total_count}</strong> 条，已同时写入 "
            f"<a href='{html.escape(export_rel)}'><code>{html.escape(export_rel)}</code></a>。</p>"
        )
    cards = []
    for i, d in enumerate(drill):
        anchor = f"bc-drill-{i}"
        tier_m = html.escape(d["tier"])
        tier_show = html.escape(d.get("tier_label_zh") or _tier_zh(d["tier"]))
        tier_cls = "tier-hp" if d["tier"] == "high_precision" else "tier-wl"
        rid = html.escape(d["request_id"] or "—")
        idx_txt = html.escape(_idx_badge(d["u_idx"], d["a_idx"]))
        model = html.escape(d["model"] or "—")
        pt = html.escape(d["pt"] or "—")
        ev_snap = d.get("evidence_snapshot_html") or ""
        sig_ev = d.get("signal_evidence_html") or ""
        cards.append(
            f'<div class="drill-card" id="{anchor}">'
            '<div class="drill-summary">'
            f'<span class="tier-tag {tier_cls}" title="机器值 {tier_m}">{tier_show}</span> '
            f'<span class="tier-mach"><code>{tier_m}</code></span> '
            f'<a class="anchor-link" href="#{anchor}" title="锚点">#{i}</a> '
            f"<code class=\"rid\" title=\"request_id / trace / id\">{rid}</code> "
            f'<span class="idx" title="messages 中下标（0-based）">{idx_txt}</span> '
            f'<span class="cohort-mini">{model} · {pt}</span> '
            '<button type="button" class="drill-toggle" aria-expanded="false">'
            "展开字段</button>"
            "</div>"
            '<div class="drill-body" hidden>'
            f"{ev_snap}"
            "<h4 class='ev-h'>各信号 ↔ 上游字段与取值</h4>"
            f"{sig_ev}"
            '<div class="field-grid">'
            "<section><h4>query</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['query'])}</pre></section>"
            "<section><h4>response</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['response'])}</pre></section>"
            "<section><h4>agent_bad_case_signals（JSON）</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['signals_json'])}</pre></section>"
            "<section><h4>agent_insight_llm</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['insight_json'])}</pre></section>"
            "<section><h4>meta（钻取子集）</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['meta_json'])}</pre></section>"
            "</div></div></div>"
        )
    intro = (
        "<p class='note'>每条卡片对应一条导出样本：徽标为<strong>中文分档</strong>，"
        "旁边 <code>high_precision</code> / <code>watchlist</code> 为 JSON 中的枚举值。"
        "点击「展开字段」可查看 <strong>meta/stats 快照</strong>、各 signal 的<strong>上游证据字段</strong>，"
        "以及 <code>query</code> / <code>response</code> 全文。<code>#编号</code> 为页内锚点，便于复制链接。</p>"
    )
    block = (
        f"<h2 id='sec-cases'>{html.escape(title)}</h2>"
        f"{extra_note}{intro}"
        '<div class="drill-list">'
        f"{''.join(cards)}</div>"
    )
    return block


def _build_llm_digest_compact(
    n_rows: int,
    tier_cnt: Counter,
    high_c: Counter,
    med_c: Counter,
    cohort_rows: List[dict],
) -> str:
    """Minimal text for page-top LLM (shorter latency / tokens)."""
    hp = int(tier_cnt.get("high_precision", 0))
    wl = int(tier_cnt.get("watchlist", 0))
    nn = int(tier_cnt.get("none", 0))
    lines = [
        f"n={n_rows} high_precision={hp} watchlist={wl} none={nn}",
    ]
    if high_c:
        top_h = high_c.most_common(8)
        lines.append("high_signals: " + ", ".join(f"{c}:{n}" for c, n in top_h))
    if med_c:
        top_m = med_c.most_common(4)
        lines.append("med_signals: " + ", ".join(f"{c}:{n}" for c, n in top_m))
    ranked = sorted(
        [r for r in cohort_rows if int(r.get("count") or 0) > 0],
        key=lambda r: -int(r.get("count") or 0),
    )[:6]
    if ranked:
        bits = []
        for r in ranked:
            sig = str(r.get("top_signal_codes") or "")
            if len(sig) > 42:
                sig = sig[:42] + "…"
            bits.append(
                f"{r.get('agent_request_model', '')}|pt={r.get('agent_pt', '')}|"
                f"{r.get('tier', '')}|n={r.get('count')}|{sig}"
            )
        lines.append("top_cohorts: " + " / ".join(bits))
    return "\n".join(lines)


def _rule_based_exec_summary(
    n_rows: int,
    tier_cnt: Counter,
    high_c: Counter,
    med_c: Counter,
) -> str:
    hp = int(tier_cnt.get("high_precision", 0))
    wl = int(tier_cnt.get("watchlist", 0))
    nn = int(tier_cnt.get("none", 0))
    lines = [
        "【结论摘要】",
        f"- 本批合计 {n_rows} 条；其中强怀疑（主证据）{hp} 条、待观察（弱证据）{wl} 条、未标记 {nn} 条。",
    ]
    if high_c:
        bits = [f"{c}（{n} 次）" for c, n in high_c.most_common(5)]
        lines.append("- high 权重信号出现较多的有：" + "；".join(bits) + "。")
    lines.extend(
        [
            "",
            "【后续分析建议】",
            "- 结合下方「按模型堆叠图」与 cohort 表，看强怀疑是否集中在少数模型或日期桶。",
            "- 对 high 权重信号做共现统计，避免单条启发式误杀。",
            "- 强怀疑档建议优先人工复核；待观察档宜抽样或与业务规则对照。",
            "",
            "【阅读提示】",
            "- 「强怀疑」多为结构化主证据；「待观察」多为弱信号组合，请勿混读。",
        ]
    )
    return "\n".join(lines)


def _fetch_exec_summary_llm(
    digest: str,
    *,
    model: str,
    api_key: str,
    api_base: str,
    timeout_sec: int = 60,
) -> Optional[str]:
    """OpenAI-compatible ``/v1/chat/completions`` (DashScope 兼容模式、OpenAI 等)."""
    url = api_base.rstrip("/") + "/chat/completions"
    system = (
        "你是数据分析顾问。只根据用户给的若干行批次汇总写报告页首短导读；"
        "禁止编造未出现的数字、档位名或信号 code；表述简练。"
    )
    user = (
        "以下为同一批次的极简汇总（非逐条日志）。请用纯中文、纯文本 output，分三块，块间空一行：\n"
        "【结论摘要】3～5行，每行以「- 」，基于 n= / high_precision= / high_signals。\n"
        "【后续分析建议】2～4行，每行「- 」，可执行即可。\n"
        "【阅读提示】1～2行，区分强怀疑(high_precision)与待观察(watchlist)。\n\n"
        f"{digest}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 768,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as e:
        print(f"WARNING: LLM 导读请求失败: {e}", file=sys.stderr)
        return None
    try:
        return str(body["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError):
        print(f"WARNING: LLM 导读返回格式异常: {str(body)[:800]}", file=sys.stderr)
        return None


def _exec_summary_section_html(body_text: str, source_note: str) -> str:
    return (
        "<section class='exec-summary' id='sec-guide'>"
        "<h2>报告导读（结论与后续分析）</h2>"
        f"<p class='note'>{html.escape(source_note)}</p>"
        f"<pre class='exec-summary-pre'>{html.escape(body_text)}</pre>"
        "</section>"
    )


def _insight_section_rich_html(rows: List[dict], tier: str, limit: int) -> str:
    """较完整的单条 insight 卡片（来自 agent_insight_llm）。"""
    tier_zh = _tier_zh(tier)
    cards: List[str] = []
    n = 0
    for row in rows:
        if n >= limit:
            break
        meta = get_dj_meta(row)
        if str(meta.get("agent_bad_case_tier", "")) != tier:
            continue
        ins = meta.get("agent_insight_llm") or {}
        hl = (ins.get("headline") or "").strip()
        if not hl:
            continue
        n += 1
        rid = str(
            meta.get("agent_request_id") or row.get("request_id") or row.get("trace_id") or row.get("id") or ""
        ).strip()
        model = str(meta.get("agent_request_model") or "")
        pr = (ins.get("human_review_priority") or "").strip()
        align = (ins.get("narrative_alignment") or "").strip()
        audit = (ins.get("audit_notes") or "").strip()
        facets = ins.get("viz_facets") or []
        if not isinstance(facets, list):
            facets = [str(facets)]
        facets_s = "、".join(str(x) for x in facets[:12] if x)
        causes = ins.get("root_causes") or []
        cause_lis = []
        if isinstance(causes, list):
            for c in causes[:5]:
                if not isinstance(c, dict):
                    cause_lis.append(f"<li>{html.escape(str(c))}</li>")
                    continue
                factor = html.escape(str(c.get("factor") or ""))
                conf = html.escape(str(c.get("confidence") or ""))
                r1 = html.escape(str(c.get("rationale_one_line") or "")[:280])
                cited = c.get("cited_fields") or []
                if not isinstance(cited, list):
                    cited = [str(cited)]
                cf = html.escape(", ".join(str(x) for x in cited[:8]))
                cite_html = f' <span class="cite">依据字段: {cf}</span>' if cf else ""
                cause_lis.append(
                    f"<li><strong>{factor}</strong>（置信 {conf}）— {r1}{cite_html}</li>"
                )
        causes_html = "<ul class='causes'>" + "".join(cause_lis) + "</ul>" if cause_lis else ""
        meta_line = (
            f"<span class='ins-meta'><code>{html.escape(rid or '—')}</code> · "
            f"{html.escape(model or '—')}</span>"
        )
        badges = []
        if pr:
            badges.append(f"<span class='ins-badge pr'>{html.escape(pr)}</span>")
        if align:
            badges.append(f"<span class='ins-badge al'>{html.escape(align)}</span>")
        badge_html = " ".join(badges)
        audit_html = (
            f"<p class='ins-audit'>{html.escape(audit)}</p>" if audit else ""
        )
        facets_html = (
            f"<p class='ins-facets'><strong>建议制图维度</strong>：{html.escape(facets_s)}</p>"
            if facets_s
            else ""
        )
        cards.append(
            "<div class='insight-card'>"
            f"<div class='insight-h'>{html.escape(hl)} {badge_html}</div>"
            f"{meta_line}"
            f"{causes_html}{facets_html}{audit_html}"
            "</div>"
        )
    if not cards:
        return (
            f"<section class='insight-sec' id='sec-insights'><h2>单条 Insight 摘录（{html.escape(tier_zh)}）</h2>"
            "<p class='note'>本档下暂无带 <code>headline</code> 的 "
            "<code>meta.agent_insight_llm</code>（可能未跑 insight 算子，或解析失败）。</p></section>"
        )
    return (
        f"<section class='insight-sec' id='sec-insights'><h2>单条 Insight 摘录（{html.escape(tier_zh)}）</h2>"
        "<p class='note'>来自 <code>agent_insight_llm_mapper</code>：摘要句、复核优先级、"
        "成因与建议制图维度；可与上文图表对照，下文「典型样例」支持逐条展开互证。</p>"
        f"<div class='insight-list'>{''.join(cards)}</div></section>"
    )


def _chart_tier_bar(tier_cnt: Counter, plt_mod) -> Optional[str]:
    if plt_mod is None:
        return None
    order = ("high_precision", "watchlist", "none")
    labels = [t for t in order if tier_cnt.get(t, 0) > 0]
    if not labels:
        labels = list(tier_cnt.keys())
    vals = [tier_cnt[t] for t in labels]
    labels_zh = [f"{_tier_zh(t)}\n({t})" for t in labels]
    fig, ax = plt_mod.subplots(figsize=(7, 3.5))
    colors = {
        "high_precision": "#c0392b",
        "watchlist": "#f39c12",
        "none": "#95a5a6",
    }
    bars = ax.bar(
        labels_zh,
        vals,
        color=[colors.get(x, "#3498db") for x in labels],
    )
    try:
        ax.bar_label(bars, labels=[str(int(v)) for v in vals], padding=3, fontsize=10)
    except AttributeError:  # matplotlib < 3.4
        pass
    ax.set_title("分档样本数（柱顶数字为条数；中文为展示名）")
    ax.set_ylabel("条数")
    plt_mod.setp(ax.xaxis.get_majorticklabels(), rotation=12, ha="center")
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _chart_signals(
    sig_cnt: Counter,
    plt_mod,
    title: str,
    color: str = "#2980b9",
    top_n: int = 14,
) -> Optional[str]:
    if plt_mod is None or not sig_cnt:
        return None
    items = sig_cnt.most_common(top_n)
    labels = [x[0] for x in items]
    vals = [x[1] for x in items]
    fig, ax = plt_mod.subplots(figsize=(7, max(2.5, 0.35 * len(labels))))
    labels_r = labels[::-1]
    vals_r = vals[::-1]
    bars = ax.barh(labels_r, vals_r, color=color)
    try:
        ax.bar_label(bars, labels=[str(int(v)) for v in vals_r], padding=3, fontsize=9)
    except AttributeError:
        pass
    ax.set_title(title)
    ax.set_xlabel("条数")
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _chart_by_model(model_tier: Dict[str, Counter], plt_mod) -> Optional[str]:
    if plt_mod is None or len(model_tier) == 0:
        return None
    models = sorted(model_tier.keys())
    tiers = ("high_precision", "watchlist", "none")
    fig, ax = plt_mod.subplots(figsize=(max(6, len(models) * 0.9), 3.8))
    n = len(models)
    x = list(range(n))
    w = 0.65
    bottom = [0.0] * n
    colors = {"high_precision": "#c0392b", "watchlist": "#f39c12", "none": "#bdc3c7"}
    for tier in tiers:
        vs = [float(model_tier[m].get(tier, 0)) for m in models]
        if not any(vs):
            continue
        leg = f"{_tier_zh(tier)} ({tier})"
        ax.bar(x, vs, bottom=bottom, label=leg, color=colors[tier], width=w)
        vmax = max(vs) if vs else 1.0
        for i, v in enumerate(vs):
            if v > 0:
                yc = bottom[i] + v / 2.0
                ax.text(
                    x[i],
                    yc,
                    str(int(v)),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if v >= vmax * 0.25 else "#222",
                )
        bottom = [b + v for b, v in zip(bottom, vs)]
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title("按 request_model 堆叠分档（每段数字为该档条数）")
    ax.set_ylabel("条数")
    ax.legend()
    plt_mod.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _html_page(
    title: str,
    input_path: str,
    n_rows: int,
    tier_cnt: Counter,
    cohort_rows: List[dict],
    attribution_table: str,
    exec_summary_html: str,
    charts_html: str,
    drilldown_html: str,
    insight_section_html: str,
) -> str:
    gen_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tier_rows = "".join(
        "<tr>"
        f"<td>{html.escape(_tier_zh(k))}</td>"
        f"<td><code>{html.escape(k)}</code></td>"
        f"<td>{v}</td>"
        "</tr>"
        for k, v in sorted(tier_cnt.items(), key=lambda x: -x[1])
    )
    cohort_lines = []
    for r in cohort_rows:
        if not r.get("count") and not r.get("top_signal_codes"):
            continue
        tm = str(r.get("tier", "") or "")
        cohort_lines.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('agent_request_model', '')))}</td>"
            f"<td>{html.escape(str(r.get('agent_pt', '')))}</td>"
            f"<td>{html.escape(_tier_zh(tm))} <code>{html.escape(tm)}</code></td>"
            f"<td>{int(r.get('count') or 0)}</td>"
            f"<td>{html.escape(str(r.get('top_signal_codes', '')))}</td>"
            "</tr>"
        )

    css = (
        "body{font-family:'PingFang SC','Hiragino Sans GB','Microsoft YaHei',"
        "'Noto Sans SC',system-ui,sans-serif;margin:24px;max-width:1100px;}"
        "h1{font-size:1.35rem;}.meta{color:#555;font-size:0.9rem;}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}"
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:left;}"
        "th{background:#f4f4f4;}img{max-width:100%;height:auto;}"
        "ul{line-height:1.5;}.model{color:#555;font-size:0.85rem;}"
        "details{margin-top:2rem;padding:12px;background:#fafafa;border:1px solid #eee;}"
        "summary{cursor:pointer;font-weight:600;}"
        ".note{color:#666;font-size:0.88rem;}"
        ".drill-list{display:flex;flex-direction:column;gap:10px;margin:1rem 0;}"
        ".drill-card{border:1px solid #ddd;border-radius:8px;background:#fff;"
        "box-shadow:0 1px 2px rgba(0,0,0,0.04);}"
        ".drill-summary{display:flex;flex-wrap:wrap;align-items:center;gap:8px;"
        "padding:10px 12px;background:#fafafa;border-radius:8px 8px 0 0;"
        "border-bottom:1px solid #eee;}"
        ".drill-body{padding:12px;border-radius:0 0 8px 8px;}"
        ".tier-tag{font-size:0.75rem;padding:2px 8px;border-radius:999px;"
        "font-weight:600;}"
        ".tier-hp{background:#fadbd8;color:#922b21;}"
        ".tier-wl{background:#fdebd0;color:#9c640c;}"
        ".rid{font-size:0.85rem;background:#f4f6f8;padding:2px 6px;border-radius:4px;}"
        ".idx{font-size:0.82rem;color:#555;}"
        ".cohort-mini{font-size:0.82rem;color:#666;}"
        ".anchor-link{font-weight:600;margin-right:4px;}"
        "button.drill-toggle{margin-left:auto;font-size:0.85rem;cursor:pointer;"
        "padding:6px 12px;border:1px solid #bbb;border-radius:6px;background:#fff;}"
        "button.drill-toggle:hover{background:#f0f0f0;}"
        ".field-grid{display:grid;gap:14px;}"
        "@media(min-width:900px){.field-grid{grid-template-columns:1fr 1fr;}}"
        ".field-grid section{margin:0;}"
        ".field-grid h4{margin:0 0 6px;font-size:0.9rem;color:#333;}"
        "pre.drill-pre{margin:0;white-space:pre-wrap;word-break:break-word;"
        "max-height:320px;overflow:auto;font-size:0.82rem;line-height:1.45;"
        "background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:6px;}"
        ".tier-mach{font-size:0.78rem;color:#888;}"
        ".tier-legend li{margin:6px 0;}"
        "table.inner{font-size:0.82rem;margin:0.5rem 0;width:100%;}"
        "table.inner td:first-child{white-space:nowrap;width:38%;vertical-align:top;}"
        ".sig-evidence-wrap{display:flex;flex-direction:column;gap:10px;margin:0.8rem 0;}"
        ".sig-evidence{border:1px solid #e0e0e0;border-radius:6px;padding:8px 10px;background:#fcfcfc;}"
        ".sig-evidence-h{margin-bottom:6px;font-size:0.88rem;}"
        ".sig-evidence .wtag{color:#555;font-size:0.8rem;margin-left:6px;}"
        ".sig-evidence .det{color:#666;font-size:0.8rem;margin-left:6px;}"
        ".snap table.inner{margin-top:4px;}"
        "h4.ev-h{margin:12px 0 6px;font-size:0.95rem;}"
        ".exec-summary-pre{margin:0.5rem 0 0;white-space:pre-wrap;word-break:"
        "break-word;line-height:1.55;font-size:0.92rem;background:#f7f9fc;border:1px solid #e2e8f0;"
        "border-radius:8px;padding:14px 16px;}"
        ".insight-list{display:flex;flex-direction:column;gap:12px;margin:1rem 0;}"
        ".insight-card{border:1px solid #e0e0e0;border-radius:8px;padding:12px 14px;"
        "background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.04);}"
        ".insight-h{font-weight:600;font-size:1rem;margin-bottom:6px;line-height:1.4;}"
        ".ins-meta{display:block;font-size:0.85rem;color:#555;margin-bottom:8px;}"
        ".ins-badge{font-size:0.75rem;padding:2px 8px;border-radius:6px;margin-left:6px;}"
        ".ins-badge.pr{background:#fce4ec;color:#880e4f;}"
        ".ins-badge.al{background:#e3f2fd;color:#0d47a1;}"
        "ul.causes{margin:6px 0 0 1rem;padding:0;font-size:0.88rem;line-height:1.45;}"
        "ul.causes li{margin:4px 0;}"
        ".cite{color:#666;font-size:0.82rem;}"
        ".ins-facets,.ins-audit{font-size:0.86rem;margin:8px 0 0;color:#444;}"
        "section.insight-sec{margin-top:2rem;}"
    )
    thead = (
        "<thead><tr><th>请求模型</th><th>日期桶 (pt)</th>"
        "<th>分档（中文 / 枚举）</th><th>条数</th><th>常见信号代码</th></tr></thead>"
    )
    tier_legend = (
        "<h2 id='sec-tiers'>如何理解「分档」</h2>"
        "<ul class='tier-legend'>"
        "<li><strong>强怀疑（主证据）</strong> — JSON 中机器值为 <code>high_precision</code>："
        "通常对应较强的结构化证据（如多项 LLM 评估偏低）。"
        "若菜谱里打开 <code>high_precision_on_tool_fail_alone</code>，"
        "也可能仅凭多次 tool 失败模式进入此档，解读时请结合业务。</li>"
        "<li><strong>待观察（弱证据）</strong> — <code>watchlist</code>："
        "多为启发式或单一弱信号，<strong>不宜</strong>直接等同「劣质样本」，适合抽样核对。</li>"
        "<li><strong>未标记</strong> — <code>none</code>：未命中当前分层规则。</li>"
        "</ul>"
        "<p class='note'>提示：<code>high_precision</code> 在本项目语义为「值得优先复核」，"
        "与「模型高精度」无关；命令行 / JSON 仍保留英文枚举，便于脚本处理。</p>"
    )
    nav = (
        "<p class='note'><strong>快速跳转：</strong> "
        "<a href='#sec-guide'>导读</a> · "
        "<a href='#sec-tiers'>分档说明</a> · "
        "<a href='#sec-charts'>图表</a> · "
        "<a href='#sec-insights'>Insight</a> · "
        "<a href='#sec-cases'>典型样例</a> · "
        "<a href='#sec-attrib'>归因表</a> · "
        "<a href='#sec-cohort'>队列明细</a></p>"
    )
    adv = (
        "<p>进阶说明见 <code>demos/agent/BAD_CASE_INSIGHTS.md</code>、"
        "<code>ENTITY_RELATION_TUNING.md</code>、"
        "<code>demos/agent/scripts/README.md</code>。</p>"
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>{css}</style>
</head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">
  生成时间 {html.escape(gen_at)} ·
  数据文件 <code>{html.escape(input_path)}</code><br/>
  本报告载入 <strong>{n_rows}</strong> 条样本（若使用 --limit 则仅为其子集）
</div>
{nav}
{exec_summary_html}
<h2 id='sec-counts'>各分档条数</h2>
<table><thead><tr><th>展示名</th><th>机器枚举</th><th>条数</th></tr></thead>
<tbody>{tier_rows}</tbody></table>
{tier_legend}
<h2 id='sec-charts'>图表（整体分布）</h2>
<p class='note'>柱或堆叠段上的<strong>数字为该组样本数</strong>；信号图为本批内出现次数。详情可在后文 Insight 与「典型样例」中<strong>逐条展开</strong>查看。</p>
{charts_html}
{insight_section_html}
{drilldown_html}
<h2 id='sec-attrib'>信号归因对照表</h2>
<p class='note'>说明各信号 <code>code</code> 与上游 <strong>meta / stats</strong> 及常见算子的对应关系，便于与「典型样例」中的证据表对照。</p>
{attribution_table}
<h2 id='sec-cohort'>按模型 × 日期 × 分档 的队列明细</h2>
<table>{thead}<tbody>{"".join(cohort_lines)}</tbody></table>
<details>
<summary>进阶 / 调试资源</summary>
{adv}
</details>
<script>
(function () {{
  document.querySelectorAll("button.drill-toggle").forEach(function (btn) {{
    btn.addEventListener("click", function () {{
      var card = btn.closest(".drill-card");
      if (!card) return;
      var body = card.querySelector(".drill-body");
      if (!body) return;
      var open = body.hasAttribute("hidden");
      if (open) {{
        body.removeAttribute("hidden");
        btn.textContent = "收起字段";
        btn.setAttribute("aria-expanded", "true");
      }} else {{
        body.setAttribute("hidden", "");
        btn.textContent = "展开字段";
        btn.setAttribute("aria-expanded", "false");
      }}
    }});
  }});
}})();
</script>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="processed.jsonl")
    ap.add_argument("--output", required=True, help="Output .html path")
    ap.add_argument(
        "--title",
        default="智能体交互 · Bad-case 分析报告",
        help="HTML title / H1",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max rows to read")
    ap.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip matplotlib figures (table-only HTML)",
    )
    ap.add_argument(
        "--sample-headlines",
        type=int,
        default=10,
        help="Max high_precision insight cards (0=off)",
    )
    ap.add_argument(
        "--drilldown-limit",
        type=int,
        default=-1,
        help="0=关闭「典型样例」整节；>0=仅导出/保留前 N 条（截断全量清单）；默认 -1=不截断导出",
    )
    ap.add_argument(
        "--drilldown-display-max",
        type=int,
        default=50,
        help="HTML 内嵌展开的强怀疑/待观察样例条数上限（其余仅写入 jsonl）",
    )
    ap.add_argument(
        "--no-drilldown-export",
        action="store_true",
        help="不写 *_drilldown_full.jsonl（仍可按 display-max 展示页内卡片）",
    )
    ap.add_argument(
        "--llm-summary",
        action="store_true",
        help="调用 OpenAI 兼容接口生成页首导读（需环境变量 API Key）",
    )
    ap.add_argument(
        "--llm-model",
        default=os.environ.get("BAD_CASE_REPORT_LLM_MODEL", "qwen3.5-plus"),
        help="页首导读所用模型（默认 qwen3.5-plus 或环境变量 BAD_CASE_REPORT_LLM_MODEL）",
    )
    ap.add_argument(
        "--llm-api-base",
        default=os.environ.get(
            "OPENAI_API_BASE",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        help="Chat Completions base URL（需含 /v1，实际请求 .../chat/completions）",
    )
    ap.add_argument(
        "--llm-api-key",
        default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
        help="API Key（默认读环境变量 DASHSCOPE_API_KEY 或 OPENAI_API_KEY）",
    )
    args = ap.parse_args()

    rows = load_merged_rows(args.input, args.limit)
    if not rows:
        print("ERROR: no rows loaded; check --input path.", file=sys.stderr)
        return 2

    tier_cnt = _tier_counts(rows)
    high_c, med_c = _signal_counts_by_weight(rows)
    model_tier = _model_tier_matrix(rows)
    cohort = aggregate_cohort_stdlib(rows)
    att_html = _attribution_table_html()

    chart_tier = chart_sig_high = chart_sig_med = chart_model = None
    plt_mod = None
    if not args.no_charts:
        plt_mod = _get_plt()
    if plt_mod is not None:
        chart_tier = _chart_tier_bar(tier_cnt, plt_mod)
        if high_c:
            chart_sig_high = _chart_signals(
                high_c,
                plt_mod,
                "本批 high 权重主证据信号（柱末数字为条数）",
                "#c0392b",
            )
        if med_c:
            chart_sig_med = _chart_signals(
                med_c,
                plt_mod,
                "附录：medium 启发式 / 弱证据信号（柱末数字为条数）",
                "#2980b9",
            )
        if len(model_tier) >= 1:
            chart_model = _chart_by_model(model_tier, plt_mod)

    rule_summary = _rule_based_exec_summary(len(rows), tier_cnt, high_c, med_c)
    digest_llm = _build_llm_digest_compact(len(rows), tier_cnt, high_c, med_c, cohort)
    llm_summary: Optional[str] = None
    if args.llm_summary:
        key = (args.llm_api_key or "").strip()
        if key:
            llm_summary = _fetch_exec_summary_llm(
                digest_llm,
                model=args.llm_model,
                api_key=key,
                api_base=args.llm_api_base,
            )
        else:
            print("WARNING: --llm-summary 已开启但未配置 API Key。", file=sys.stderr)
    body_summary = llm_summary or rule_summary
    summary_note = (
        "以下由大模型根据上方同批次聚合摘要生成，数字请务必与下方表格交叉核对。"
        if llm_summary
        else "以下由离线规则根据当前批次统计即时生成。若需更自然的表述，可加参数 --llm-summary 并配置 API Key。"
    )
    exec_summary_html = _exec_summary_section_html(body_summary, summary_note)

    charts_blocks: List[str] = []
    if chart_tier:
        charts_blocks.append(
            "<h3>各分档条数</h3>"
            f"<p class='note'><img src='{chart_tier}' alt='tier 分布'/></p>"
        )
    if chart_model:
        charts_blocks.append(
            "<h3>按请求模型堆叠</h3>"
            f"<p class='note'><img src='{chart_model}' alt='按模型'/></p>"
        )
    if chart_sig_high:
        charts_blocks.append(
            "<h3>主证据信号 Top</h3>"
            f"<p class='note'><img src='{chart_sig_high}' alt='high 信号'/></p>"
        )
    if chart_sig_med:
        charts_blocks.append(
            "<h3>弱证据 / 启发式信号</h3>"
            "<p class='note'>多为单条弱提示，须与分档与其它字段共同解读。</p>"
            f"<p class='note'><img src='{chart_sig_med}' alt='medium 信号'/></p>"
        )
    if charts_blocks:
        charts_html = "\n".join(charts_blocks)
    elif args.no_charts or plt_mod is None:
        charts_html = "<p class='note'>未生成图表（已加 --no-charts 或未安装 matplotlib）。</p>"
    else:
        charts_html = "<p class='note'>本批暂无可用绘图数据。</p>"

    drill_html = ""
    export_rel: Optional[str] = None
    out_path = Path(args.output)
    if args.drilldown_limit != 0:
        cap_export = args.drilldown_limit if args.drilldown_limit > 0 else None
        drill_all = _collect_drilldown(rows, cap_export)
        total_drill = len(drill_all)
        display_n = max(0, args.drilldown_display_max)
        drill_show = drill_all[:display_n] if display_n else []
        if not args.no_drilldown_export and drill_all:
            export_path = out_path.with_name(out_path.stem + "_drilldown_full.jsonl")
            _write_drilldown_jsonl(export_path, drill_all)
            export_rel = export_path.name
            print(f"Wrote drilldown export {export_path.resolve()}")
        drill_html = _drilldown_section_html(
            drill_show,
            total_count=total_drill,
            export_rel=export_rel,
        )

    insight_section_html = ""
    if args.sample_headlines > 0:
        insight_section_html = _insight_section_rich_html(
            rows, "high_precision", args.sample_headlines
        )

    page = _html_page(
        args.title,
        str(Path(args.input).resolve()),
        len(rows),
        tier_cnt,
        cohort,
        att_html,
        exec_summary_html,
        charts_html,
        drill_html,
        insight_section_html,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"Wrote {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
