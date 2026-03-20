#!/usr/bin/env python3
"""HTML report: bad-case tiers, signals, cohort table, embedded charts.

Merges ``*_stats.jsonl`` with ``processed.jsonl`` when needed.

Example:
  python demos/agent/scripts/generate_bad_case_report.py \\
    --input ./outputs/agent_quality/processed.jsonl \\
    --output ./outputs/agent_quality/bad_case_report.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
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


def _get_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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
        "<thead><tr><th>signal code</th><th>角色</th><th>典型权重</th>"
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


def _collect_drilldown(rows: List[dict], limit: int) -> List[dict]:
    """Bad-case rows (high_precision, watchlist) with stable ids for the report UI."""
    tier_rank = {"high_precision": 0, "watchlist": 1}
    scored: List[Tuple[int, int, dict]] = []
    for i, row in enumerate(rows):
        meta = get_dj_meta(row)
        tier = str(meta.get("agent_bad_case_tier", "none"))
        if tier not in tier_rank:
            continue
        scored.append((tier_rank[tier], i, row))
    scored.sort(key=lambda x: (x[0], x[1]))
    out: List[dict] = []
    for _tr, _orig_i, row in scored[:limit]:
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
        out.append(
            {
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
        )
    return out


def _idx_badge(u_idx: object, a_idx: object) -> str:
    parts = []
    if u_idx is not None:
        parts.append(f"user_idx={u_idx}")
    if a_idx is not None:
        parts.append(f"asst_idx={a_idx}")
    return " · ".join(parts) if parts else "—"


def _drilldown_section_html(drill: List[dict]) -> str:
    if not drill:
        return (
            "<h2>样本钻取（强怀疑 / 待观察）</h2>"
            "<p class='note'>本批无 <code>high_precision</code> / "
            "<code>watchlist</code> 样本，或已将钻取条数上限设为 0。</p>"
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
    block = (
        "<h2>样本钻取（强怀疑 / 待观察）</h2>"
        "<p class='note'>卡片标签为<strong>中文分档</strong>；旁注 <code>high_precision</code> / "
        "<code>watchlist</code> 为导出中的机器枚举。展开后可见 <strong>meta/stats 快照</strong>、"
        "每条 signal 对应的<strong>上游字段与当前取值</strong>，再对照 <code>query/response</code>。"
        "<code>#n</code> 为页内锚点。</p>"
        '<div class="drill-list">'
        f"{''.join(cards)}</div>"
    )
    return block


def _insight_samples(
    rows: List[dict],
    tier: str,
    limit: int,
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for row in rows:
        if len(out) >= limit:
            break
        meta = get_dj_meta(row)
        if str(meta.get("agent_bad_case_tier", "")) != tier:
            continue
        ins = meta.get("agent_insight_llm") or {}
        hl = (ins.get("headline") or "").strip()
        if hl:
            out.append((hl, str(meta.get("agent_request_model", ""))))
    return out


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
    ax.bar(
        labels_zh,
        vals,
        color=[colors.get(x, "#3498db") for x in labels],
    )
    ax.set_title("Bad-case 分档计数（中文为展示名，括号为导出枚举）")
    ax.set_ylabel("Samples")
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
    ax.barh(labels[::-1], vals[::-1], color=color)
    ax.set_title(title)
    ax.set_xlabel("Count")
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _chart_by_model(model_tier: Dict[str, Counter], plt_mod) -> Optional[str]:
    if plt_mod is None or len(model_tier) == 0:
        return None
    models = sorted(model_tier.keys())
    tiers = ("high_precision", "watchlist", "none")
    fig, ax = plt_mod.subplots(figsize=(max(6, len(models) * 0.9), 3.8))
    bottom = [0] * len(models)
    colors = {"high_precision": "#c0392b", "watchlist": "#f39c12", "none": "#bdc3c7"}
    for tier in tiers:
        vs = [model_tier[m].get(tier, 0) for m in models]
        if not any(vs):
            continue
        leg = f"{_tier_zh(tier)} ({tier})"
        ax.bar(models, vs, bottom=bottom, label=leg, color=colors[tier], width=0.65)
        bottom = [b + v for b, v in zip(bottom, vs)]
    ax.set_title("按模型的分档占比（图例：中文名 + 机器枚举）")
    ax.set_ylabel("Samples")
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
    chart_tier: Optional[str],
    chart_model: Optional[str],
    chart_sig_high: Optional[str],
    chart_sig_med: Optional[str],
    attribution_table: str,
    samples_hp: List[Tuple[str, str]],
    drilldown_html: str,
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
    sample_block = ""
    if samples_hp:
        parts = []
        for h, m in samples_hp:
            parts.append(
                "<li><span class='model'>"
                f"{html.escape(m)}</span> — {html.escape(h)}</li>"
            )
        lis = "".join(parts)
        sample_block = (
            "<h2>Insight 摘录（强怀疑档 / high_precision）</h2><ul>"
            f"{lis}</ul>"
        )

    charts = []
    if chart_tier:
        charts.append(
            "<h2>Tier overview</h2>"
            f"<img src='{chart_tier}' alt='tiers'/>"
        )
    if chart_model:
        charts.append(
            "<h2>By request model</h2>"
            f"<img src='{chart_model}' alt='by model'/>"
        )
    if chart_sig_high:
        charts.append(
            "<h3>本批次 high 权重信号</h3>"
            f"<img src='{chart_sig_high}' alt='signals high'/>"
        )
    if chart_sig_med:
        charts.append(
            "<h3>附录：medium 启发式信号</h3>"
            "<p class='note'>多为单条弱证据；tier 需与其它信号组合才可能进 watchlist。</p>"
            f"<img src='{chart_sig_med}' alt='signals medium'/>"
        )
    if charts:
        charts_html = "\n".join(charts)
    else:
        charts_html = "<p>(Charts skipped — no matplotlib)</p>"

    css = (
        "body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px;}"
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
    )
    thead = (
        "<thead><tr><th>agent_request_model</th><th>agent_pt</th>"
        "<th>tier（中文 / 枚举）</th><th>count</th><th>top_signal_codes</th></tr></thead>"
    )
    tier_legend = (
        "<h2>Tier 分层怎么读</h2>"
        "<ul class='tier-legend'>"
        "<li><strong>强怀疑（主证据）</strong> — 机器值 <code>high_precision</code>："
        "建议优先人工复核（多为 LLM eval 主证据）。"
        "若菜谱设 <code>high_precision_on_tool_fail_alone: true</code>，"
        "仅凭 tool 正则命中多次失败也可进此档。</li>"
        "<li><strong>待观察（弱证据）</strong> — <code>watchlist</code>："
        "启发式或单一弱证据，不宜直接等同「坏样本」。</li>"
        "<li><strong>未标记</strong> — <code>none</code>：未命中分层规则。</li>"
        "</ul>"
        "<p class='note'><code>high_precision</code> 在此项目中表示 "
        "「强怀疑需复核」而非「模型高精度」；jq / JSON 仍用英文枚举。</p>"
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
  Generated {html.escape(gen_at)} ·
  Input: <code>{html.escape(input_path)}</code><br/>
  Rows: <strong>{n_rows}</strong>
</div>
<h2>Tier counts</h2>
<table><thead><tr><th>展示名</th><th>机器枚举</th><th>条数</th></tr></thead>
<tbody>{tier_rows}</tbody></table>
{tier_legend}
<h2>Bad-case mining · 归因链</h2>
<p>下列说明各 <code>agent_bad_case_signals[].code</code> 依赖的 <strong>meta / stats</strong> 字段
及典型算子来源（自明支撑 bad case mining）。下方<strong>样本钻取</strong>中按条展示取值。</p>
{attribution_table}
{drilldown_html}
<h2>Charts</h2>
{charts_html}
<h2>Cohort detail (model × pt × tier)</h2>
<table>{thead}<tbody>{"".join(cohort_lines)}</tbody></table>
{sample_block}
<details>
<summary>Advanced / debugging</summary>
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
        default="Agent bad-case report",
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
        default=8,
        help="Max high_precision insight headlines to list (0=off)",
    )
    ap.add_argument(
        "--drilldown-limit",
        type=int,
        default=48,
        help="Max high/watchlist samples with expandable field drill-down (0=skip section)",
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
                "High-weight signals (this batch)",
                "#c0392b",
            )
        if med_c:
            chart_sig_med = _chart_signals(
                med_c,
                plt_mod,
                "Appendix: medium / heuristic signals",
                "#2980b9",
            )
        if len(model_tier) >= 1:
            chart_model = _chart_by_model(model_tier, plt_mod)

    samples: List[Tuple[str, str]] = []
    if args.sample_headlines > 0:
        samples = _insight_samples(rows, "high_precision", args.sample_headlines)

    drill_html = ""
    if args.drilldown_limit > 0:
        drill_rows = _collect_drilldown(rows, args.drilldown_limit)
        drill_html = _drilldown_section_html(drill_rows)

    page = _html_page(
        args.title,
        args.input,
        len(rows),
        tier_cnt,
        cohort,
        chart_tier,
        chart_model,
        chart_sig_high,
        chart_sig_med,
        att_html,
        samples,
        drill_html,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"Wrote {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
