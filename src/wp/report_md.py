from __future__ import annotations

from pathlib import Path

import pandas as pd


def _escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [col for col in columns if col in frame.columns]
    if not available:
        return ""
    header = "| " + " | ".join(available) + " |"
    divider = "| " + " | ".join(["---"] * len(available)) + " |"
    rows = []
    for _, row in frame[available].iterrows():
        rows.append("| " + " | ".join(_escape(row[col]) for col in available) + " |")
    return "\n".join([header, divider, *rows])


def render_markdown(top50: pd.DataFrame, output_path: str | Path, buy_plan: pd.DataFrame | None = None) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    buy_plan = buy_plan if buy_plan is not None else pd.DataFrame()
    if top50.empty:
        output.write_text("# WP 次日涨停概率 Top50\n\n无符合条件股票。\n", encoding="utf-8")
        return
    buy_columns = [
        "buy_rank",
        "portfolio_group",
        "suggest_position_pct",
        "ts_code",
        "name",
        "pct_chg",
        "sector_name",
        "p_limitup_t1",
        "wp_score",
        "risk_penalty_score",
        "buy_reason",
    ]
    columns = [
        "rank",
        "ts_code",
        "name",
        "pct_chg",
        "sector_name",
        "p_limitup_t1",
        "wp_score",
        "signal_level",
        "core_reason",
        "risk_reason",
    ]
    content = ["# WP 次日涨停概率 Top50", ""]
    content.append("## 14:20 尾盘买入观察计划")
    if buy_plan.empty:
        content.append("")
        content.append("当前无买入观察计划。")
    else:
        content.append("")
        content.append(_table(buy_plan, buy_columns))
    content.extend(["", "## Top50", "", _table(top50, columns), ""])
    output.write_text("\n".join(content), encoding="utf-8")
