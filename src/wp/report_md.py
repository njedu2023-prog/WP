from __future__ import annotations

from pathlib import Path

import pandas as pd


def _escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(top50: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if top50.empty:
        output.write_text("# WP 次日涨停概率 Top50\n\n无符合条件股票。\n", encoding="utf-8")
        return
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
    available = [col for col in columns if col in top50.columns]
    header = "| " + " | ".join(available) + " |"
    divider = "| " + " | ".join(["---"] * len(available)) + " |"
    rows = []
    for _, row in top50[available].iterrows():
        rows.append("| " + " | ".join(_escape(row[col]) for col in available) + " |")
    output.write_text("# WP 次日涨停概率 Top50\n\n" + "\n".join([header, divider, *rows]) + "\n", encoding="utf-8")
