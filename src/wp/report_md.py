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


def render_markdown(
    top50: pd.DataFrame,
    output_path: str | Path,
    buy_plan: pd.DataFrame | None = None,
    observation_pool: pd.DataFrame | None = None,
    decision_support: dict | None = None,
    exit_guidance: pd.DataFrame | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    buy_plan = buy_plan if buy_plan is not None else pd.DataFrame()
    observation_pool = observation_pool if observation_pool is not None else buy_plan
    decision_support = decision_support or {}
    exit_guidance = exit_guidance if exit_guidance is not None else pd.DataFrame()
    observation_columns = [
        "quality_rank",
        "observation_status",
        "rank_change",
        "first_seen",
        "ts_code",
        "name",
        "pct_chg",
        "sector_name",
        "tail_profit_score",
        "risk_penalty_score",
        "amount_ratio_5d",
        "limit_rule_pct",
        "last_seen",
        "buy_reason",
    ]
    columns = [
        "rank",
        "ts_code",
        "name",
        "pct_chg",
        "sector_name",
        "tail_profit_score",
        "tail_profit_eligible",
        "p_limitup_t1",
        "wp_score",
        "signal_level",
        "core_reason",
        "risk_reason",
    ]
    content = ["# WP Top50", ""]
    content.extend(
        [
            "## WP V2 人工决策辅助",
            "",
            "仅辅助人工下单，不接入券商、不读取账户、不自动交易。",
            "",
            f"- 当前建议：{_escape(decision_support.get('action', '继续观察'))}",
            f"- 当前首选：{_escape(decision_support.get('candidate_name', ''))} {_escape(decision_support.get('candidate_code', ''))}",
            f"- 判断依据：{_escape(decision_support.get('reason', ''))}",
            "",
        ]
    )
    content.append("## 尾盘观察")
    if observation_pool.empty:
        content.append("")
        content.append("当前无具备资格的尾盘观察票。")
    else:
        content.append("")
        content.append(_table(observation_pool, observation_columns))
    content.extend(["", "## T+1 人工卖出建议", ""])
    if exit_guidance.empty:
        content.append("今天没有需要复核的系统观察记录；实际持仓以人工确认为准。")
    else:
        content.append(
            _table(
                exit_guidance,
                ["ts_code", "name", "open_return_pct", "current_return_pct", "guidance_action", "guidance_reason", "next_checkpoint"],
            )
        )
    content.extend(["", "## Top50", ""])
    content.append(_table(top50, columns) if not top50.empty else "无符合条件股票。")
    content.append("")
    output.write_text("\n".join(content), encoding="utf-8")
