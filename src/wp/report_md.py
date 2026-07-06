from __future__ import annotations

from pathlib import Path

import pandas as pd


def render_markdown(top50: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if top50.empty:
        output.write_text("# WP 次日涨停概率 Top50\n\n无符合条件股票。\n", encoding="utf-8")
        return
    output.write_text("# WP 次日涨停概率 Top50\n\n" + top50.to_markdown(index=False), encoding="utf-8")
