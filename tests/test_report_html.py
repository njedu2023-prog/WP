import pandas as pd

from wp.report_html import render_html


def test_report_html_contains_title(tmp_path):
    path = tmp_path / "latest.html"
    render_html(pd.DataFrame(), pd.DataFrame(), {"status": "无符合条件股票", "data_time": "now"}, path)
    assert "WP Top50" in path.read_text(encoding="utf-8")
