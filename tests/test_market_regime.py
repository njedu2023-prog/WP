import pandas as pd

from wp.market_regime import assess_market_regime


def _market(pct_values):
    return pd.DataFrame(
        [
            {
                "ts_code": f"600{index:03d}.SH",
                "pct_chg": pct,
                "suspended_flag": 0,
                "data_quality_flag": 0,
                "today_limitup": pct >= 9.5,
                "price": 10.0,
                "amount": 100_000_000,
            }
            for index, pct in enumerate(pct_values)
        ]
    )


def test_market_regime_allows_strong_breadth_and_avoids_weak_breadth():
    strong = assess_market_regime(_market([2.0] * 220 + [-1.0] * 30), pd.DataFrame([{"sector_name": "电力"}]))
    weak = assess_market_regime(_market([-4.0] * 220 + [1.0] * 30), pd.DataFrame())
    assert strong["state"] == "允许寻找机会"
    assert weak["state"] == "回避"
    assert strong["manual_decision_support_only"] is True


def test_market_regime_rejects_incomplete_universe():
    result = assess_market_regime(_market([1.0] * 20), pd.DataFrame())
    assert result["state"] == "数据不足"
    assert result["manual_action"] == "建议空仓"
