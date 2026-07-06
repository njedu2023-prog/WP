import pandas as pd

from wp.ranking import rank_candidates


def test_ranking_orders_by_probability():
    df = pd.DataFrame(
        [
            {"ts_code": "A", "p_limitup_t1": 20, "wp_score": 50, "sector_strength_score": 50, "acceptance_score": 50, "amount": 1},
            {"ts_code": "B", "p_limitup_t1": 40, "wp_score": 50, "sector_strength_score": 50, "acceptance_score": 50, "amount": 1},
        ]
    )
    top, _ = rank_candidates(df, "now")
    assert top.iloc[0]["ts_code"] == "B"
