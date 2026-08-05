from scripts.prune_wp_runtime_cache import prune_directory


def test_prune_directory_keeps_rolling_date_window(tmp_path):
    cache = tmp_path / "daily"
    cache.mkdir()
    for day in range(1, 61):
        (cache / f"2026{day:04d}__digest.parquet").write_text(
            "cache",
            encoding="utf-8",
        )
    (cache / "schema.json").write_text("{}", encoding="utf-8")

    kept, removed = prune_directory(cache, keep_date_count=50)

    assert kept == 51
    assert removed == 10
    assert (cache / "20260011__digest.parquet").exists()
    assert not (cache / "20260010__digest.parquet").exists()
    assert (cache / "schema.json").exists()
