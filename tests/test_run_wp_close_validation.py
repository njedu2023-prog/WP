import json
from types import SimpleNamespace

from scripts import run_wp_close_validation


def test_v3_close_validation_commits_only_when_truth_state_changes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    ledger = tmp_path / "outputs" / "json" / "wp_v3_candidate_ledger.json"
    registry = tmp_path / "outputs" / "json" / "wp_model_registry_v3.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"sessions":[]}\n', encoding="utf-8")
    registry.write_text('{"models":[]}\n', encoding="utf-8")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:3] == [
            run_wp_close_validation.sys.executable,
            "-m",
            "wp.close_validation",
        ]:
            ledger.write_text(
                '{"sessions":[{"truth_status":"verified"}]}\n',
                encoding="utf-8",
            )
            payload = ledger.with_name("wp_buy_plan_validation.json")
            payload.write_text(
                json.dumps(
                    {
                        "summary": {
                            "pending_count": 3,
                            "pending_due_count": 0,
                        }
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_wp_close_validation.subprocess, "run", fake_run)

    assert run_wp_close_validation.run_once() == 0
    assert len(commands) == 2
    assert "Validate WP V3 next-day close" in commands[1]
