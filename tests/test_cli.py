from pathlib import Path

from typer.testing import CliRunner

from transformer_evolution_llm import api
from transformer_evolution_llm.cli import app


def test_cli_run_and_frontier(tmp_path: Path, tiny_spec):
    cfg = tmp_path / "spec.yaml"
    api.save_spec(tiny_spec, cfg)
    out = tmp_path / "frontier.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(cfg),
            "--generations",
            "1",
            "--mode",
            "simulate",
            "--seed",
            "2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    result_frontier = runner.invoke(app, ["frontier", str(out)])
    assert result_frontier.exit_code == 0, result_frontier.output


def test_cli_cache(tmp_path: Path):
    runner = CliRunner()
    out_dir = tmp_path / "cache"
    result = runner.invoke(
        app,
        [
            "cache",
            str(out_dir),
            "--samples",
            "4",
            "--seq-len",
            "16",
            "--topk",
            "2",
            "--vocab",
            "256",
        ],
    )
    assert result.exit_code == 0, result.output
    assert list(out_dir.glob("*.npz"))
