from pathlib import Path

import ujson as json
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


def test_cli_export_seed_from_frontier(tmp_path: Path, tiny_spec, monkeypatch):
    runner = CliRunner()
    # Prepare frontier with a single completed candidate.
    runs_dir = tmp_path / "runs"
    ckpt_dir = runs_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    candidate_id = "seed-1-abcd"
    checkpoint_path = ckpt_dir / f"{candidate_id}.pt"
    checkpoint_path.write_bytes(b"dummy")

    frontier_path = runs_dir / "frontier.json"
    frontier_entry = {
        "id": candidate_id,
        "parent": None,
        "rung": 2,
        "status": "completed",
        "metrics": {},
        "spec": tiny_spec.model_dump(mode="python"),
    }
    frontier_path.write_text(json.dumps([frontier_entry]))

    configs_dir = tmp_path / "configs"
    out_cfg = configs_dir / "seed_candidate.yaml"

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "export-seed",
            str(frontier_path),
            candidate_id,
            str(out_cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_cfg.exists()
    exported_spec = api.load_spec(out_cfg)
    expected_ckpt = Path("runs/checkpoints") / f"{candidate_id}.pt"
    assert exported_spec.train.init_checkpoint == str(expected_ckpt)
