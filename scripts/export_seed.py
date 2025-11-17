"""Export a frontier candidate to a reusable YAML seed (and optional checkpoint).

Usage:
  python scripts/export_seed.py runs/frontier_phi_loop_long.json --id mutate_topk-30-bbb0 \
    --out-config configs/seed_mutate_topk.yaml \
    --checkpoint-dir runs/checkpoints_phi_loop_long \
    --out-checkpoint runs/seeds/mutate_topk-30-bbb0.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import ujson as json
import yaml


app = typer.Typer(help="Export a frontier candidate spec (and checkpoint) as a new seed.")


@app.command()
def main(
    frontier: Path = typer.Argument(..., exists=True, readable=True, help="Path to frontier_*.json"),
    id: str = typer.Option(..., "--id", help="Candidate id to export (e.g., xover-58-d356)"),
    out_config: Path = typer.Option(
        Path("configs/exported_seed.yaml"), exists=False, help="Destination YAML config path"
    ),
    checkpoint_dir: Optional[Path] = typer.Option(
        None, help="Directory containing per-candidate checkpoints (optional)"
    ),
    out_checkpoint: Optional[Path] = typer.Option(
        None, help="Destination path to copy the candidate checkpoint (optional)"
    ),
) -> None:
    """Export the candidate's spec to YAML and optionally copy its checkpoint."""
    data = json.loads(frontier.read_text())
    cand = next((c for c in data if c.get("id") == id), None)
    if cand is None:
        raise typer.BadParameter(f"Candidate id {id} not found in {frontier}")
    spec = cand.get("spec")
    if not spec:
        raise typer.BadParameter("Candidate spec missing from frontier entry")
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(spec, sort_keys=False))
    typer.echo(f"Wrote seed config to {out_config}")

    if checkpoint_dir and out_checkpoint:
        src = checkpoint_dir / f"{id}.pt"
        if src.exists():
            out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            out_checkpoint.write_bytes(src.read_bytes())
            typer.echo(f"Copied checkpoint to {out_checkpoint}")
        else:
            typer.echo(f"Checkpoint not found at {src}; skipped copy")


if __name__ == "__main__":
    app()

