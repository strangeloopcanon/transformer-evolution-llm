"""Archive a run's frontier specs and optionally delete checkpoints.

Example:
  python scripts/archive_run.py runs/exp_longctx_unbiased_m4_20251216_115525 --delete-checkpoints
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import typer
import yaml

app = typer.Typer(help="Archive frontier specs to configs/frontiers and optionally delete run checkpoints.")


def _resolve_frontier(run_or_frontier: Path) -> tuple[Path, Path | None, Path | None]:
    if run_or_frontier.is_dir():
        run_dir = run_or_frontier
        frontier = run_dir / "frontier.json"
        lineage = run_dir / "frontier_lineage.json"
        ckpt_dir = run_dir / "checkpoints"
        return frontier, (lineage if lineage.exists() else None), (ckpt_dir if ckpt_dir.exists() else None)
    frontier = run_or_frontier
    lineage = frontier.with_name(frontier.stem + "_lineage.json")
    ckpt_dir = frontier.parent / "checkpoints"
    return frontier, (lineage if lineage.exists() else None), (ckpt_dir if ckpt_dir.exists() else None)


@app.command()
def main(
    run: Path = typer.Argument(..., exists=True, readable=True, help="Run directory or frontier.json path"),
    out_dir: Path | None = typer.Option(
        None,
        help="Destination directory for archived YAML specs (default: configs/frontiers/<run_name>).",
    ),
    delete_checkpoints: bool = typer.Option(
        False, help="If set, delete the run's checkpoint directory after archiving."
    ),
    dry_run: bool = typer.Option(False, help="Print planned actions without writing/deleting anything."),
) -> None:
    frontier_path, lineage_path, checkpoint_dir = _resolve_frontier(run)
    if not frontier_path.exists():
        raise typer.BadParameter(f"frontier.json not found at {frontier_path}")

    run_name = run.name if run.is_dir() else frontier_path.parent.name
    if out_dir is None:
        out_dir = Path("configs/frontiers") / run_name

    entries = json.loads(frontier_path.read_text())
    if not isinstance(entries, list) or not entries:
        raise typer.BadParameter(f"Frontier JSON must be a non-empty list: {frontier_path}")

    simplified: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        simplified.append(
            {
                "id": entry.get("id"),
                "parent": entry.get("parent"),
                "metrics": entry.get("metrics"),
                "spec": entry.get("spec"),
            }
        )

    if dry_run:
        typer.echo(f"[dry-run] would write {len(simplified)} YAML specs under {out_dir}")
        typer.echo(f"[dry-run] would write {out_dir / 'frontier_arch.json'}")
        if lineage_path is not None:
            typer.echo(f"[dry-run] lineage found at {lineage_path}")
        if delete_checkpoints:
            if checkpoint_dir is None:
                typer.echo("[dry-run] no checkpoint directory found")
            else:
                typer.echo(f"[dry-run] would delete {checkpoint_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frontier_arch.json").write_text(json.dumps(simplified, indent=2))
    for row in simplified:
        cid = str(row.get("id") or "").strip()
        spec = row.get("spec")
        if not cid or not isinstance(spec, dict):
            continue
        (out_dir / f"{cid}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    typer.echo(f"Archived {len(simplified)} frontier specs to {out_dir}")

    if delete_checkpoints and checkpoint_dir is not None:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        typer.echo(f"Deleted checkpoints at {checkpoint_dir}")


if __name__ == "__main__":
    app()

