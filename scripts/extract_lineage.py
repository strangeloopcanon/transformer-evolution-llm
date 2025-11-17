"""Reconstruct a partial lineage from a run log and a frontier JSON.

For runs that predate built-in lineage capture, this script parses the
"Evaluating <id>" lines from a log and merges metrics/spec from the
frontier JSON when available.

Usage:
  python scripts/extract_lineage.py \
    --log runs/creative_unbiased.log \
    --frontier runs/frontier_phi_creative_unbiased.json \
    --out runs/frontier_phi_creative_unbiased_lineage_from_log.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import typer


app = typer.Typer(help="Extract partial lineage from a run log + frontier JSON.")


@app.command()
def main(
    log: Path = typer.Option(..., exists=True, readable=True),
    frontier: Optional[Path] = typer.Option(None),
    out: Path = typer.Option(...),
) -> None:
    text = log.read_text(errors="ignore")
    ids = re.findall(r"Evaluating\s+([\w-]+)", text)
    by_id: dict[str, dict] = {}
    if frontier and frontier.exists():
        data = json.loads(frontier.read_text())
        for entry in data:
            by_id[entry.get("id", "")] = entry
    nodes = []
    for cid in ids:
        info = by_id.get(cid)
        node = {
            "id": cid,
            "parents": [],  # unknown without built-in capture
            "status": info.get("status") if info else None,
            "rung": info.get("rung") if info else None,
            "metrics": info.get("metrics") if info else {},
            "spec": info.get("spec") if info else None,
        }
        nodes.append(node)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nodes, indent=2))
    typer.echo(f"Wrote partial lineage to {out} ({len(nodes)} nodes)")


if __name__ == "__main__":
    app()

