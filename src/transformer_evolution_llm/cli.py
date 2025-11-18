"""Command-line utilities for the transformer_evolution_llm package."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import ujson as json
from rich.console import Console
from rich.table import Table

from . import get_version
from .api import export_frontier_seed, run_evolution
from .cache_builder import synthesize_cache

app = typer.Typer(help="Evolutionary search loop utilities")
console = Console()


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(get_version())


@app.command()
def run(
    config: Annotated[Path, typer.Argument(..., exists=True, readable=True)],
    generations: Annotated[int, typer.Option(min=1)] = 4,
    mode: Annotated[str, typer.Option(help="Evaluation backend (simulate/full).")] = "simulate",
    seed: Annotated[int, typer.Option()] = 0,
    out: Annotated[Path, typer.Option()] = Path("runs/frontier.json"),
) -> None:
    """Execute an evolutionary search run."""
    run_evolution(config_path=config, generations=generations, mode=mode, seed=seed, out_path=out)


@app.command("cache")
def cache_cmd(
    out_dir: Annotated[Path, typer.Argument()] = Path("cache/phi_tiny_topk16"),
    samples: Annotated[int, typer.Option(min=1)] = 128,
    seq_len: Annotated[int, typer.Option(min=16)] = 2048,
    topk: Annotated[int, typer.Option(min=1)] = 8,
    vocab: Annotated[int, typer.Option(min=256)] = 100_352,
) -> None:
    """Build a synthetic teacher logit cache for pipeline testing."""
    synthesize_cache(out_dir, samples=samples, seq_len=seq_len, topk=topk, vocab=vocab)


@app.command()
def frontier(path: Annotated[Path, typer.Argument()] = Path("runs/frontier.json")) -> None:
    """Print the latest Pareto frontier file."""
    if not path.exists():
        raise typer.BadParameter(f"{path} does not exist")
    data = json.loads(path.read_text())
    table = Table(title=f"Frontier ({path})")
    table.add_column("ID")
    table.add_column("Parent")
    table.add_column("ppl_code")
    table.add_column("throughput")
    for row in data:
        table.add_row(
            row["id"],
            str(row.get("parent", "-")),
            f"{row.get('metrics', {}).get('ppl_code', 0.0):.2f}",
            f"{row.get('metrics', {}).get('throughput', 0.0):.2f}",
        )
    console.print(table)


@app.command("export-seed")
def export_seed(
    frontier_path: Annotated[Path, typer.Argument(help="Path to Pareto frontier JSON file.")],
    candidate_id: Annotated[str, typer.Argument(help="Candidate identifier to export.")],
    out: Annotated[
        Path,
        typer.Argument(help="Destination config path (e.g., configs/seed_<id>.yaml)."),
    ],
    checkpoint_dir: Annotated[
        Path,
        typer.Option(help="Directory containing candidate checkpoints."),
    ] = Path("runs/checkpoints"),
) -> None:
    """Export a frontier candidate spec + checkpoint pointer as a seed config."""
    export_frontier_seed(
        frontier_path=frontier_path,
        candidate_id=candidate_id,
        out_path=out,
        checkpoint_dir=checkpoint_dir,
    )
    console.print(f"[bold green]Seed config written:[/] {out}")


def main() -> None:
    """Entry point for `python -m transformer_evolution_llm.cli`."""
    app()


if __name__ == "__main__":
    main()
