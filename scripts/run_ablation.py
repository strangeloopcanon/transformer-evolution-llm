"""Run targeted ablations on top-N frontier entries and summarize deltas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from transformer_evolution_llm.ablation import ABLATIONS, apply_ablation
from transformer_evolution_llm.candidates import Candidate
from transformer_evolution_llm.data import DataModule
from transformer_evolution_llm.dsl import ArchitectureSpec
from transformer_evolution_llm.trainer import FullWeightTrainer

app = typer.Typer(help="Evaluate ablations (retro off, kv=1, etc.) for top-N frontier entries.")


def _load_frontier(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise typer.BadParameter("Frontier JSON must be a list of entries.")
    return data


def _select_entries(
    entries: list[dict[str, Any]], metric: str, top_n: int, ids: list[str] | None
) -> list[dict[str, Any]]:
    if ids:
        selected = []
        for ident in ids:
            entry = next((e for e in entries if e.get("id") == ident), None)
            if entry is None:
                raise typer.BadParameter(f"Candidate id {ident} not found in frontier.")
            selected.append(entry)
        return selected
    sorted_entries = sorted(entries, key=lambda e: float(e.get("metrics", {}).get(metric, 1e9)))
    return sorted_entries[:top_n]


@app.command()
def main(
    frontier: Path = typer.Argument(..., exists=True, readable=True, help="frontier_*.json"),
    out: Path = typer.Option(Path("runs/ablation_report.json")),
    device: str = typer.Option("mps"),
    steps: int = typer.Option(80, min=1),
    eval_batches: int = typer.Option(2, min=1),
    checkpoint_dir: Path = typer.Option(Path("runs/checkpoints_ablation")),
    metric: str = typer.Option("ppl_code", help="Metric used to sort top-N when ids not specified."),
    top_n: int = typer.Option(3, min=1),
    ids: list[str] = typer.Option(None, "--id", help="Explicit candidate ids to ablate."),
    ablations: list[str] = typer.Option(None, "--ablation", help="Subset of ablations to run."),
) -> None:
    entries = _load_frontier(frontier)
    subset = _select_entries(entries, metric=metric, top_n=top_n, ids=ids)
    if not subset:
        raise typer.BadParameter("No candidates selected for ablation.")
    todo = ablations or ABLATIONS

    trainer = FullWeightTrainer(
        steps=steps,
        eval_batches=eval_batches,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    report: list[dict[str, Any]] = []
    key_metrics = ["ppl_code", "long_recall", "throughput"]

    for entry in subset:
        base_id = entry["id"]
        spec = ArchitectureSpec(**entry["spec"])
        base_metrics = entry.get("metrics", {})
        typer.echo(f"[base] {base_id} -> running ablations {todo}")
        for ab in todo:
            mutated = apply_ablation(spec, ab)
            candidate = Candidate(ident=f"{base_id}__{ab}", spec=mutated)
            data_module = DataModule(mutated.data)
            iterator = data_module.batches(max_tokens=mutated.train.max_tokens)
            metrics, checkpoint = trainer.train(
                candidate=candidate,
                spec=mutated,
                batch_iter=iterator,
                seed_state_path=None,
            )
            # remove checkpoint to keep disk clean
            if checkpoint.exists():
                checkpoint.unlink(missing_ok=True)
            deltas = {
                key: metrics.get(key, float("nan")) - base_metrics.get(key, 0.0)
                for key in key_metrics
            }
            report.append(
                {
                    "base_id": base_id,
                    "ablation": ab,
                    "base_metrics": {k: base_metrics.get(k) for k in key_metrics},
                    "metrics": {k: metrics.get(k) for k in key_metrics},
                    "deltas": deltas,
                }
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    typer.echo(f"Wrote ablation report ({len(report)} entries) to {out}")


if __name__ == "__main__":
    app()

