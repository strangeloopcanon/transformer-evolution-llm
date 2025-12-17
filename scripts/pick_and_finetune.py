"""Select top structurally-diverse candidates and run longer finetunes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from transformer_evolution_llm.dsl import ArchitectureSpec
from transformer_evolution_llm.trainer import FullWeightTrainer
from transformer_evolution_llm.data import DataModule
from transformer_evolution_llm.candidates import Candidate

app = typer.Typer(help="Pick diverse candidates and run long finetunes.")


def _load_frontier(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise typer.BadParameter("Frontier JSON must be a list of entries.")
    return data


def _diversity_score(entry: dict[str, Any]) -> tuple[int, int, float]:
    metrics = entry.get("metrics", {})
    layers = int(metrics.get("layers", 0))
    moe_blocks = int(metrics.get("moe_blocks", 0))
    novelty = float(metrics.get("novelty", 0.0))
    return (layers, moe_blocks, novelty)


@app.command()
def main(
    frontier: Path = typer.Argument(..., exists=True, readable=True),
    checkpoint_root: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("runs/long_finetunes.json")),
    top_n: int = typer.Option(6, min=1),
    steps: int = typer.Option(1500, min=1),
    eval_batches: int = typer.Option(8, min=1),
    device: str = typer.Option("mps"),
    instability_threshold: float = typer.Option(50.0),
    no_improve_patience: int = typer.Option(200),
    entropy_threshold: float = typer.Option(0.3),
    token_multiplier: float = typer.Option(1.0),
):
    entries = _load_frontier(frontier)
    entries_sorted = sorted(entries, key=_diversity_score, reverse=True)[:top_n]
    trainer = FullWeightTrainer(
        checkpoint_dir=checkpoint_root,
        steps=steps,
        eval_batches=eval_batches,
        device=device,
        entropy_threshold=entropy_threshold,
        entropy_patience=5,
        instability_threshold=instability_threshold,
        no_improve_patience=no_improve_patience,
        improvement_tolerance=1e-4,
    )
    results = []
    for entry in entries_sorted:
        cid = entry["id"]
        spec = ArchitectureSpec(**entry["spec"])
        ckpt = checkpoint_root / f"{cid}.pt"
        if not ckpt.exists():
            typer.echo(f"[warn] missing checkpoint for {cid}; skipping")
            continue
        data_module = DataModule(spec.data, seed=int(getattr(spec.train, "seed", 0) or 0))
        tokens_per_batch = int(spec.data.batch_size) * int(spec.data.seq_len)
        max_tokens = spec.train.max_tokens or (trainer.steps * tokens_per_batch)
        max_tokens = int(max_tokens * token_multiplier)
        batches = data_module.batches(max_tokens=max_tokens)
        candidate = Candidate(ident=f"{cid}__long", spec=spec)
        metrics, new_ckpt = trainer.train(candidate, spec, batches, seed_state_path=ckpt)
        results.append(
            {
                "id": cid,
                "long_finetune_metrics": metrics,
                "checkpoint": str(new_ckpt),
            }
        )
        # remove long finetune checkpoint to save space
        if new_ckpt.exists():
            new_ckpt.unlink(missing_ok=True)
    out.write_text(json.dumps(results, indent=2))
    typer.echo(f"Wrote long finetune results for {len(results)} candidates to {out}")


if __name__ == "__main__":
    app()
