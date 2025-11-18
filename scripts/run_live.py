"""Utility to run live evolutionary sweeps with custom trainer settings."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import typer

from transformer_evolution_llm.api import load_spec
from transformer_evolution_llm.data import DataModule
from transformer_evolution_llm.orchestrator import EvolutionRunner
from transformer_evolution_llm.trainer import FullWeightTrainer

app = typer.Typer(help="Launch live (full-weight) evolutionary runs with custom trainer settings.")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _detect_attention_impl(device: str) -> str:
    """Best-effort attention kernel description for audit logs."""
    try:
        import torch

        if torch.backends.cuda.is_built() and device.startswith("cuda"):
            try:
                from torch.backends.cuda import sdp_kernel

                if hasattr(sdp_kernel, "is_flash_sdp_available") and sdp_kernel.is_flash_sdp_available():
                    return "flash"
                if hasattr(sdp_kernel, "is_mem_efficient_sdp_available") and sdp_kernel.is_mem_efficient_sdp_available():
                    return "sdpa"
                return "cuda_math"
            except Exception:
                return "cuda_sdpa"
        if device.startswith("mps"):
            return "mps_mha"
    except Exception:
        pass
    return "vanilla_mha"


@app.command()
def main(
    config: Path = typer.Argument(..., exists=True, readable=True, help="YAML/JSON DSL config"),
    generations: int = typer.Option(18, min=1),
    steps: int = typer.Option(120, min=1, help="Gradient steps per candidate."),
    eval_batches: int = typer.Option(4, min=1),
    device: str = typer.Option("mps", help="Device for training (mps/cpu/cuda)."),
    out: Path = typer.Option(Path("runs/frontier_live.json")),
    checkpoint_dir: Path = typer.Option(Path("runs/checkpoints_live")),
    seed: int = typer.Option(0),
    cleanup_old_checkpoints: bool = typer.Option(
        True,
        help="After the run, delete other checkpoint directories (names starting with 'checkpoints') in the same parent folder.",
    ),
    lineage_out: Path | None = typer.Option(
        None,
        help="Optional path to write a full lineage JSON (all candidates, parent links).",
    ),
    score_weight_ppl: float = typer.Option(1.0, help="Weight for ppl_code/ppl_math in parent selection."),
    score_weight_throughput: float = typer.Option(1.0, help="Weight for throughput in parent selection."),
    score_weight_long_recall: float = typer.Option(1.0, help="Weight for long_recall in parent selection."),
    score_weight_ram: float = typer.Option(1.0, help="Weight for RAM in parent selection."),
    score_weight_layers: float = typer.Option(1.0, help="Weight for layer count in parent selection."),
    score_weight_moe: float = typer.Option(1.0, help="Weight for MoE block count in parent selection."),
    score_weight_novelty: float = typer.Option(1.0, help="Weight for novelty in parent selection."),
    score_weight_instability: float = typer.Option(1.0, help="Weight for instability (minimize) in parent selection."),
    parent_selection: str = typer.Option(
        "weighted",
        help="Parent selection strategy: weighted | pareto_uniform | lexicase",
    ),
    score_weight_prior: float = typer.Option(
        0.0, help="Weight for prior_distance (minimize). 0 disables its effect."
    ),
) -> None:
    """Run an evolutionary sweep with the live trainer."""
    spec = load_spec(config)
    score_weights = {
        "ppl_code": score_weight_ppl,
        "ppl_math": score_weight_ppl,
        "throughput": score_weight_throughput,
        "long_recall": score_weight_long_recall,
        "ram": score_weight_ram,
        "layers": score_weight_layers,
        "moe_blocks": score_weight_moe,
        "novelty": score_weight_novelty,
        "instability": score_weight_instability,
    }
    if score_weight_prior != 0.0:
        score_weights["prior_distance"] = score_weight_prior
    runner = EvolutionRunner(
        spec,
        spec.evolution,
        mode="live",
        seed=seed,
        score_weight_overrides=score_weights,
    )
    # Override selection strategy from CLI if provided
    try:
        runner.cfg.parent_selection = parent_selection  # type: ignore[attr-defined]
    except Exception:
        pass
    train_cfg = spec.train
    runner.trainer = FullWeightTrainer(
        steps=steps,
        eval_batches=eval_batches,
        checkpoint_dir=checkpoint_dir,
        device=device,
        entropy_threshold=train_cfg.entropy_threshold,
        entropy_patience=train_cfg.entropy_patience,
        instability_threshold=train_cfg.instability_threshold,
        no_improve_patience=train_cfg.no_improve_patience,
        improvement_tolerance=train_cfg.improvement_tolerance,
    )
    runner.data_module = DataModule(spec.data)
    # Log device/capabilities (best effort)
    try:
        import torch

        dev = device
        msg = f"[runner] device={dev}"
        impl = _detect_attention_impl(dev)
        msg += f", attention_impl={impl}"
        typer.echo(msg)
    except Exception:
        pass
    runner.checkpoint_dir = checkpoint_dir
    runner.run(generations=generations)
    runner.save_frontier(out)
    if lineage_out is None:
        # default next to frontier
        lineage_out = out.with_name(out.stem + "_lineage.json")
    try:
        runner.save_lineage(lineage_out)
    except Exception:
        pass
    manifest_path = out.with_name(out.stem + ".manifest.json")
    manifest = {
        "config": str(config.resolve()),
        "generations": generations,
        "steps": steps,
        "eval_batches": eval_batches,
        "device": device,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "seed": seed,
        "parent_selection": parent_selection,
        "score_weights": {
            "ppl": score_weight_ppl,
            "throughput": score_weight_throughput,
            "long_recall": score_weight_long_recall,
            "ram": score_weight_ram,
            "layers": score_weight_layers,
            "moe_blocks": score_weight_moe,
            "novelty": score_weight_novelty,
            "instability": score_weight_instability,
            "prior_distance": score_weight_prior,
        },
        "attention_impl": _detect_attention_impl(device),
        "cleanup_old_checkpoints": cleanup_old_checkpoints,
        "frontier": str(out.resolve()),
        "lineage": str(lineage_out.resolve()),
    }
    commit = _git_commit()
    if commit:
        manifest["git_commit"] = commit
    manifest_path.write_text(json.dumps(manifest, indent=2))
    typer.echo(f"Frontier written to {out}")
    if cleanup_old_checkpoints:
        _cleanup_old_checkpoint_roots(checkpoint_dir)


def _cleanup_old_checkpoint_roots(current_root: Path) -> None:
    parent = current_root.parent
    if not parent.exists():
        return
    removed: list[Path] = []
    for sibling in parent.iterdir():
        if sibling == current_root or not sibling.is_dir():
            continue
        if not sibling.name.startswith("checkpoints"):
            continue
        shutil.rmtree(sibling, ignore_errors=True)
        removed.append(sibling)
    if removed:
        formatted = ", ".join(str(path) for path in removed)
        typer.echo(f"Removed old checkpoint dirs: {formatted}")


if __name__ == "__main__":
    app()
