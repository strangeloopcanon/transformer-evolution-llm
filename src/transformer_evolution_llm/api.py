"""Public API for downstream modules."""

from __future__ import annotations

from pathlib import Path

from .dsl import ArchitectureSpec, load_architecture_spec, save_architecture_spec
from .orchestrator import EvolutionRunner

__all__ = ["ArchitectureSpec", "load_spec", "save_spec", "run_evolution"]


def load_spec(path: str | Path) -> ArchitectureSpec:
    """Read an architecture spec from disk."""
    return load_architecture_spec(path)


def save_spec(spec: ArchitectureSpec, path: str | Path) -> None:
    """Persist an architecture spec to disk."""
    save_architecture_spec(spec, path)


def run_evolution(
    config_path: str | Path,
    generations: int,
    mode: str = "simulate",
    seed: int = 0,
    out_path: str | Path = "runs/frontier.json",
) -> None:
    """Entry point used by the CLI to run a full search."""
    spec = load_spec(config_path)
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode=mode,
        seed=seed,
    )
    runner.run(generations)
    runner.save_frontier(Path(out_path))
