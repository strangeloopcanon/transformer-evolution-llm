"""Public API for downstream modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ujson as json

from .dsl import ArchitectureSpec, load_architecture_spec, save_architecture_spec
from .orchestrator import EvolutionRunner

__all__ = [
    "ArchitectureSpec",
    "load_spec",
    "save_spec",
    "run_evolution",
    "export_frontier_seed",
    "prune_checkpoints",
]


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


def export_frontier_seed(
    frontier_path: str | Path,
    candidate_id: str,
    out_path: str | Path,
    checkpoint_dir: str | Path = "runs/checkpoints",
) -> None:
    """Export a frontier candidate as a reusable seed config.

    The exported spec includes a train.init_checkpoint field pointing to the
    expected checkpoint for the selected candidate so future live runs can
    start from learned weights instead of reinitializing.
    """
    frontier_path = Path(frontier_path)
    if not frontier_path.exists():
        msg = f"Frontier file not found: {frontier_path}"
        raise FileNotFoundError(msg)
    data: list[dict[str, Any]] = json.loads(frontier_path.read_text())
    match = next((row for row in data if row.get("id") == candidate_id), None)
    if match is None:
        msg = f"Candidate {candidate_id} not found in {frontier_path}"
        raise ValueError(msg)
    spec_data = match.get("spec")
    if not isinstance(spec_data, dict):
        msg = f"Frontier entry for {candidate_id} missing spec payload"
        raise ValueError(msg)
    spec = ArchitectureSpec(**spec_data)
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir_path / f"{candidate_id}.pt"
    if not checkpoint_path.exists():
        msg = f"Checkpoint for {candidate_id} not found at {checkpoint_path}"
        raise FileNotFoundError(msg)
    spec.train.init_checkpoint = str(checkpoint_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_architecture_spec(spec, out_path)


def _ids_from_state(state_path: Path) -> set[str]:
    """Collect candidate ids referenced by a saved state file."""
    state: dict[str, Any] = json.loads(state_path.read_text())
    keep: set[str] = set()
    for key in ("frontier", "pool", "parents"):
        items = state.get(key, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    keep.add(item)
    history = state.get("history", [])
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict):
                cid = item.get("id")
                if cid:
                    keep.add(cid)
            elif isinstance(item, str):
                keep.add(item)
    return keep


def _ids_from_frontier(frontier_path: Path) -> set[str]:
    """Collect candidate ids referenced by a frontier file."""
    data: list[dict[str, Any]] = json.loads(frontier_path.read_text())
    keep: set[str] = set()
    for row in data:
        cid = row.get("id")
        if cid:
            keep.add(cid)
    return keep


def prune_checkpoints(
    checkpoint_dir: str | Path,
    frontier_path: str | Path | None = None,
    state_path: str | Path | None = None,
    dry_run: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Remove checkpoints not referenced by a frontier/state.

    Returns (kept, removed) lists of paths.
    """
    if frontier_path is None and state_path is None:
        msg = "Provide at least one of frontier_path or state_path"
        raise ValueError(msg)
    checkpoint_dir_path = Path(checkpoint_dir)
    if not checkpoint_dir_path.exists():
        msg = f"Checkpoint directory not found: {checkpoint_dir_path}"
        raise FileNotFoundError(msg)

    keep_ids: set[str] = set()
    if frontier_path is not None:
        frontier_path = Path(frontier_path)
        if not frontier_path.exists():
            msg = f"Frontier file not found: {frontier_path}"
            raise FileNotFoundError(msg)
        keep_ids.update(_ids_from_frontier(frontier_path))
    if state_path is not None:
        state_path = Path(state_path)
        if not state_path.exists():
            msg = f"State file not found: {state_path}"
            raise FileNotFoundError(msg)
        keep_ids.update(_ids_from_state(state_path))

    kept: list[Path] = []
    removed: list[Path] = []
    for path in checkpoint_dir_path.glob("*.pt"):
        cid = path.stem
        if cid in keep_ids:
            kept.append(path)
        else:
            removed.append(path)
            if not dry_run:
                path.unlink(missing_ok=True)
    return kept, removed
