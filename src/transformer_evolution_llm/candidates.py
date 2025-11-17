"""Candidate and frontier tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .dsl import ArchitectureSpec

Status = Literal["pending", "running", "completed", "failed"]


@dataclass
class Candidate:
    """Represents a single architecture variant."""

    ident: str
    spec: ArchitectureSpec
    parent: str | None = None
    parent_checkpoint: Path | None = None
    seed_state_path: Path | None = None
    rung: int = 0
    status: Status = "pending"
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint: Path | None = None

    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value

    def score(self, weights: dict[str, float]) -> float:
        return sum(self.metrics.get(k, 0.0) * w for k, w in weights.items())

    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.ident,
            "parent": self.parent,
            "rung": self.rung,
            "status": self.status,
            "metrics": self.metrics,
            "spec": self.spec.model_dump(mode="python"),
        }


ObjectiveDirection = Literal["max", "min"]


class ParetoFrontier:
    """Maintains a Pareto-optimal set of candidates."""

    def __init__(self, objectives: dict[str, ObjectiveDirection]) -> None:
        self.objectives = objectives
        self._entries: list[Candidate] = []

    @property
    def entries(self) -> list[Candidate]:
        return list(self._entries)

    def update(self, candidate: Candidate) -> None:
        dominated = []
        for idx, existing in enumerate(self._entries):
            if self._dominates(candidate, existing):
                dominated.append(idx)
            elif self._dominates(existing, candidate):
                return
        for idx in reversed(dominated):
            del self._entries[idx]
        self._entries.append(candidate)

    def _dominates(self, a: Candidate, b: Candidate) -> bool:
        better_or_equal = True
        strictly_better = False
        for metric, direction in self.objectives.items():
            a_val = a.metrics.get(metric)
            b_val = b.metrics.get(metric)
            if a_val is None or b_val is None:
                continue
            if direction == "max":
                better = a_val >= b_val
                strictly = a_val > b_val
            else:
                better = a_val <= b_val
                strictly = a_val < b_val
            if not better:
                better_or_equal = False
                break
            if strictly:
                strictly_better = True
        return better_or_equal and strictly_better

    def to_json(self) -> list[dict[str, Any]]:
        return [entry.serialize() for entry in self._entries]
