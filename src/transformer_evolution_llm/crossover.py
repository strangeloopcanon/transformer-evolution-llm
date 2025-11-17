"""Crossover helpers for combining parent architectures and checkpoints."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from .dsl import ArchitectureSpec, BlockConfig
from .models import EvolutionModel


def splice_blocks(
    spec_a: ArchitectureSpec,
    spec_b: ArchitectureSpec,
    rng: random.Random,
) -> tuple[list[BlockConfig], int, int]:
    cut_a = rng.randrange(1, len(spec_a.model.blocks) + 1)
    cut_b = rng.randrange(1, len(spec_b.model.blocks) + 1)
    blocks = spec_a.model.blocks[:cut_a] + spec_b.model.blocks[cut_b:]
    return blocks, cut_a, cut_b


def crossover_specs(
    spec_a: ArchitectureSpec,
    spec_b: ArchitectureSpec,
    rng: random.Random,
) -> ArchitectureSpec:
    blocks, _, _ = splice_blocks(spec_a, spec_b, rng)
    data = spec_a.model_dump(mode="python")
    data["model"]["blocks"] = [block.model_dump(mode="python") for block in blocks]
    return ArchitectureSpec(**data)


def merge_checkpoints(
    child_spec: ArchitectureSpec,
    cut_a: int,
    cut_b: int,
    parent_a_blocks: int,
    parent_b_blocks: int,
    parent_a_ckpt: Path | None,
    parent_b_ckpt: Path | None,
    out_path: Path,
) -> Path | None:
    model = EvolutionModel(child_spec.model)
    child_state = model.state_dict()
    if parent_a_ckpt and parent_a_ckpt.exists():
        _transfer_blocks(
            child_state,
            torch.load(
                parent_a_ckpt, map_location="cpu"
            ),  # nosec B614 - loading trusted local checkpoints
            source_start=0,
            source_stop=cut_a,
            target_start=0,
        )
    if parent_b_ckpt and parent_b_ckpt.exists():
        _transfer_blocks(
            child_state,
            torch.load(parent_b_ckpt, map_location="cpu"),  # nosec B614 - trusted local checkpoints
            source_start=cut_b,
            source_stop=parent_b_blocks,
            target_start=cut_a,
        )
    torch.save(child_state, out_path)
    return out_path


def _transfer_blocks(
    child_state: dict[str, torch.Tensor],
    parent_state: dict[str, torch.Tensor],
    source_start: int,
    source_stop: int,
    target_start: int,
) -> None:
    for key, value in parent_state.items():
        if key.startswith("blocks."):
            idx = int(key.split(".")[1])
            if source_start <= idx < source_stop:
                new_idx = idx - source_start + target_start
                new_key = key.replace(f"blocks.{idx}.", f"blocks.{new_idx}.", 1)
                if new_key in child_state and child_state[new_key].shape == value.shape:
                    child_state[new_key] = value.clone()
        elif target_start == 0 and key in child_state and child_state[key].shape == value.shape:
            child_state[key] = value.clone()
