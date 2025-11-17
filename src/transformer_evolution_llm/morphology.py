"""Morphological utilities for expert alignment and crossover support."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from .models import EvolutionModel, MoELayer


def sort_moe_experts(model: EvolutionModel) -> None:
    for block in model.blocks:
        if isinstance(block.ffn, MoELayer):
            block.ffn.sort_experts()


def match_experts_to_parent(model: EvolutionModel, parent_state: dict[str, torch.Tensor]) -> None:
    """Align expert order with the parent state using norm proximity."""

    if not parent_state:
        return

    state = model.state_dict()
    for name, module in model.named_modules():
        if isinstance(module, MoELayer):
            expert_weights = []
            for idx in range(module.cfg.n_experts):
                key = f"{name}.experts.{idx}.net.fc1.weight"
                if key in state:
                    expert_weights.append(state[key].clone())
            parent_weights = []
            for idx in range(module.cfg.n_experts):
                key = f"{name}.experts.{idx}.net.fc1.weight"
                if key in parent_state:
                    parent_weights.append(parent_state[key].clone())
            if not parent_weights:
                continue
            permutation = _match_by_norm(parent_weights, expert_weights)
            module.reorder(permutation)


def _match_by_norm(
    parent_weights: Iterable[torch.Tensor],
    child_weights: Iterable[torch.Tensor],
) -> list[int]:
    parent_norms = [w.norm().item() for w in parent_weights]
    child_norms = [w.norm().item() for w in child_weights]
    order = sorted(
        range(len(child_norms)),
        key=lambda i: abs(child_norms[i] - parent_norms[min(i, len(parent_norms) - 1)]),
    )
    return order
