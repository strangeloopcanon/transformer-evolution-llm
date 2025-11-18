"""Registration hooks for plugin-style architectural components.

This module provides a minimal, opt-in registry so external packages can
attach custom blocks (e.g., attention variants, adapters) without
editing the core model files. Registration happens via simple Python
imports and does not affect behavior unless a matching plugin name is
present in the config.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .dsl import CustomModuleConfig


class ComponentBuilder(Protocol):
    """Callable that builds a module for a custom component."""

    def __call__(self, cfg: CustomModuleConfig, dim: int):
        ...


_REGISTRY: dict[str, ComponentBuilder] = {}


def register_component(name: str, builder: ComponentBuilder) -> None:
    """Register a custom architectural component builder.

    Parameters
    ----------
    name:
        Identifier used in ``CustomModuleConfig.name``.
    builder:
        Callable that takes the component config and model dimension and
        returns an ``nn.Module`` instance.
    """
    _REGISTRY[name] = builder


def get_component(name: str) -> ComponentBuilder | None:
    """Return the registered builder for ``name``, if any."""
    return _REGISTRY.get(name)


def list_components() -> list[str]:
    """Return the list of registered component names."""
    return sorted(_REGISTRY)

