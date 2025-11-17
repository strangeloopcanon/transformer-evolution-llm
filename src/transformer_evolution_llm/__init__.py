"""
transformer_evolution_llm
==========================

Utilities for describing, mutating, and evaluating MoE-centric transformer
architectures on resource-constrained hardware.
"""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Return the installed package version or '0.0.0' when unavailable."""
    try:
        return version("transformer_evolution_llm")
    except PackageNotFoundError:
        return "0.0.0"


__all__ = ["get_version"]
