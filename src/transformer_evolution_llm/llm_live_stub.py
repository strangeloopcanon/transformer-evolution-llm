"""Placeholder for LLM live tests.

This module keeps the Interface Contract satisfied while the real
Phi-tiny/OLMoE eval harness is still under construction. It exits
with code 0 yet performs basic config validation to catch regressions.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console


def main() -> None:
    """Ensure required config artifacts exist and report status."""
    console = Console()
    frontier = Path("runs/frontier.json")
    configs = Path("configs")

    if not configs.exists():
        raise SystemExit("Missing configs/ directory required for live eval.")

    console.print("[bold green]LLM live stub[/]: configs directory found.")
    if frontier.exists():
        console.print("Existing frontier state located at runs/frontier.json")
    else:
        console.print("No frontier snapshot yet; skipping pairwise replay.")


if __name__ == "__main__":
    main()
