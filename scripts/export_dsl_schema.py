"""Export the JSON Schema for the typed evolution DSL.

This mirrors the upstream project's pattern of publishing a schema for editors,
CI validation, and downstream tooling.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from transformer_evolution_llm.dsl import ArchitectureSpec

app = typer.Typer(help="Export JSON Schema for `ArchitectureSpec`.")


@app.command()
def main(
    out: Path = typer.Argument(..., help="Output path (usually .json)."),
    pretty: bool = typer.Option(True, help="Write pretty-printed JSON."),
) -> None:
    schema = ArchitectureSpec.model_json_schema()
    text = json.dumps(schema, indent=2 if pretty else None, sort_keys=False)
    out.write_text(text)
    typer.echo(f"Wrote DSL schema to {out}")


if __name__ == "__main__":
    app()

