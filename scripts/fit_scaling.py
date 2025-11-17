"""Fit simple scaling-law trends from one or more frontier JSON files.

Usage:
  python scripts/fit_scaling.py runs/frontier_phi_creative_unbiased.json \
    runs/frontier_phi_creative_overnight_lexi.json

Outputs log-log linear fits for:
  - ppl_code vs params
  - throughput vs (hidden_dim * heads)
  - ppl_code vs max_tokens (if available)
The script prints slopes/intercepts and predicted values for target sizes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import typer


app = typer.Typer(help="Fit lightweight scaling trends from frontier JSON artifacts.")


def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Return slope/intercept for y = a + b x."""
    if len(xs) != len(ys):
        msg = "xs and ys must be same length"
        raise ValueError(msg)
    n = len(xs)
    if n == 0:
        raise ValueError("no data points to fit")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0, mean_y
    slope = num / den
    intercept = mean_y - slope * mean_x
    return intercept, slope


def _log_fit(pairs: Iterable[tuple[float, float]]) -> tuple[float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for x, y in pairs:
        if x <= 0 or y <= 0:
            continue
        xs.append(math.log(x))
        ys.append(math.log(y))
    if not xs:
        raise ValueError("no positive pairs for log fit")
    a, b = _linear_fit(xs, ys)
    # log-space line: log y = a + b log x => y = exp(a) * x^b
    return a, b


def _load_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    raise ValueError(f"{path} is not a frontier JSON list")


@app.command()
def main(
    frontiers: list[Path] = typer.Argument(..., exists=True, readable=True),
    target_params: float = typer.Option(1.5e9, help="Target params for extrapolation."),
    target_tokens: float = typer.Option(5.0e9, help="Target tokens for extrapolation."),
) -> None:
    entries: list[dict] = []
    for frontier in frontiers:
        entries.extend(_load_entries(frontier))
    typer.echo(f"Loaded {len(entries)} entries from {len(frontiers)} frontier files.")

    params_ppl: list[tuple[float, float]] = []
    params_tokens: list[tuple[float, float]] = []
    throughput_pairs: list[tuple[float, float]] = []

    for entry in entries:
        metrics = entry.get("metrics", {})
        spec = entry.get("spec", {})
        params = float(metrics.get("params", 0.0))
        ppl = float(metrics.get("ppl_code", 0.0))
        throughput = float(metrics.get("throughput", 0.0))
        max_tokens = float(spec.get("train", {}).get("max_tokens") or 0.0)
        model = spec.get("model", {})
        emb_dim = float(model.get("emb", {}).get("dim") or 0.0)
        heads_total = 0
        head_count = 0
        for block in model.get("blocks", []):
            attn = block.get("attn")
            if attn and attn.get("heads"):
                heads_total += int(attn["heads"])
                head_count += 1
        if head_count:
            avg_heads = heads_total / head_count
        else:
            avg_heads = 0.0

        if params > 0 and ppl > 0:
            params_ppl.append((params, ppl))
        if params > 0 and max_tokens > 0:
            params_tokens.append((params, max_tokens))
        if throughput > 0 and emb_dim > 0 and avg_heads > 0:
            throughput_pairs.append((emb_dim * avg_heads, throughput))

    def report_fit(name: str, pairs: list[tuple[float, float]], target: float) -> None:
        if not pairs:
            typer.echo(f"[{name}] insufficient data.")
            return
        a, b = _log_fit(pairs)
        predicted = math.exp(a) * (target ** b)
        typer.echo(f"[{name}] log-fit: log y = {a:.3f} + {b:.3f} log x")
        typer.echo(f"          y ≈ exp({a:.3f}) * x^{b:.3f}")
        typer.echo(f"          Predicted y at x={target:.3e}: {predicted:.3f}")

    report_fit("ppl_code vs params", params_ppl, target_params)
    report_fit("tokens vs params", params_tokens, target_params)
    report_fit("throughput vs hidden*heads", throughput_pairs, target_tokens)


if __name__ == "__main__":
    app()

