"""Helpers for generating teacher logit caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()


def synthesize_cache(
    out_dir: Path,
    *,
    samples: int = 128,
    seq_len: int = 2048,
    topk: int = 8,
    vocab: int = 100_352,
) -> Path:
    """Create a synthetic logit cache for pipeline testing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.random.randint(0, vocab, size=(samples, seq_len, topk), dtype=np.int32)
    logp = np.log(np.random.dirichlet([1.0] * topk, size=(samples * seq_len)))
    logp = logp.reshape(samples, seq_len, topk).astype(np.float32)
    uids = np.array([f"sample-{i}" for i in range(samples)], dtype=object)
    shard_path = out_dir / "synthetic_cache.npz"
    np.savez(shard_path, topk_idx=idx, topk_logp=logp, uids=uids)
    console.print(f"Synthetic cache written to {shard_path}")
    return shard_path
