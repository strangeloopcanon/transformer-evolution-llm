from pathlib import Path

import torch

from transformer_evolution_llm.cache_builder import synthesize_cache
from transformer_evolution_llm.logits_cache import TopKLogitCache


def test_synthesize_cache_and_loader(tmp_path: Path):
    shard = synthesize_cache(
        tmp_path,
        samples=3,
        seq_len=4,
        topk=2,
        vocab=32,
    )
    assert shard.exists()
    cache = TopKLogitCache(tmp_path)
    ids = ["sample-0", "sample-1"]
    idx, logp = cache.fetch(ids, topk=2)
    assert isinstance(idx, torch.Tensor)
    assert idx.shape[-1] == 2
    assert logp.shape == idx.shape
