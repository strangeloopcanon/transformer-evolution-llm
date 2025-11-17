from pathlib import Path

from transformer_evolution_llm.cache_builder import synthesize_cache
from transformer_evolution_llm.logits_cache import TopKLogitCache


def test_topk_logit_cache(tmp_path: Path):
    synthesize_cache(tmp_path, samples=2, seq_len=4, topk=2, vocab=16)
    cache = TopKLogitCache(tmp_path)
    idx, logp = cache.fetch(["sample-0"], topk=2)
    assert idx.shape[-1] == 2
    assert logp.shape == idx.shape
