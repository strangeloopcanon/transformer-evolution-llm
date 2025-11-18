from pathlib import Path

import torch

from transformer_evolution_llm.candidates import Candidate
from transformer_evolution_llm.data import TokenBatch
from transformer_evolution_llm.trainer import FullWeightTrainer


def synthetic_batches(vocab: int, seq_len: int, steps: int):
    for _ in range(steps):
        ids = torch.randint(0, vocab, (2, seq_len))
        yield TokenBatch(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            uids=["synthetic"],
        )


def test_full_weight_trainer_runs(tmp_path: Path, tiny_spec) -> None:
    trainer = FullWeightTrainer(checkpoint_dir=tmp_path, steps=2, eval_batches=1, device="cpu")
    candidate = Candidate(ident="cand-1", spec=tiny_spec)
    metrics, ckpt = trainer.train(
        candidate,
        tiny_spec,
        synthetic_batches(tiny_spec.model.head.vocab, tiny_spec.data.seq_len, steps=4),
    )
    assert "ppl_code" in metrics
    # Router metrics should always be present for tooling, even if no MoE blocks exist.
    assert "router_entropy" in metrics
    assert "router_lb" in metrics
    assert "router_load_max" in metrics
    assert "router_load_min" in metrics
    assert "stop_reason_code" in metrics
    assert "nan_seen" in metrics
    assert "loss_spike" in metrics
    assert ckpt.exists()
