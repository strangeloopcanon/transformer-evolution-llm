import pytest

from transformer_evolution_llm.dsl import ArchitectureSpec


@pytest.fixture()
def tiny_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        model={
            "name": "phi-tiny-probe",
            "emb": {"dim": 512, "vocab": 4096},
            "blocks": [
                {
                    "attn": {"kind": "GQA", "heads": 4, "head_dim": 32},
                    "ffn": {"type": "dense", "hidden": 2048},
                }
            ],
            "head": {"tie_embeddings": True, "vocab": 4096},
        },
        train={
            "lr": 1e-3,
            "warmup": 10,
            "clip": 1.0,
            "bf16": True,
            "grad_ckpt": False,
            "max_tokens": 1024,
        },
        data={
            "tokenizer": "hf-internal-testing/tiny-random-gpt2",
            "seq_len": 64,
            "batch_size": 1,
            "workers": 0,
            "shards": [{"name": "lighteval/mgsm", "split": "train", "weight": 1.0}],
        },
    )
