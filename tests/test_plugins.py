from __future__ import annotations

import torch
from torch import nn

from transformer_evolution_llm.dsl import ArchitectureSpec
from transformer_evolution_llm.models import EvolutionModel
from transformer_evolution_llm.plugins import register_component


class DummyPlugin(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(x)


def test_plugin_registry_overrides_custom_module() -> None:
    register_component(
        "dummy_plugin",
        lambda cfg, dim: DummyPlugin(dim),
    )
    spec = ArchitectureSpec(
        model={
            "name": "plugin-test",
            "emb": {"dim": 16, "vocab": 32},
            "blocks": [
                {
                    "attn": {"kind": "GQA", "heads": 1, "head_dim": 16},
                    "ffn": {"type": "dense", "hidden": 64},
                    "extras": [
                        {"type": "custom", "name": "dummy_plugin", "params": {}},
                    ],
                }
            ],
            "head": {"tie_embeddings": True, "vocab": 32},
        },
        train={
            "lr": 1e-3,
            "warmup": 10,
            "clip": 1.0,
            "bf16": True,
            "grad_ckpt": False,
            "max_tokens": 128,
        },
        data={
            "tokenizer": "hf-internal-testing/tiny-random-gpt2",
            "seq_len": 8,
            "batch_size": 1,
            "workers": 0,
            "shards": [{"name": "lighteval/mgsm", "split": "train", "weight": 1.0}],
        },
    )
    model = EvolutionModel(spec.model)
    block = model.blocks[0]
    plugins = [extra for extra in block.extras if isinstance(extra, DummyPlugin)]
    assert plugins, "expected DummyPlugin to be used for custom module extra"
