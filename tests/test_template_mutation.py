import random

from transformer_evolution_llm import template_mutation as tm
from transformer_evolution_llm.dsl import ArchitectureSpec
from transformer_evolution_llm.template_mutation import (
    MutationTemplate,
    _generate_random_template,
    apply_template_mutation,
)


def test_template_mutation_changes_blocks(monkeypatch):
    spec = ArchitectureSpec(
        model={
            "name": "template-test",
            "emb": {"dim": 128, "vocab": 100},
            "blocks": [
                {
                    "attn": {"kind": "GQA", "heads": 4, "head_dim": 32},
                    "ffn": {"type": "dense", "hidden": 512},
                }
            ],
            "head": {"vocab": 100, "tie_embeddings": True},
        },
        train={"lr": 1e-3, "warmup": 1, "clip": 1.0},
        data={
            "tokenizer": "gpt2",
            "seq_len": 64,
            "batch_size": 1,
            "workers": 0,
            "shards": [{"name": "ag_news", "split": "train", "weight": 1.0}],
        },
    )

    def fake_templates():
        return [
            MutationTemplate(
                name="test-add-extra",
                weight=1.0,
                conditions={},
                actions=[
                    {
                        "add_extra": {
                            "selector": "random",
                            "extra_type": "gated",
                            "params": {"targets": ["attn"], "init_weight": 0.3},
                        }
                    }
                ],
            )
        ]

    monkeypatch.setattr(tm, "load_templates", fake_templates)

    rng = random.Random(0)  # noqa: S311 - deterministic test RNG
    mutated = apply_template_mutation(spec, rng)
    assert mutated.model.blocks[0].extras, "expected template mutation to add an extra module"


def test_generate_random_template_produces_actions():
    spec = ArchitectureSpec(
        model={
            "name": "auto-template",
            "emb": {"dim": 64, "vocab": 50},
            "blocks": [
                {
                    "attn": {"kind": "GQA", "heads": 2, "head_dim": 32},
                    "ffn": {"type": "dense", "hidden": 256},
                }
            ],
            "head": {"vocab": 50, "tie_embeddings": True},
        },
        train={"lr": 1e-3, "warmup": 1, "clip": 1.0},
        data={
            "tokenizer": "gpt2",
            "seq_len": 32,
            "batch_size": 1,
            "workers": 0,
            "shards": [{"name": "ag_news", "split": "train", "weight": 1.0}],
        },
    )
    rng = random.Random(42)  # noqa: S311 - deterministic test RNG
    template = _generate_random_template(spec, rng)
    assert template.actions, "auto-generated template should have at least one action"
