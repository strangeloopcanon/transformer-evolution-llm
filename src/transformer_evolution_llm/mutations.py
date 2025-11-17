"""Function-preserving mutation utilities."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable

from .dsl import (
    ArchitectureSpec,
    CustomModuleConfig,
    DenseFFNConfig,
    GatedModuleConfig,
    MoEFFNConfig,
    RecurrenceConfig,
    RetroConfig,
    SSMConfig,
)
from .template_mutation import apply_template_mutation

MutationFn = Callable[[ArchitectureSpec, random.Random], ArchitectureSpec]


def clone_spec(spec: ArchitectureSpec) -> ArchitectureSpec:
    return spec.model_copy(deep=True)


def dense_to_moe(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    idx = rng.randrange(len(child.model.blocks))
    block = child.model.blocks[idx]
    if isinstance(block.ffn, MoEFFNConfig):
        return child
    dense = block.ffn
    if not isinstance(dense, DenseFFNConfig):
        msg = "dense_to_moe expects a dense FFN block."
        raise TypeError(msg)
    block.ffn = MoEFFNConfig(
        hidden=dense.hidden,
        n_experts=32,
        k=2,
        balance=0.05,
        shared=1,
    )
    return child


def mutate_topk(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    moe_blocks = [b for b in child.model.blocks if isinstance(b.ffn, MoEFFNConfig)]
    if not moe_blocks:
        return child
    target = rng.choice(moe_blocks)
    if not isinstance(target.ffn, MoEFFNConfig):
        msg = "mutate_topk requires a MoE FFN."
        raise TypeError(msg)
    target.ffn.k = rng.choice([1, 2, 4])
    target.ffn.capacity_factor = rng.uniform(1.0, 1.5)
    return child


def shift_moe(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    blocks = child.model.blocks
    moe_idx = [i for i, b in enumerate(blocks) if isinstance(b.ffn, MoEFFNConfig)]
    if not moe_idx:
        return child
    src = rng.choice(moe_idx)
    dst = max(0, min(len(blocks) - 1, src + rng.choice([-2, -1, 1, 2])))
    if src == dst:
        return child
    block = copy.deepcopy(blocks[src])
    del blocks[src]
    blocks.insert(dst, block)
    return child


def make_gqa(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    idx = rng.randrange(len(child.model.blocks))
    block = child.model.blocks[idx]
    if not block.attn:
        return child
    attn = block.attn
    attn.kind = "GQA"
    attn.kv_groups = max(1, attn.heads // 4)
    return child


def toggle_precision(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    child.train.bf16 = not child.train.bf16
    return child


def insert_retro_module(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    retro = RetroConfig(
        memory_tokens=rng.choice([512, 1024, 2048]),
        stride=rng.choice([32, 64, 128]),
        aggregator=rng.choice(["mean", "attention", "gate"]),
        gating_weight=rng.uniform(0.1, 0.9),
    )
    block.extras.append(retro)
    return child


def insert_custom_module(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    module = CustomModuleConfig(
        name=f"exp-{rng.randrange(10_000)}",
        params={
            "dim": rng.choice([256, 512, 1024]),
            "activation": rng.choice(["silu", "gelu", "relu"]),
            "notes": "auto-generated",
        },
    )
    block.extras.append(module)
    return child


def toggle_gated_mix(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    gated = next((extra for extra in block.extras if isinstance(extra, GatedModuleConfig)), None)
    if gated:
        gated.init_weight = 1.0 - gated.init_weight
        gated.targets = list(reversed(gated.targets))
    else:
        block.extras.append(
            GatedModuleConfig(
                targets=rng.sample(["attn", "ffn", "ssm"], k=rng.choice([2, 3])),
                init_weight=rng.uniform(0.05, 0.5),
            )
        )
    return child


def toggle_ssm(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    idx = rng.randrange(len(child.model.blocks))
    block = child.model.blocks[idx]
    if block.ssm:
        block.ssm = None
    else:
        block.ssm = SSMConfig(kind="mamba2", d_state=16, d_conv=4, dt_rank=8, chunk=128, gate=0.1)
    return child


def tune_kv(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    # pick a block with attention
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    assert b.attn is not None
    heads = max(1, int(b.attn.heads))
    choices = [k for k in [1, 2, 4, 8] if k <= heads]
    if not choices:
        return child
    b.attn.kv_groups = int(rng.choice(choices))
    return child


def tune_rope(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    assert b.attn is not None
    # Ensure rope is enabled; jitter rope_theta
    if b.attn.rope is None:
        b.attn.rope = "yarn"
    base = float(b.attn.rope_theta or 10000.0)
    jitter = rng.uniform(0.5, 2.0)
    b.attn.rope_theta = max(1000.0, min(200000.0, base * jitter))
    return child


def add_recurrence(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    if len(child.model.blocks) < 3:
        return child
    start = rng.randrange(0, len(child.model.blocks) - 1)
    end = min(len(child.model.blocks), start + rng.choice([2, 3, 4]))
    rec = RecurrenceConfig(
        start=start,
        end=end,
        adapter=rng.choice(["linear", "gated"]),
        concat_prelude=rng.choice([True, False]),
        init_state=rng.choice(["zeros", "noise"]),
        noise_std=rng.uniform(0.01, 0.05),
        train_recurrence=rng.choice([1, 2]),
        max_train_recurrence=rng.choice([4, 6, 8]),
        curriculum_fraction=rng.uniform(0.1, 0.4),
        test_recurrences=[1, 2, 4, 8, 16],
    )
    child.model.recurrences.append(rec)
    return child


def tune_recurrence(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    if not child.model.recurrences:
        return add_recurrence(spec, rng)
    rec = rng.choice(child.model.recurrences)
    rec.max_train_recurrence = max(
        rec.train_recurrence,
        int(rec.max_train_recurrence + rng.choice([-1, 0, 2])),
    )
    rec.curriculum_fraction = max(
        0.0, min(1.0, rec.curriculum_fraction + rng.uniform(-0.1, 0.1))
    )
    rec.concat_prelude = rng.choice([True, rec.concat_prelude])
    rec.adapter = rng.choice(["linear", "gated"])
    rec.init_state = rng.choice(["zeros", "noise"])
    return child


REGISTRY: dict[str, MutationFn] = {
    "dense_to_moe": dense_to_moe,
    "mutate_topk": mutate_topk,
    "shift_moe": shift_moe,
    "make_gqa": make_gqa,
    "toggle_precision": toggle_precision,
    "insert_retro_module": insert_retro_module,
    "insert_custom_module": insert_custom_module,
    "toggle_gated_mix": toggle_gated_mix,
    "toggle_ssm": toggle_ssm,
    "tune_kv": tune_kv,
    "tune_rope": tune_rope,
    "add_recurrence": add_recurrence,
    "tune_recurrence": tune_recurrence,
    "template_mutation": lambda spec, rng: apply_template_mutation(spec, rng),
}


def mutate(
    spec: ArchitectureSpec, rng: random.Random | None = None
) -> tuple[str, ArchitectureSpec]:
    rng = rng or random.Random()  # noqa: S311  # nosec B311 - deterministic enough for search
    name = rng.choice(list(REGISTRY))
    mutated = REGISTRY[name](spec, rng)
    return name, mutated
