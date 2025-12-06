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


def tune_experts(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Jitter MoE expert count and selector top-k to explore capacity/compute tradeoffs."""
    child = clone_spec(spec)
    moe_blocks = [b for b in child.model.blocks if isinstance(b.ffn, MoEFFNConfig)]
    if moe_blocks:
        target = rng.choice(moe_blocks)
        if isinstance(target.ffn, MoEFFNConfig):
            # adjust expert count within a modest range
            candidates = [n for n in [8, 12, 16, 24, 32, 48] if n != target.ffn.n_experts]
            if candidates:
                target.ffn.n_experts = rng.choice(candidates)
            # adjust k bounded by n_experts
            possible_k = [k for k in [1, 2, 4, 8] if k <= target.ffn.n_experts]
            if possible_k:
                target.ffn.k = rng.choice(possible_k)
    # selector top-k tweak on a random attention block
    attn_blocks = [b for b in child.model.blocks if b.attn is not None]
    if attn_blocks:
        b = rng.choice(attn_blocks)
        if b.attn and getattr(b.attn, "selector", "none") != "none":
            b.attn.selector_topk = int(rng.choice([24, 32, 48, 64, 96, 128]))
    return child


def tune_router(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Jitter MoE router behaviour (sigmoid/softmax, bias detaching, shared expert)."""
    child = clone_spec(spec)
    moe_blocks = [b for b in child.model.blocks if isinstance(b.ffn, MoEFFNConfig)]
    if not moe_blocks:
        return child
    target = rng.choice(moe_blocks)
    if not isinstance(target.ffn, MoEFFNConfig):
        return child
    target.ffn.router_type = rng.choice(["softmax", "sigmoid"])
    target.ffn.router_bias_detached = rng.choice([True, False])
    target.ffn.shared_expert = rng.choice([True, False])
    target.ffn.k = rng.choice(
        [min(target.ffn.n_experts, k) for k in [2, 4, 8] if k <= target.ffn.n_experts]
        or [target.ffn.k]
    )
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
        memory_tokens=int(rng.choice([512, 1024, 2048])),
        stride=int(rng.choice([32, 64, 128])),
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
    if b.attn is None:
        return child
    heads = max(1, int(b.attn.heads))
    choices = [k for k in [1, 2, 4, 8] if k <= heads]
    if not choices:
        return child
    b.attn.kv_groups = int(rng.choice(choices))
    return child


def toggle_selector(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Flip selector-based sparsity on an attention block and retune its knobs."""
    child = clone_spec(spec)
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    if b.attn is None:
        return child
    if getattr(b.attn, "selector", "none") != "none":
        b.attn.selector = "none"
        b.attn.selector_topk = None
        b.attn.selector_heads = None
        b.attn.selector_dim = None
        b.attn.selector_rope = "none"
        b.attn.selector_detach = False
    else:
        b.attn.selector = "dsa"
        b.attn.selector_topk = int(rng.choice([32, 64, 96, 128, 192]))
        b.attn.selector_heads = int(rng.choice([1, 2, 4]))
        b.attn.selector_dim = b.attn.head_dim
        b.attn.selector_rope = rng.choice(["partial", "full", "none"])
        b.attn.selector_detach = rng.choice([True, False])
    return child


def tune_rope(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    if b.attn is None:
        return child
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
    # Allow wider spans so recurrence can stitch distant stages together.
    max_span = max(2, min(len(child.model.blocks), rng.choice([4, 6, 8, len(child.model.blocks)])))
    span = rng.randint(2, max_span)
    end = min(len(child.model.blocks), start + span)
    if end <= start:
        end = min(len(child.model.blocks), start + 1)
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
    rec.curriculum_fraction = max(0.0, min(1.0, rec.curriculum_fraction + rng.uniform(-0.1, 0.1)))
    rec.concat_prelude = rng.choice([True, rec.concat_prelude])
    rec.adapter = rng.choice(["linear", "gated"])
    rec.init_state = rng.choice(["zeros", "noise"])
    return child


def tune_attn_gating(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    if b.attn is None:
        return child

    if b.attn.gating_pos == "none":
        # Enable gating with a random position/op
        b.attn.gating_pos = rng.choice(["output", "value"])
        b.attn.gating_op = rng.choice(["dense", "diagonal"])
    else:
        # 33% chance turn off, 66% chance change params
        if rng.random() < 0.33:
            b.attn.gating_pos = "none"
        else:
            if rng.random() < 0.5:
                b.attn.gating_pos = rng.choice(["output", "value"])
            else:
                b.attn.gating_op = rng.choice(["dense", "diagonal"])
    return child


def tune_attn_shape(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Jitter attention heads/head_dim/kv_groups while keeping model dim stable."""
    child = clone_spec(spec)
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    attn = b.attn
    if attn is None:
        return child
    d_model = child.model.emb.dim
    # pick heads that divide d_model reasonably
    candidate_heads = [h for h in [4, 6, 8, 12, 16] if d_model % h == 0]
    if not candidate_heads:
        return child
    heads = rng.choice(candidate_heads)
    head_dim = d_model // heads
    attn.heads = heads
    attn.head_dim = head_dim
    # kv_groups in [1, heads] preferring divisors
    kv_candidates = [k for k in [1, 2, 4, 8, heads] if k <= heads]
    attn.kv_groups = int(rng.choice(kv_candidates))
    return child


def tune_attn_sparsity(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Explore sparsity/window settings (local/global strides)."""
    child = clone_spec(spec)
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    attn = b.attn
    if attn is None:
        return child
    sparsity_opts = ["none", "local_global"]
    attn.sparsity = rng.choice(sparsity_opts)
    if attn.sparsity == "local_global":
        attn.sw = rng.choice([64, 96, 128, 192, 256])
        attn.global_stride = rng.choice([32, 64, 96, 128])
    else:
        attn.sw = None
        attn.global_stride = None
    return child


def tune_ffn_width_activation(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Adjust FFN hidden size and activation."""
    child = clone_spec(spec)
    if not child.model.blocks:
        return child
    block = rng.choice(child.model.blocks)
    if block.ffn is None:
        return child
    # Only adjust dense FFNs
    if getattr(block.ffn, "type", "dense") != "dense":
        return child
    hidden = getattr(block.ffn, "hidden", None)
    if hidden:
        scale = rng.uniform(0.75, 1.5)
        new_hidden = max(256, min(int(hidden * scale), 8192))
        block.ffn.hidden = new_hidden  # type: ignore[attr-defined]
    block.ffn.activation = rng.choice(["swiglu", "gelu", "silu", "relu"])  # type: ignore[attr-defined]
    return child


def tune_router_coeffs(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Jitter MoE router temperatures and load-balance coefficients."""
    child = clone_spec(spec)
    moe_blocks = [b for b in child.model.blocks if isinstance(b.ffn, MoEFFNConfig)]
    if not moe_blocks:
        return child
    block = rng.choice(moe_blocks)
    if not isinstance(block.ffn, MoEFFNConfig):
        return child
    block.ffn.router_temperature = rng.choice([None, rng.uniform(0.3, 2.0)])  # type: ignore[attr-defined]
    block.ffn.router_lb_weight = rng.choice([None, rng.uniform(0.0, 0.1)])  # type: ignore[attr-defined]
    block.ffn.router_aux_weight = rng.choice([None, rng.uniform(0.0, 0.1)])  # type: ignore[attr-defined]
    return child


def tune_retro(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Adjust retro memory slots/stride and gating."""
    child = clone_spec(spec)
    if not child.model.blocks:
        return child
    block = rng.choice(child.model.blocks)
    # Ensure a retro extra exists
    retro = None
    for extra in block.extras:
        if isinstance(extra, RetroConfig):
            retro = extra
            break
    if retro is None:
        retro = RetroConfig(
            memory_tokens=int(rng.choice([256, 512, 1024])),
            stride=int(rng.choice([32, 64, 128])),
            aggregator=rng.choice(["mean", "attention", "gate"]),
            gating_weight=rng.uniform(0.1, 0.5),
        )
        block.extras.append(retro)
    else:
        retro.memory_tokens = int(rng.choice([256, 512, 768, 1024]))  # type: ignore[attr-defined]
        retro.stride = int(rng.choice([16, 32, 64, 128]))  # type: ignore[attr-defined]
        retro.aggregator = rng.choice(["mean", "attention", "gate"])  # type: ignore[attr-defined]
        retro.gating_weight = rng.uniform(0.1, 0.5)  # type: ignore[attr-defined]
    return child


def toggle_qk_norm(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Toggle QK norm clamp."""
    child = clone_spec(spec)
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    attn = b.attn
    if attn is None:
        return child
    current = getattr(attn, "qk_norm_max", None)
    if current is None:
        attn.qk_norm_max = rng.choice([0.5, 1.0, 2.0, 4.0])
    else:
        attn.qk_norm_max = None
    return child


def duplicate_block_span(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Append a copied span of blocks to increase depth without disturbing existing indices."""
    child = clone_spec(spec)
    blocks = child.model.blocks
    if not blocks:
        return child
    start = rng.randrange(len(blocks))
    span_len = rng.choice([1, 2, 3])
    end = min(len(blocks), start + span_len)
    duplicated = [copy.deepcopy(b) for b in blocks[start:end]]
    blocks.extend(duplicated)
    return child


def shuffle_block_span(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Shuffle a local span of blocks to create new phase orderings."""
    child = clone_spec(spec)
    blocks = child.model.blocks
    if len(blocks) < 3:
        return child
    span_len = rng.choice([2, 3, 4])
    if span_len > len(blocks):
        return child
    start = rng.randrange(0, len(blocks) - span_len + 1)
    span = blocks[start : start + span_len]
    rng.shuffle(span)
    blocks[start : start + span_len] = span
    return child


def add_additional_recurrence(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Add a second recurrence window to create multi-stage looping."""
    child = clone_spec(spec)
    if len(child.model.blocks) < 2:
        return child
    start = rng.randrange(0, len(child.model.blocks) - 1)
    end = rng.randrange(start + 1, len(child.model.blocks) + 1)
    rec = RecurrenceConfig(
        start=start,
        end=end,
        adapter=rng.choice(["linear", "gated"]),
        adapter_dim=None,
        concat_prelude=rng.choice([True, False]),
        init_state=rng.choice(["zeros", "noise"]),
        noise_std=rng.uniform(0.01, 0.05),
        train_recurrence=rng.choice([1, 2, 3]),
        max_train_recurrence=rng.choice([2, 4, 6]),
        curriculum_fraction=rng.uniform(0.1, 0.4),
        test_recurrences=[1, 2, 4, 8],
    )
    child.model.recurrences.append(rec)
    return child


def add_extra_combo(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Attach multiple extras (retro + gated mix) to encourage hybrid blocks."""
    child = clone_spec(spec)
    if not child.model.blocks:
        return child
    block = rng.choice(child.model.blocks)
    existing_types = {type(extra) for extra in block.extras}
    if RetroConfig not in existing_types:
        block.extras.append(
            RetroConfig(
                memory_tokens=int(rng.choice([256, 512, 1024])),
                stride=int(rng.choice([32, 64, 128])),
                aggregator=rng.choice(["mean", "attention", "gate"]),
                gating_weight=rng.uniform(0.1, 0.5),
            )
        )
    if GatedModuleConfig not in existing_types:
        block.extras.append(GatedModuleConfig(init_weight=rng.uniform(0.05, 0.3)))
    return child


def graph_jitter(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Apply a handful of neutral edits to increase structural entropy."""
    child = clone_spec(spec)
    jitter_ops = [
        "duplicate_block_span",
        "shuffle_block_span",
        "add_recurrence",
        "add_additional_recurrence",
        "toggle_ssm",
        "insert_retro_module",
        "insert_custom_module",
        "toggle_gated_mix",
        "tune_attn_gating",
        "tune_kv",
        "tune_rope",
        "dense_to_moe",
        "mutate_topk",
        "shift_moe",
        "make_gqa",
    ]
    steps = rng.randint(2, 4)
    current = child
    for name in rng.sample(jitter_ops, k=min(steps, len(jitter_ops))):
        fn = REGISTRY.get(name)
        if fn:
            current = fn(current, rng)
    return current


REGISTRY: dict[str, MutationFn] = {
    "duplicate_block_span": duplicate_block_span,
    "shuffle_block_span": shuffle_block_span,
    "add_additional_recurrence": add_additional_recurrence,
    "add_extra_combo": add_extra_combo,
    "tune_attn_gating": tune_attn_gating,
    "dense_to_moe": dense_to_moe,
    "mutate_topk": mutate_topk,
    "shift_moe": shift_moe,
    "tune_router": tune_router,
    "make_gqa": make_gqa,
    "toggle_precision": toggle_precision,
    "insert_retro_module": insert_retro_module,
    "insert_custom_module": insert_custom_module,
    "toggle_gated_mix": toggle_gated_mix,
    "toggle_ssm": toggle_ssm,
    "tune_kv": tune_kv,
    "toggle_selector": toggle_selector,
    "tune_rope": tune_rope,
    "tune_attn_shape": tune_attn_shape,
    "tune_attn_sparsity": tune_attn_sparsity,
    "tune_ffn_width_activation": tune_ffn_width_activation,
    "tune_router_coeffs": tune_router_coeffs,
    "tune_retro": tune_retro,
    "toggle_qk_norm": toggle_qk_norm,
    "add_recurrence": add_recurrence,
    "tune_recurrence": tune_recurrence,
    "graph_jitter": graph_jitter,
    "tune_experts": tune_experts,
    "template_mutation": lambda spec, rng: apply_template_mutation(spec, rng),
}


def mutate(
    spec: ArchitectureSpec,
    rng: random.Random | None = None,
    weights: dict[str, float] | None = None,
    steps: int = 1,
) -> tuple[str, ArchitectureSpec]:
    """Apply one or more registered mutations. If weights provided, sample by weight."""
    rng = rng or random.Random()  # noqa: S311  # nosec B311 - deterministic enough for search
    names = list(REGISTRY)

    def _pick() -> str:
        if weights:
            w = [max(0.0, float(weights.get(n, 0.0))) for n in names]
            if any(w):
                total = sum(w) or 1.0
                probs = [x / total for x in w]
                return rng.choices(names, weights=probs, k=1)[0]
        return rng.choice(names)

    applied: list[str] = []
    current = spec
    for _ in range(max(1, steps)):
        name = _pick()
        current = REGISTRY[name](current, rng)
        applied.append(name)
    label = "+".join(applied) if len(applied) > 1 else applied[0]
    return label, current
