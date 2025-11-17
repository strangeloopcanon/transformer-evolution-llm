"""Template-driven mutation engine for architectural edits."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import yaml

from .dsl import (
    ArchitectureSpec,
    AttentionConfig,
    BlockConfig,
    CustomModuleConfig,
    DenseFFNConfig,
    GatedModuleConfig,
    MoEFFNConfig,
    RetroConfig,
    SSMConfig,
)

TEMPLATE_PATH = Path("configs/mutation_templates.yaml")


class TemplateAction(TypedDict, total=False):
    selector: str
    extra_type: str
    params: dict[str, Any]
    block_template: str
    position: str
    new_type: str
    n_experts: int
    k: int
    balance: float
    capacity_factor: float
    # Tuning knobs
    qk_norm_max: float | None
    sw_jitter: int
    temperature: float
    lb_coeff: float
    entropy_coeff: float
    sparsity: str
    block_size: int | None
    block_stride: int | None
    global_stride: int | None
    dilation: int | None


ActionMap = dict[str, TemplateAction]


@dataclass
class MutationTemplate:
    name: str
    weight: float
    conditions: dict[str, Any]
    actions: list[ActionMap]


def load_templates() -> list[MutationTemplate]:
    if not TEMPLATE_PATH.exists():
        return _seed_templates()
    data = yaml.safe_load(TEMPLATE_PATH.read_text())
    templates = []
    for entry in data.get("templates", []):
        templates.append(
            MutationTemplate(
                name=entry["name"],
                weight=float(entry.get("weight", 1.0)),
                conditions=entry.get("conditions", {}),
                actions=cast(list[ActionMap], entry.get("actions", [])),
            )
        )
    return templates


def apply_template_mutation(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    base_templates = load_templates()
    dynamic_templates: list[MutationTemplate] = []
    for tpl in base_templates:
        if rng.random() < 0.5:
            dynamic_templates.append(_mutate_template(tpl, rng))
    dynamic_templates.append(_generate_random_template(spec, rng))
    templates = base_templates + dynamic_templates
    eligible = [tpl for tpl in templates if _matches_conditions(spec, tpl.conditions)]
    if not eligible:
        return spec
    template = rng.choices(eligible, weights=[tpl.weight for tpl in eligible], k=1)[0]
    new_spec = spec.model_copy(deep=True)
    for action in template.actions:
        _apply_action(new_spec, action, rng)
    return new_spec


def _matches_conditions(spec: ArchitectureSpec, conditions: dict[str, Any]) -> bool:
    blocks = spec.model.blocks
    if not conditions:
        return True
    if conditions.get("requires_ssm_block"):
        if not any(block.ssm for block in blocks):
            return False
    if conditions.get("requires_dense_ffn"):
        if not any(isinstance(block.ffn, DenseFFNConfig) for block in blocks):
            return False
    if conditions.get("min_blocks"):
        if len(blocks) < int(conditions["min_blocks"]):
            return False
    return True


def _apply_action(spec: ArchitectureSpec, action: ActionMap, rng: random.Random) -> None:
    if "add_extra" in action:
        _add_extra(spec, action["add_extra"], rng)
    elif "insert_block" in action:
        _insert_block(spec, action["insert_block"], rng)
    elif "replace_ffn" in action:
        _replace_ffn(spec, action["replace_ffn"], rng)
    elif "remove_block" in action:
        _remove_block(spec, action["remove_block"], rng)
    elif "tune_attn" in action:
        _tune_attn(spec, dict(action["tune_attn"]), rng)
    elif "tune_router" in action:
        _tune_router(spec, dict(action["tune_router"]), rng)


def _select_block_index(spec: ArchitectureSpec, selector: str, rng: random.Random) -> int | None:
    blocks = spec.model.blocks
    candidates: list[int]
    if selector == "random":
        candidates = list(range(len(blocks)))
    elif selector == "random_moe":
        candidates = [
            idx for idx, block in enumerate(blocks) if isinstance(block.ffn, MoEFFNConfig)
        ]
    elif selector == "random_dense":
        candidates = [
            idx for idx, block in enumerate(blocks) if isinstance(block.ffn, DenseFFNConfig)
        ]
    elif selector == "random_ssm":
        candidates = [idx for idx, block in enumerate(blocks) if block.ssm is not None]
    else:
        candidates = list(range(len(blocks)))
    if not candidates:
        return None
    return rng.choice(candidates)


def _add_extra(spec: ArchitectureSpec, params: TemplateAction, rng: random.Random) -> None:
    idx = _select_block_index(spec, params.get("selector", "random"), rng)
    if idx is None:
        return
    block = spec.model.blocks[idx]
    extra_type = params.get("extra_type", "custom")
    extra_params = params.get("params", {})
    if extra_type == "retro":
        block.extras.append(
            RetroConfig(
                memory_tokens=extra_params.get("memory_tokens", spec.data.seq_len // 4),
                stride=extra_params.get("stride", max(16, spec.data.seq_len // 16)),
                aggregator=extra_params.get("aggregator", "gate"),
                gating_weight=extra_params.get("gating_weight", 0.25),
            )
        )
    elif extra_type == "gated":
        targets = extra_params.get("targets") or ["attn", "ffn"]
        block.extras.append(
            GatedModuleConfig(
                targets=targets,
                init_weight=float(extra_params.get("init_weight", 0.2)),
                learnable=extra_params.get("learnable", True),
            )
        )
    elif extra_type == "feedback":
        source = rng.randrange(0, idx + 1) if idx > 0 else 0
        block.extras.append(
            CustomModuleConfig(
                name="feedback_gate",
                params={
                    "type": "feedback",
                    "source_block": source,
                    "strength": extra_params.get("strength", 0.1),
                },
            )
        )
    else:
        block.extras.append(
            CustomModuleConfig(
                name=extra_params.get("name", "exp"),
                params=extra_params.get("params", {"dim": spec.model.emb.dim}),
            )
        )


def _insert_block(spec: ArchitectureSpec, params: TemplateAction, rng: random.Random) -> None:
    template_name = params.get("block_template", "dense_attn")
    position = params.get("position", "end")
    blocks = spec.model.blocks
    new_block = _build_block(template_name, spec, rng)
    if new_block is None:
        return
    if position == "start":
        blocks.insert(0, new_block)
    elif position == "end":
        blocks.append(new_block)
    elif position == "random":
        blocks.insert(rng.randrange(0, len(blocks) + 1), new_block)
    else:
        blocks.append(new_block)


def _build_block(name: str, spec: ArchitectureSpec, rng: random.Random) -> BlockConfig | None:
    dim = spec.model.emb.dim
    attn = _default_attention(spec)
    ffn: DenseFFNConfig | MoEFFNConfig = DenseFFNConfig(hidden=dim * 4, activation="swiglu")
    extras: list[Any] = []
    ssm = None
    if name == "retro_moe":
        ffn = MoEFFNConfig(
            hidden=dim * 4,
            n_experts=16,
            k=2,
            balance=0.05,
            capacity_factor=1.2,
            shared=1,
        )
        extras.append(
            RetroConfig(
                memory_tokens=min(512, spec.data.seq_len),
                stride=max(32, spec.data.seq_len // 8),
                aggregator="gate",
                gating_weight=0.3,
            )
        )
    elif name == "ssm_dense":
        ssm = SSMConfig(kind="mamba2", d_state=16, d_conv=4, dt_rank=8, chunk=128, gate=0.1)
    elif name == "feedback_dense":
        extras.append(
            CustomModuleConfig(
                name="feedback_gate",
                params={
                    "type": "feedback",
                    "source_block": max(0, len(spec.model.blocks) - 1),
                    "strength": rng.uniform(0.1, 0.3),
                },
            )
        )
        extras.append(
            GatedModuleConfig(
                targets=["attn", "ffn"],
                init_weight=rng.uniform(0.15, 0.35),
                learnable=True,
            )
        )
        extras.append(
            RetroConfig(
                memory_tokens=min(512, spec.data.seq_len),
                stride=max(16, spec.data.seq_len // 16),
                aggregator="gate",
                gating_weight=rng.uniform(0.2, 0.4),
            )
        )
    return BlockConfig(attn=attn, ffn=ffn, ssm=ssm, extras=extras)


def _replace_ffn(spec: ArchitectureSpec, params: TemplateAction, rng: random.Random) -> None:
    idx = _select_block_index(spec, params.get("selector", "random"), rng)
    if idx is None:
        return
    block = spec.model.blocks[idx]
    new_type = params.get("new_type", "dense")
    if new_type == "moe":
        block.ffn = MoEFFNConfig(
            hidden=spec.model.emb.dim * 4,
            n_experts=params.get("n_experts", 8),
            k=params.get("k", 2),
            balance=params.get("balance", 0.05),
            capacity_factor=params.get("capacity_factor", 1.2),
            shared=1,
        )
    else:
        block.ffn = DenseFFNConfig(hidden=spec.model.emb.dim * 4, activation="swiglu")


def _remove_block(spec: ArchitectureSpec, params: TemplateAction, rng: random.Random) -> None:
    if len(spec.model.blocks) <= 1:
        return
    idx = _select_block_index(spec, params.get("selector", "random"), rng)
    if idx is None:
        return
    spec.model.blocks.pop(idx)


def _default_attention(spec: ArchitectureSpec) -> AttentionConfig | None:
    for block in spec.model.blocks:
        if block.attn:
            return block.attn.model_copy(deep=True)
    return None


def _seed_templates() -> list[MutationTemplate]:
    return [
        MutationTemplate(
            name="seed-feedback",
            weight=1.0,
            conditions={"requires_ssm_block": True},
            actions=[
                {
                    "add_extra": {
                        "selector": "random_ssm",
                        "extra_type": "feedback",
                        "params": {"strength": 0.1},
                    }
                }
            ],
        )
    ]


def _mutate_template(template: MutationTemplate, rng: random.Random) -> MutationTemplate:
    mutated = copy.deepcopy(template)
    mutated.name = f"{template.name}-mut-{rng.randrange(10_000)}"
    mutated.weight = max(0.1, template.weight * rng.uniform(0.8, 1.2))
    for action in mutated.actions:
        key = next(iter(action))
        params = action[key]
        if isinstance(params, dict) and "params" in params and isinstance(params["params"], dict):
            for param_key, value in params["params"].items():
                if isinstance(value, (int, float)):
                    params["params"][param_key] = value * rng.uniform(0.8, 1.2)
    return mutated


def _generate_random_template(spec: ArchitectureSpec, rng: random.Random) -> MutationTemplate:
    action_type = rng.choice(["add_extra", "insert_block", "replace_ffn"])
    # Occasionally tune stability/router knobs
    if rng.random() < 0.2:
        action_type = rng.choice(["tune_attn", "tune_router"])
    action: ActionMap
    if action_type == "add_extra":
        action = {
            "add_extra": {
                "selector": rng.choice(["random", "random_moe", "random_ssm"]),
                "extra_type": rng.choice(["gated", "retro", "feedback"]),
                "params": {
                    "gating_weight": rng.uniform(0.1, 0.5),
                    "memory_tokens": rng.randint(64, spec.data.seq_len),
                    "strength": rng.uniform(0.05, 0.3),
                },
            }
        }
    elif action_type == "insert_block":
        action = {
            "insert_block": {
                "position": rng.choice(["start", "end", "random"]),
                "block_template": rng.choice(["retro_moe", "ssm_dense"]),
            }
        }
    else:
        if action_type == "tune_attn":
            action = {
                "tune_attn": {
                    "selector": rng.choice(["random", "random_ssm", "random_dense", "random_moe"]),
                    "qk_norm_max": rng.choice([None, rng.uniform(5.0, 12.0)]),
                    "sw_jitter": rng.choice([-64, -32, 0, 32, 64]),
                    "sparsity": rng.choice(
                        ["none", "sliding", "block", "local_global", "dilated", "local_block"]
                    ),
                    "block_size": rng.choice([None, 32, 64, 128]),
                    "block_stride": rng.choice([None, 32, 64, 128]),
                    "global_stride": rng.choice([None, 16, 32, 64, 128]),
                    "dilation": rng.choice([None, 2, 4, 8]),
                }
            }
        elif action_type == "tune_router":
            action = {
                "tune_router": {
                    "temperature": rng.uniform(0.7, 1.5),
                    "lb_coeff": rng.uniform(0.002, 0.03),
                    "entropy_coeff": rng.uniform(0.0, 0.01),
                }
            }
        else:
            action = {
                "replace_ffn": {
                    "selector": "random",
                    "new_type": rng.choice(["dense", "moe"]),
                    "n_experts": rng.choice([8, 16, 32]),
                    "k": rng.choice([1, 2, 4]),
                    "balance": rng.uniform(0.02, 0.08),
                }
            }
    return MutationTemplate(
        name=f"auto-{rng.randrange(10_000)}",
        weight=1.0,
        conditions={},
        actions=[action],
    )


def _tune_attn(spec: ArchitectureSpec, params: dict[str, Any], rng: random.Random) -> None:
    idx = _select_block_index(spec, params.get("selector", "random"), rng)
    if idx is None:
        return
    block = spec.model.blocks[idx]
    if not block.attn:
        return
    qk_val = params.get("qk_norm_max")
    if qk_val is not None:
        block.attn.qk_norm_max = float(qk_val)
    sw_jitter = int(params.get("sw_jitter", 0))
    if sw_jitter != 0:
        current = block.attn.sw or spec.data.seq_len // 8
        block.attn.sw = max(8, min(spec.data.seq_len, int(current + sw_jitter)))
    if "sparsity" in params and block.attn is not None:
        block.attn.sparsity = params["sparsity"]
    if "block_size" in params and params["block_size"] is not None and block.attn is not None:
        block.attn.block_size = int(params["block_size"])
    if "block_stride" in params and params["block_stride"] is not None and block.attn is not None:
        block.attn.block_stride = int(params["block_stride"])
    if "global_stride" in params and params["global_stride"] is not None and block.attn is not None:
        block.attn.global_stride = int(params["global_stride"])
    if "dilation" in params and params["dilation"] is not None and block.attn is not None:
        block.attn.dilation = int(params["dilation"])


def _tune_router(spec: ArchitectureSpec, params: dict[str, Any], rng: random.Random) -> None:
    # Global training coefficients
    if "lb_coeff" in params:
        spec.train.router_lb_coeff = float(params["lb_coeff"])
    if "entropy_coeff" in params:
        spec.train.router_entropy_coeff = float(params["entropy_coeff"])
    # Per-block MoE temperature
    candidates = [b for b in spec.model.blocks if b.ffn and getattr(b.ffn, "type", "") == "moe"]
    if not candidates:
        return
    block = rng.choice(candidates)
    temp = float(params.get("temperature", 1.0))
    if isinstance(block.ffn, MoEFFNConfig):
        block.ffn.router_temperature = temp
