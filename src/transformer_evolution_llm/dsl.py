"""Typed configuration DSL for evolutionary architecture search."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import ujson as json
import yaml
from pydantic import (
    BaseModel,
    Field,
    RootModel,
    ValidationError,
    ValidationInfo,
    field_validator,
)


class EmbeddingConfig(BaseModel):
    """Token embedding definition."""

    dim: int = Field(gt=0)
    vocab: int | None = Field(default=None, gt=0)
    rope: str | None = None
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class AttentionConfig(BaseModel):
    """Attention block configuration."""

    kind: Literal["MHA", "GQA", "MQA"] = "MHA"
    heads: int = Field(gt=0)
    head_dim: int = Field(gt=0, alias="head_dim")
    rope: str | None = None
    rope_theta: float | None = Field(default=None, gt=0.0)
    sw: int | None = Field(default=None, alias="sliding_window")
    kv_groups: int | None = Field(default=None, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    # Stability knobs (optional, evolvable)
    qk_norm_max: float | None = Field(default=None, gt=0.0)
    # Sparsity pattern (optional)
    sparsity: Literal["none", "sliding", "block", "local_global", "dilated", "local_block"] = "none"
    block_size: int | None = Field(default=None, gt=0)
    block_stride: int | None = Field(default=None, gt=0)
    # For local_global pattern: reuse sw as local window; add explicit global stride
    global_stride: int | None = Field(default=None, gt=0)
    dilation: int | None = Field(default=None, gt=0)

    model_config = {"populate_by_name": True}

    @property
    def hidden_dim(self) -> int:
        return self.heads * self.head_dim


class DenseFFNConfig(BaseModel):
    """Standard feed-forward block."""

    type: Literal["dense"] = "dense"
    hidden: int = Field(gt=0)
    activation: Literal["silu", "gelu", "relu", "swiglu"] = "swiglu"
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class MoEFFNConfig(BaseModel):
    """Mixture-of-Experts feed-forward block."""

    type: Literal["moe"] = "moe"
    hidden: int = Field(gt=0)
    n_experts: int = Field(gt=1)
    k: int = Field(default=2, ge=1)
    capacity_factor: float = Field(default=1.2, gt=0.0)
    balance: float = Field(default=0.05, ge=0.0)
    shared: int = Field(default=0, ge=0)
    router_temperature: float | None = Field(default=None, gt=0.0)


FfnConfig = Annotated[DenseFFNConfig | MoEFFNConfig, Field(discriminator="type")]


class SSMConfig(BaseModel):
    """State-space model parallel branch."""

    kind: Literal["mamba2", "ssm"] = "mamba2"
    d_state: int = Field(gt=0)
    d_conv: int = Field(gt=0)
    dt_rank: int = Field(gt=0)
    chunk: int = Field(gt=0)
    gate: float = Field(default=0.1, ge=0.0, le=1.0)


class HeadConfig(BaseModel):
    """LM head configuration."""

    tie_embeddings: bool = True
    vocab: int = Field(gt=0)


class RetroConfig(BaseModel):
    """Retro-attention style memory augmentation."""

    type: Literal["retro"] = "retro"
    memory_tokens: int = Field(default=1024, gt=0)
    stride: int = Field(default=64, gt=0)
    aggregator: Literal["mean", "attention", "gate"] = "gate"
    gating_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class GatedModuleConfig(BaseModel):
    """Lightweight gate that blends multiple submodules."""

    type: Literal["gated"] = "gated"
    targets: list[str] = Field(default_factory=lambda: ["attn", "ffn"])
    init_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    learnable: bool = True


class CustomModuleConfig(BaseModel):
    """Escape hatch for experimental modules without schema friction."""

    type: Literal["custom"] = "custom"
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


ExtraModuleConfig = Annotated[
    RetroConfig | GatedModuleConfig | CustomModuleConfig,
    Field(discriminator="type"),
]


class BlockConfig(BaseModel):
    """Single transformer block definition."""

    name: str | None = None
    attn: AttentionConfig | None = None
    ffn: FfnConfig | None = None
    ssm: SSMConfig | None = None
    extras: list[ExtraModuleConfig] = Field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "attn": self.attn.model_dump(mode="python") if self.attn else None,
            "ffn": self.ffn.model_dump(mode="python") if self.ffn else None,
            "ssm": self.ssm.model_dump(mode="python") if self.ssm else None,
            "extras": [extra.model_dump(mode="python") for extra in self.extras],
        }


class RecurrenceConfig(BaseModel):
    """Wrap a contiguous block range in a recurrent loop."""

    start: int
    end: int
    adapter: Literal["linear", "gated"] = "linear"
    adapter_dim: int | None = None
    concat_prelude: bool = True
    init_state: Literal["zeros", "noise"] = "zeros"
    noise_std: float = Field(default=0.02, ge=0.0)
    train_recurrence: int = Field(default=1, ge=1)
    max_train_recurrence: int = Field(default=4, ge=1)
    curriculum_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    test_recurrences: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])

    @field_validator("end")
    @classmethod
    def valid_range(cls, value: int, info: ValidationInfo) -> int:
        start = info.data.get("start", 0)
        if value <= start:
            msg = "recurrence end must be greater than start"
            raise ValueError(msg)
        return value


class TrainSchedule(BaseModel):
    """Training hyper parameters shared by all stages."""

    lr: float = Field(gt=0)
    warmup: int = Field(default=2000, ge=0)
    clip: float = Field(default=1.0, gt=0.0)
    bf16: bool = True
    grad_checkpoint: bool = Field(default=True, alias="grad_ckpt")
    max_tokens: int | None = Field(default=None, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    seed: int = 0
    # MoE routing auxiliaries (optional)
    router_lb_coeff: float = Field(default=0.0, ge=0.0)
    router_entropy_coeff: float = Field(default=0.0, ge=0.0)
    entropy_threshold: float = Field(default=0.5, ge=0.0)
    entropy_patience: int = Field(default=3, ge=1)
    instability_threshold: float = Field(default=5.0, ge=0.0)
    no_improve_patience: int = Field(default=20, ge=1)
    improvement_tolerance: float = Field(default=1e-3, ge=0.0)
    ppl_stop_threshold: float | None = Field(default=2.5, ge=0.0)
    init_checkpoint: str | None = None

    model_config = {"populate_by_name": True}
    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig())


class OptimizerConfig(BaseModel):
    """Optimizer selection and hyperparameters (non-evolvable by default)."""

    name: Literal["adamw", "lion"] = "adamw"
    lr: float | None = Field(default=None, gt=0.0)
    betas: tuple[float, float] | None = None
    eps: float | None = Field(default=None, gt=0.0)
    weight_decay: float | None = Field(default=None, ge=0.0)


class DatasetShard(BaseModel):
    """Weighted dataset slice for quick probes."""

    name: str
    split: str = "train"
    weight: float = Field(gt=0.0)
    cache_path: str | None = None
    revision: str | None = None


class DataConfig(BaseModel):
    """Collection of dataset shards."""

    tokenizer: str
    hf_revision: str = "main"
    seq_len: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=1, gt=0)
    workers: int = Field(default=2, ge=0)
    shards: list[DatasetShard] = Field(default_factory=list)
    healing_shards: list[DatasetShard] = Field(default_factory=list)
    healing_tokens: int | None = Field(default=None, gt=0)

    @field_validator("shards")
    @classmethod
    def non_empty(cls, value: list[DatasetShard]) -> list[DatasetShard]:
        if not value:
            raise ValueError("Provide at least one dataset shard.")
        return value

    @property
    def total_weight(self) -> float:
        return sum(shard.weight for shard in self.shards)


class ModelConfig(BaseModel):
    """Full architecture definition."""

    name: str = "candidate"
    emb: EmbeddingConfig
    blocks: list[BlockConfig]
    head: HeadConfig
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    recurrences: list[RecurrenceConfig] = Field(default_factory=list)

    @field_validator("blocks")
    @classmethod
    def non_zero_layers(cls, value: list[BlockConfig]) -> list[BlockConfig]:
        if not value:
            raise ValueError("Model requires at least one block.")
        return value

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    def moe_block_count(self) -> int:
        return sum(1 for block in self.blocks if isinstance(block.ffn, MoEFFNConfig))


class EvolutionConfig(BaseModel):
    """Tunable parameters governing mutation + selection."""

    rung0_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"gate_entropy_min": 1.0, "gate_entropy_max": 3.0}
    )
    rung1_tokens: int = 200_000
    rung2_tokens: int = 1_000_000
    population: int = 12
    topk_keep: float = Field(default=0.33, gt=0.0, le=1.0)
    crossover_prob: float = Field(default=0.2, ge=0.0, le=1.0)
    parent_selection: Literal["weighted", "pareto_uniform", "lexicase"] = "weighted"
    pareto_objectives: list[str] = Field(
        default_factory=lambda: ["ppl_code", "ppl_math", "long_recall", "throughput", "ram"]
    )
    objectives: dict[str, Literal["max", "min"]] | None = None
    composite_metrics: list["CompositeMetricConfig"] = Field(default_factory=list)


class PriorConfig(BaseModel):
    """Soft priors to inform budgets and a gentle manifold without constraining search."""

    tokens_per_param: float = Field(default=4.0, gt=0.0)
    window_scale: float = Field(default=6.0, gt=0.0)  # for sqrt(seq_len) * scale
    rope_theta_default: float = Field(default=10000.0, gt=0.0)
    prior_weight: float = Field(default=0.0, ge=0.0)  # selection weight; default unused
    compute_penalty_weight: float = Field(default=0.0, ge=0.0)


class ArchitectureSpec(BaseModel):
    """Top-level DSL entity."""

    model: ModelConfig
    train: TrainSchedule
    data: DataConfig
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    priors: PriorConfig = Field(default_factory=PriorConfig)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary for logging."""
        return {
            "name": self.model.name,
            "layers": self.model.n_layers,
            "moe_blocks": self.model.moe_block_count(),
            "seq_len": self.data.seq_len,
            "population": self.evolution.population,
            "custom_modules": sum(len(block.extras) for block in self.model.blocks),
        }


class ArchitectureSpecList(RootModel[list[ArchitectureSpec]]):
    """Helper for multi-candidate configs."""


def load_architecture_spec(path: str | Path) -> ArchitectureSpec:
    """Load a spec from YAML or JSON."""
    path = Path(path)
    data: Any
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    try:
        return ArchitectureSpec(**data)
    except ValidationError as exc:
        raise ValueError(f"Invalid config {path}") from exc


def save_architecture_spec(spec: ArchitectureSpec, path: str | Path) -> None:
    """Persist a spec as YAML based on file suffix."""
    path = Path(path)
    if path.suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(spec.model_dump(mode="python"), sort_keys=False))
    else:
        path.write_text(json.dumps(spec.model_dump(mode="python"), indent=2))


class CompositeMetricConfig(BaseModel):
    """Define a derived metric based on existing metrics."""

    name: str
    op: Literal["ratio", "product", "weighted_sum"] = "ratio"
    numerator: str | None = None
    denominator: str | None = None
    terms: dict[str, float] = Field(default_factory=dict)
    epsilon: float = 1e-6
