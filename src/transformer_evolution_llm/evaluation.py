"""Static and dynamic evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dsl import ArchitectureSpec, BlockConfig, DenseFFNConfig, MoEFFNConfig


def _attn_hidden(block: BlockConfig) -> int:
    if not block.attn:
        return 0
    return block.attn.heads * block.attn.head_dim


def estimate_params(spec: ArchitectureSpec) -> float:
    """Crude parameter count estimator."""
    vocab_value = spec.model.emb.vocab or spec.model.head.vocab
    if vocab_value is None:
        msg = "Vocabulary size must be specified on embedding or head."
        raise ValueError(msg)
    vocab = int(vocab_value)
    params = float(spec.model.emb.dim * vocab)  # embeddings
    for block in spec.model.blocks:
        hidden = _attn_hidden(block)
        if hidden:
            params += 3 * hidden * hidden  # qkv + proj
        if isinstance(block.ffn, DenseFFNConfig):
            params += 2 * hidden * block.ffn.hidden
        elif isinstance(block.ffn, MoEFFNConfig):
            params += block.ffn.n_experts * block.ffn.hidden * hidden
    params += spec.model.head.vocab * spec.model.emb.dim
    return params


def kv_bytes_per_token(spec: ArchitectureSpec) -> float:
    dtype_bytes = 2  # fp16/bf16
    total = 0
    for block in spec.model.blocks:
        if not block.attn:
            continue
        kv_groups = block.attn.kv_groups or block.attn.heads
        total += (block.attn.heads + kv_groups) * block.attn.head_dim * dtype_bytes
    return total


def throughput_proxy(spec: ArchitectureSpec, seq_len: int) -> float:
    """Rough tokens/second proxy."""
    hidden = sum(_attn_hidden(block) for block in spec.model.blocks)
    denom = max(1, hidden * spec.model.n_layers * seq_len)
    return 1e9 / denom


@dataclass
class StaticCheckResult:
    ok: bool
    metrics: dict[str, float]
    reasons: list[str]


class StaticChecker:
    """Rung 0 filtering without any training."""

    def __init__(
        self,
        max_params: float = 8e9,
        max_kv_bytes: float = 48_000,
        min_throughput: float = 1.0,
    ) -> None:
        self.max_params = max_params
        self.max_kv_bytes = max_kv_bytes
        self.min_throughput = min_throughput

    def run(self, spec: ArchitectureSpec) -> StaticCheckResult:
        params = estimate_params(spec)
        kv = kv_bytes_per_token(spec)
        tps = throughput_proxy(spec, spec.data.seq_len)
        reasons: list[str] = []
        # Sanity bounds for new knobs
        for block in spec.model.blocks:
            if (
                block.attn
                and block.attn.qk_norm_max is not None
                and not (0.0 < block.attn.qk_norm_max <= 50.0)
            ):
                reasons.append("qk_norm_max outside (0, 50]")
            if block.attn and block.attn.kv_groups is not None:
                if block.attn.kv_groups <= 0:
                    reasons.append("kv_groups must be >=1")
                if block.attn.kv_groups > block.attn.heads:
                    reasons.append("kv_groups cannot exceed heads")
            if (
                block.attn
                and block.attn.block_size is not None
                and block.attn.block_size > spec.data.seq_len
            ):
                reasons.append("block_size exceeds seq_len")
            if block.attn and block.attn.block_stride is not None:
                if block.attn.block_stride <= 0 or block.attn.block_stride > spec.data.seq_len:
                    reasons.append("block_stride must be in (0, seq_len]")
            if block.attn and block.attn.sw is not None and block.attn.sw <= 0:
                reasons.append("sliding_window must be > 0")
            if block.attn and block.attn.sw is not None and block.attn.sw > spec.data.seq_len:
                reasons.append("sliding_window exceeds seq_len")
            if block.attn and block.attn.sparsity == "local_global":
                if block.attn.sw is None or block.attn.sw <= 0:
                    reasons.append("local_global requires positive sliding_window (sw)")
                if (
                    block.attn.global_stride is None
                    or block.attn.global_stride <= 0
                    or block.attn.global_stride > spec.data.seq_len
                ):
                    reasons.append("local_global requires 0 < global_stride <= seq_len")
            if block.attn and block.attn.sparsity == "local_block":
                if block.attn.sw is None or block.attn.sw <= 0:
                    reasons.append("local_block requires sliding_window (sw)")
                if block.attn.block_size is None or block.attn.block_size <= 0:
                    reasons.append("local_block requires block_size > 0")
                if (
                    block.attn.block_stride is None
                    or block.attn.block_stride <= 0
                    or block.attn.block_stride > spec.data.seq_len
                ):
                    reasons.append("local_block requires 0 < block_stride <= seq_len")
            if block.attn and block.attn.dilation is not None and block.attn.dilation <= 0:
                reasons.append("dilation must be > 0 when using dilated sparsity")
        if params > self.max_params:
            reasons.append(f"params {params/1e9:.2f}B exceeds {self.max_params/1e9:.2f}B limit")
        if kv > self.max_kv_bytes:
            reasons.append(f"kv bytes/token {kv:.0f} > limit {self.max_kv_bytes}")
        if tps < self.min_throughput:
            reasons.append(f"throughput proxy {tps:.2f} below {self.min_throughput}")
        metrics = {
            "params": params,
            "kv_bytes_per_token": kv,
            "throughput_proxy": tps,
        }
        return StaticCheckResult(ok=not reasons, metrics=metrics, reasons=reasons)


def merge_metrics(existing: dict[str, float], updates: dict[str, Any]) -> dict[str, float]:
    merged = dict(existing)
    for key, value in updates.items():
        if isinstance(value, (int, float)):
            merged[key] = float(value)
    return merged
