"""Synthetic evaluator used to debug the selection stack quickly."""

from __future__ import annotations

import math
import random
from typing import Protocol

from .candidates import Candidate
from .evaluation import StaticChecker, estimate_params, kv_bytes_per_token, throughput_proxy


class Evaluator(Protocol):
    def rung0(self, candidate: Candidate) -> bool: ...

    def rung1(self, candidate: Candidate) -> bool: ...

    def rung2(self, candidate: Candidate) -> bool: ...


class SimulatedEvaluator:
    """Produces deterministic-yet-random metrics for quick iteration."""

    def __init__(self, checker: StaticChecker | None = None, seed: int = 0) -> None:
        self.checker = checker or StaticChecker()
        self.seed = seed

    def _rng(self, candidate: Candidate) -> random.Random:
        salt = hash(candidate.ident) ^ self.seed
        return random.Random(salt & 0xFFFFFFFF)  # noqa: S311  # nosec B311 - deterministic seed

    def rung0(self, candidate: Candidate) -> bool:
        result = self.checker.run(candidate.spec)
        candidate.metrics.update(result.metrics)
        if not result.ok:
            candidate.status = "failed"
        return result.ok

    def rung1(self, candidate: Candidate) -> bool:
        rng = self._rng(candidate)
        moe_bonus = candidate.spec.model.moe_block_count() * 0.02
        gate_entropy = 1.5 + moe_bonus + rng.uniform(-0.2, 0.2)
        overflow = max(0.0, rng.gauss(0.02 + moe_bonus, 0.01))
        router_loss = max(0.1, rng.gauss(0.7 - moe_bonus, 0.05))
        candidate.metrics.update(
            {
                "router_loss": router_loss,
                "gate_entropy": gate_entropy,
                "capacity_overflow": overflow,
            }
        )
        success = 0.9 < gate_entropy < 3.0 and overflow < 0.2
        if not success:
            candidate.status = "failed"
        return success

    def rung2(self, candidate: Candidate) -> bool:
        rng = self._rng(candidate)
        params = estimate_params(candidate.spec)
        kv = kv_bytes_per_token(candidate.spec)
        throughput = throughput_proxy(candidate.spec, candidate.spec.data.seq_len)
        ppl_code = max(1.0, rng.gauss(1.6, 0.1) - math.log10(params) * 0.05)
        ppl_math = ppl_code + rng.uniform(-0.1, 0.1)
        long_recall = max(0.0, min(1.0, 0.5 + rng.uniform(-0.05, 0.05) + throughput * 0.01))
        candidate.metrics.update(
            {
                "ppl_code": ppl_code,
                "ppl_math": ppl_math,
                "long_recall": long_recall,
                "throughput": throughput,
                "ram": params * 2 / (1024**3),
                "kv_bytes_per_token": kv,
            }
        )
        candidate.status = "completed"
        return True


def evaluator_for_mode(mode: str, checker: StaticChecker | None = None, seed: int = 0) -> Evaluator:
    if mode == "simulate":
        return SimulatedEvaluator(checker=checker, seed=seed)
    raise ValueError(f"Unsupported evaluator mode '{mode}'")
