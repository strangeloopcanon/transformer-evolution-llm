"""Evolution loop orchestration."""

from __future__ import annotations

import random
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import ujson as json
from rich.console import Console
from rich.table import Table

from .candidates import Candidate, ObjectiveDirection, ParetoFrontier
from .crossover import merge_checkpoints, splice_blocks
from .data import DataModule
from .dsl import ArchitectureSpec, CompositeMetricConfig, EvolutionConfig
from .evaluation import StaticChecker, estimate_params
from .mutations import mutate
from .simulators import evaluator_for_mode
from .trainer import FullWeightTrainer

console = Console()

ObjectiveDir = dict[str, ObjectiveDirection]


def default_objectives() -> ObjectiveDir:
    return {
        "ppl_code": "min",
        "ppl_math": "min",
        "long_recall": "max",
        "throughput": "max",
        "ram": "min",
        "layers": "max",
        "moe_blocks": "max",
        "novelty": "max",
        "instability": "min",
        "graph_entropy": "max",
    }


class EvolutionRunner:
    """Coordinates mutation, evaluation, and frontier tracking."""

    def __init__(
        self,
        base_spec: ArchitectureSpec,
        evolution_cfg: EvolutionConfig,
        mode: str = "simulate",
        objective_dir: ObjectiveDir | None = None,
        seed: int = 0,
        score_weight_overrides: dict[str, float] | None = None,
    ) -> None:
        self.base_spec = base_spec
        self.cfg = evolution_cfg
        self.mode = mode
        config_objectives = getattr(self.cfg, "objectives", None)
        self.objective_dir = objective_dir or config_objectives or default_objectives()
        self.score_weights = {
            k: (1.0 if v == "max" else -1.0)
            * (score_weight_overrides.get(k, 1.0) if score_weight_overrides else 1.0)
            for k, v in self.objective_dir.items()
        }
        self.mutation_weights: dict[str, float] | None = None
        self.mutation_steps: int = int(getattr(self.cfg, "mutation_steps", 1) or 1)
        # Structural elite retention to keep deeper/MoE-rich candidates alive
        self.structural_elite_k = int(getattr(self.cfg, "structural_elite_k", 2) or 0)
        self.structural_elite_weights: dict[str, float] = {
            "layers": 1.0,
            "moe_blocks": 3.0,
            "selector_blocks": 2.0,
        }
        cfg_elite_weights = getattr(self.cfg, "structural_elite_weights", None)
        if isinstance(cfg_elite_weights, dict):
            self.structural_elite_weights.update(
                {k: float(v) for k, v in cfg_elite_weights.items() if k in self.structural_elite_weights}
            )
        self.frontier = ParetoFrontier(self.objective_dir)
        self.rng = random.Random(seed)  # noqa: S311  # nosec B311 - seeded per run
        self.checker = StaticChecker(
            max_params=8.0e9,
            max_kv_bytes=64_000,
            min_throughput=0.5,
        )
        self.trainer = FullWeightTrainer() if mode == "live" else None
        self.data_module = DataModule(base_spec.data) if mode == "live" else None
        self.evaluator = (
            None if mode == "live" else evaluator_for_mode(mode, checker=self.checker, seed=seed)
        )
        configured_composites = getattr(self.cfg, "composite_metrics", []) or []
        self._composite_metrics = self._merge_composites(
            configured_composites, self._default_composites()
        )
        self.pool: list[Candidate] = []
        self.counter = 0
        self.checkpoint_dir = Path("runs/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        init_ckpt = getattr(base_spec.train, "init_checkpoint", None)
        self._init_checkpoint: Path | None = Path(init_ckpt) if init_ckpt else None
        # lineage tracking: candidate id -> list of parent ids
        self._parents: dict[str, list[str]] = {}
        self._history: list[Candidate] = []
        # rung schedule ratios relative to configured trainer.steps
        self._rung1_ratio = 0.2
        self._rung2_ratio = 1.0
        self.ppl_stop_threshold = base_spec.train.ppl_stop_threshold
        # Promotion heuristics (live mode) for high-budget candidates
        self._promotion_prob = float(getattr(self.cfg, "promotion_prob", 0.0) or 0.0)
        self._promotion_min_layers = int(getattr(self.cfg, "promotion_min_layers", 0) or 0)
        self._promotion_min_moe_blocks = int(getattr(self.cfg, "promotion_min_moe_blocks", 0) or 0)
        self._promotion_steps_multiplier = float(
            getattr(self.cfg, "promotion_steps_multiplier", 1.0) or 1.0
        )
        self._promotion_tokens_multiplier = float(
            getattr(self.cfg, "promotion_tokens_multiplier", 1.0) or 1.0
        )
        self._promotion_min_router_entropy = float(
            getattr(self.cfg, "promotion_min_router_entropy", 0.0) or 0.0
        )
        self._promotion_min_recurrence_gain = float(
            getattr(self.cfg, "promotion_min_recurrence_gain", 0.0) or 0.0
        )
        self._promotion_max_instability = getattr(self.cfg, "promotion_max_instability", None)

    def run(self, generations: int) -> list[Candidate]:
        base_candidate = Candidate(
            ident=self._new_id("seed"), spec=self.base_spec.model_copy(deep=True)
        )
        if self._init_checkpoint is not None:
            base_candidate.seed_state_path = self._init_checkpoint
        self._parents[base_candidate.ident] = []
        self._evaluate_candidate(base_candidate)
        self.pool.append(base_candidate)
        self._history.append(base_candidate)
        survivors = [base_candidate]
        for _ in range(generations):
            candidate = self._spawn_candidate()
            console.print(f"[cyan]Evaluating[/] {candidate.ident}")
            self._evaluate_candidate(candidate)
            if candidate.status == "completed":
                survivors.append(candidate)
            self.pool.append(candidate)
            self._history.append(candidate)
            self._trim_pool()
        return survivors

    def _evaluate_candidate(self, candidate: Candidate) -> None:
        candidate.metrics["layers"] = float(candidate.spec.model.n_layers)
        candidate.metrics["moe_blocks"] = float(candidate.spec.model.moe_block_count())
        candidate.metrics["graph_entropy"] = self._graph_entropy(candidate.spec)
        # Selector-related proxies: count blocks with selector enabled and average top-k
        selector_blocks: float = 0.0
        selector_topk_sum: float = 0.0
        selector_count: float = 0.0
        for block in candidate.spec.model.blocks:
            attn = block.attn
            if attn and getattr(attn, "selector", "none") != "none":
                selector_blocks += 1.0
                topk_val = getattr(attn, "selector_topk", None)
                if topk_val is not None:
                    selector_topk_sum += float(topk_val)
                    selector_count += 1.0
        candidate.metrics["selector_blocks"] = selector_blocks
        candidate.metrics["selector_topk_avg"] = (
            selector_topk_sum / selector_count if selector_count > 0 else 0.0
        )
        # novelty vs parent or base
        ref = None
        if candidate.parent:
            parent = next((c for c in self.pool if c.ident == candidate.parent), None)
            ref = parent.spec if parent else self.base_spec
        else:
            ref = self.base_spec
        candidate.metrics["novelty"] = float(self._structural_distance(ref, candidate.spec))
        # Enforce rung0 thresholds even in live mode to block trivial models
        thresholds = getattr(self.cfg, "rung0_thresholds", {}) or {}
        min_layers = float(thresholds.get("min_layers", 0.0) or 0.0)
        min_moe = float(thresholds.get("min_moe_blocks", 0.0) or 0.0)
        min_selector = float(thresholds.get("min_selector_blocks", 0.0) or 0.0)
        if (
            candidate.metrics["layers"] < min_layers
            or candidate.metrics["moe_blocks"] < min_moe
            or candidate.metrics["selector_blocks"] < min_selector
        ):
            candidate.status = "failed"
            return
        if self.mode == "live":
            static = self.checker.run(candidate.spec)
            candidate.metrics.update(static.metrics)
            if not static.ok:
                candidate.status = "failed"
                return
            if self.trainer is None or self.data_module is None:
                raise RuntimeError("Live mode requires trainer and data module.")
            # Prior-aware token budget
            params = float(estimate_params(candidate.spec))
            tokens_budget = int(candidate.spec.priors.tokens_per_param * params)
            base_tokens = candidate.spec.train.max_tokens or self.cfg.rung2_tokens
            mult = 1.0
            if candidate.spec.model.n_layers >= 4:
                mult += 0.2
            if candidate.spec.model.moe_block_count() >= 1:
                mult += 0.1
            scaled_tokens = int(min(base_tokens * mult, tokens_budget))
            # Multi-fidelity schedule: rung1 (short), possibly rung2 (full)
            base_steps = getattr(self.trainer, "steps", None)
            if not isinstance(base_steps, int):
                base_steps = 100
            # Rung 1
            self.trainer.steps = max(1, int(base_steps * self._rung1_ratio))
            batches = self.data_module.batches(max_tokens=scaled_tokens)
            seed_state = candidate.seed_state_path or candidate.parent_checkpoint
            metrics1, checkpoint = self.trainer.train(
                candidate=candidate,
                spec=candidate.spec,
                batch_iter=batches,
                seed_state_path=seed_state,
            )
            candidate.metrics.update(metrics1)
            candidate.checkpoint = checkpoint
            # Early stop heuristic: clearly poor ppl
            ppl1 = float(candidate.metrics.get("ppl_code", 1e9))
            threshold = self.ppl_stop_threshold
            if threshold is not None and ppl1 > threshold:
                candidate.status = "completed"
                self._apply_composite_metrics(candidate)
                self.frontier.update(candidate)
                # restore trainer steps
                self.trainer.steps = base_steps
                return
            # Rung 2 (full)
            self.trainer.steps = max(1, int(base_steps * self._rung2_ratio))
            batches = self.data_module.batches(max_tokens=scaled_tokens)
            metrics2, checkpoint = self.trainer.train(
                candidate=candidate,
                spec=candidate.spec,
                batch_iter=batches,
                seed_state_path=checkpoint,
            )
            candidate.metrics.update(metrics2)
            candidate.checkpoint = checkpoint
            if candidate.seed_state_path is not None:
                try:
                    Path(candidate.seed_state_path).unlink(missing_ok=True)
                except OSError:
                    pass
                candidate.seed_state_path = None
            candidate.status = "completed"
            # Optional promotion rung: give complex candidates extra budget
            if self._should_promote(candidate, tokens_budget, scaled_tokens):
                self._run_promotion(candidate, base_steps, tokens_budget, scaled_tokens)
            # Prior distance metric (not in objectives by default)
            candidate.metrics["prior_distance"] = self._prior_distance(candidate.spec)
            self._apply_composite_metrics(candidate)
            self.frontier.update(candidate)
            # Restore trainer step budget
            self.trainer.steps = base_steps
            return
        if not self.evaluator.rung0(candidate):  # type: ignore[union-attr]
            return
        candidate.rung = 1
        if not self.evaluator.rung1(candidate):  # type: ignore[union-attr]
            return
        candidate.rung = 2
        if not self.evaluator.rung2(candidate):  # type: ignore[union-attr]
            return
        self._apply_composite_metrics(candidate)
        self.frontier.update(candidate)

    def _should_promote(self, candidate: Candidate, tokens_budget: int, used_tokens: int) -> bool:
        """Decide whether to apply a high-budget promotion rung to this candidate."""
        if self._promotion_prob <= 0.0:
            return False
        if self.mode != "live" or self.trainer is None or self.data_module is None:
            return False
        if tokens_budget <= 0 or used_tokens >= tokens_budget:
            return False
        layers = candidate.spec.model.n_layers
        moe_blocks = candidate.spec.model.moe_block_count()
        if layers < self._promotion_min_layers:
            return False
        if moe_blocks < self._promotion_min_moe_blocks:
            return False
        router_entropy = candidate.metrics.get("router_entropy")
        if router_entropy is not None and router_entropy < self._promotion_min_router_entropy:
            return False
        recurrence_gain = candidate.metrics.get("recurrence_gain")
        if recurrence_gain is not None and recurrence_gain < self._promotion_min_recurrence_gain:
            return False
        if self._promotion_max_instability is not None:
            instability = candidate.metrics.get("instability")
            if instability is not None and instability > self._promotion_max_instability:
                return False
        if self.rng.random() > self._promotion_prob:
            return False
        return True

    def _run_promotion(
        self,
        candidate: Candidate,
        base_steps: int,
        tokens_budget: int,
        used_tokens: int,
    ) -> None:
        """Apply an additional high-budget training rung to the candidate."""
        if self.trainer is None or self.data_module is None:
            return
        extra_tokens = int(self._promotion_tokens_multiplier * max(tokens_budget - used_tokens, 0))
        if extra_tokens <= 0:
            return
        max_tokens = min(tokens_budget, used_tokens + extra_tokens)
        promo_steps = max(1, int(base_steps * self._promotion_steps_multiplier))
        original_steps = self.trainer.steps
        try:
            self.trainer.steps = promo_steps
            batches = self.data_module.batches(max_tokens=max_tokens)
            seed_state = candidate.checkpoint
            metrics3, checkpoint3 = self.trainer.train(
                candidate=candidate,
                spec=candidate.spec,
                batch_iter=batches,
                seed_state_path=seed_state,
            )
            candidate.metrics.update(metrics3)
            candidate.checkpoint = checkpoint3
        finally:
            self.trainer.steps = original_steps

    @staticmethod
    def _structural_distance(a: ArchitectureSpec, b: ArchitectureSpec) -> float:
        la, lb = a.model.n_layers, b.model.n_layers
        diff: float = float(abs(la - lb))
        for i in range(min(la, lb)):
            ba = a.model.blocks[i]
            bb = b.model.blocks[i]
            ta = getattr(ba.ffn, "type", None) if ba.ffn else None
            tb = getattr(bb.ffn, "type", None) if bb.ffn else None
            if ta != tb:
                diff += 1.0
            if bool(ba.ssm) != bool(bb.ssm):
                diff += 1.0
            if len(ba.extras) != len(bb.extras):
                diff += 0.5
            if ba.attn and bb.attn:
                if (ba.attn.kv_groups or ba.attn.heads) != (bb.attn.kv_groups or bb.attn.heads):
                    diff += 0.5
                if (ba.attn.rope or None) != (bb.attn.rope or None):
                    diff += 0.5
                if ba.attn.sparsity != bb.attn.sparsity:
                    diff += 0.5
                if (ba.attn.sw or None) != (bb.attn.sw or None):
                    diff += 0.25
                if (ba.attn.global_stride or None) != (bb.attn.global_stride or None):
                    diff += 0.25
                if (ba.attn.block_size or None) != (bb.attn.block_size or None):
                    diff += 0.25
                if (ba.attn.block_stride or None) != (bb.attn.block_stride or None):
                    diff += 0.25
        # Recurrence differences matter for novelty
        rec_a = [(r.start, r.end, r.adapter, r.concat_prelude) for r in a.model.recurrences]
        rec_b = [(r.start, r.end, r.adapter, r.concat_prelude) for r in b.model.recurrences]
        diff += abs(len(rec_a) - len(rec_b)) * 0.5
        for idx in range(min(len(rec_a), len(rec_b))):
            if rec_a[idx] != rec_b[idx]:
                diff += 0.5
        denom = max(1.0, 0.5 * float(la + lb))
        return float(diff) / denom

    def _prior_distance(self, spec: ArchitectureSpec) -> float:
        # Gentle distance from a typical manifold:
        # - head_dim≈64; FFN≈4×d_model; kv_groups≈2
        # - local_global windows ≈ sqrt(seq_len) * window_scale
        # - rope_theta ≈ rope_theta_default
        import math

        d_model = spec.model.emb.dim
        seq_len = spec.data.seq_len
        scale = spec.priors.window_scale
        rope_default = spec.priors.rope_theta_default
        target_ffn = 4.0 * d_model
        target_hd = 64.0
        target_kv = 2.0
        target_sw = math.sqrt(max(1.0, float(seq_len))) * scale
        dist = 0.0
        count = 1e-6
        for block in spec.model.blocks:
            if block.attn:
                count += 1.0
                dist += abs(float(block.attn.head_dim) - target_hd) / target_hd
                if block.attn.kv_groups is not None:
                    dist += abs(float(block.attn.kv_groups) - target_kv) / target_kv * 0.5
                if (block.attn.sparsity or "none") == "local_global":
                    if block.attn.sw is not None:
                        dist += abs(float(block.attn.sw) - target_sw) / max(1.0, target_sw) * 0.5
                    if block.attn.global_stride is not None:
                        target_g = max(1.0, math.sqrt(float(seq_len)))
                        dist += abs(float(block.attn.global_stride) - target_g) / target_g * 0.25
                if block.attn.rope_theta is not None:
                    dist += abs(float(block.attn.rope_theta) - rope_default) / rope_default * 0.25
            if block.ffn is not None and getattr(block.ffn, "type", "dense") == "dense":
                count += 1.0
                dist += abs(float(block.ffn.hidden) - target_ffn) / target_ffn
        return float(dist / count)

    @staticmethod
    def _graph_entropy(spec: ArchitectureSpec) -> float:
        import math

        tokens: list[str] = []
        for block in spec.model.blocks:
            if block.attn:
                tokens.append(f"attn:{block.attn.kind}")
                tokens.append(f"sparsity:{block.attn.sparsity or 'none'}")
                if block.attn.gating_pos and block.attn.gating_pos != "none":
                    tokens.append(f"gate:{block.attn.gating_pos}-{block.attn.gating_op or 'dense'}")
            if block.ffn:
                tokens.append(f"ffn:{getattr(block.ffn, 'type', 'dense')}")
            if block.ssm:
                tokens.append(f"ssm:{block.ssm.kind}")
            for extra in block.extras:
                tokens.append(f"extra:{getattr(extra, 'type', type(extra).__name__)}")
        for rec in spec.model.recurrences:
            tokens.append(f"rec:{rec.start}-{rec.end}")
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        total = sum(counts.values())
        probs = [c / total for c in counts.values() if c > 0]
        entropy = -sum(p * math.log(p) for p in probs)
        diversity = len(counts)
        depth_bonus = math.log1p(spec.model.n_layers)
        return float(entropy + 0.05 * diversity + 0.1 * depth_bonus)

    def _apply_composite_metrics(self, candidate: Candidate) -> None:
        if not self._composite_metrics:
            return
        for comp in self._composite_metrics:
            value = self._compute_composite(comp, candidate.metrics)
            if value is None:
                continue
            candidate.metrics[comp.name] = value

    @staticmethod
    def _compute_composite(comp, metrics: dict[str, float]) -> float | None:
        try:
            if comp.op == "ratio":
                if not comp.numerator or not comp.denominator:
                    return None
                num = metrics.get(comp.numerator)
                den = metrics.get(comp.denominator)
                if num is None or den is None:
                    return None
                denom = den if abs(den) >= comp.epsilon else comp.epsilon
                return float(num) / float(denom)
            if comp.op == "product":
                if not comp.numerator or not comp.denominator:
                    return None
                num = metrics.get(comp.numerator)
                den = metrics.get(comp.denominator)
                if num is None or den is None:
                    return None
                return float(num) * float(den)
            if comp.op == "weighted_sum":
                if not comp.terms:
                    return None
                total = 0.0
                for name, weight in comp.terms.items():
                    val = metrics.get(name)
                    if val is None:
                        return None
                    total += float(weight) * float(val)
                return total
        except Exception:
            return None
        return None

    @staticmethod
    def _default_composites() -> list[CompositeMetricConfig]:
        return [
            CompositeMetricConfig(
                name="ppl_per_long_recall",
                op="ratio",
                numerator="ppl_code",
                denominator="long_recall",
                epsilon=1e-3,
            ),
            CompositeMetricConfig(
                name="ppl_per_param",
                op="ratio",
                numerator="ppl_code",
                denominator="params",
                epsilon=1e-6,
            ),
            CompositeMetricConfig(
                name="ppl_per_throughput",
                op="ratio",
                numerator="ppl_code",
                denominator="throughput",
                epsilon=1e-6,
            ),
        ]

    @staticmethod
    def _merge_composites(
        primary: list[CompositeMetricConfig], defaults: list[CompositeMetricConfig]
    ) -> list[CompositeMetricConfig]:
        existing = {comp.name for comp in primary}
        merged = list(primary)
        for comp in defaults:
            if comp.name not in existing:
                merged.append(comp)
        return merged

    def _select_parent(self) -> Candidate:
        strategy = getattr(self.cfg, "parent_selection", "weighted")
        if strategy == "pareto_uniform" and self.frontier.entries:
            return self.rng.choice(self.frontier.entries)
        if strategy == "lexicase" and self.pool:
            # Lexicase: randomly order objectives; progressively filter to the best on each
            objectives = list(self.objective_dir.keys())
            self.rng.shuffle(objectives)
            candidates = list(self.pool)
            for obj in objectives:
                if not candidates:
                    break
                direction = self.objective_dir[obj]
                best_val = None
                for cand in candidates:
                    val = cand.metrics.get(obj, 0.0)
                    if best_val is None:
                        best_val = val
                    else:
                        if (direction == "max" and val > best_val) or (
                            direction == "min" and val < best_val
                        ):
                            best_val = val
                # keep top 10% (or all equal) on this objective
                if best_val is not None:
                    tol = 1e-9
                    filtered = []
                    # compute threshold for top-10%
                    values = [c.metrics.get(obj, 0.0) for c in candidates]
                    if direction == "max":
                        values_sorted = sorted(values, reverse=True)
                    else:
                        values_sorted = sorted(values)
                    cutoff_idx = max(0, int(0.1 * (len(values_sorted) - 1)))
                    threshold = values_sorted[cutoff_idx]
                    for cand in candidates:
                        val = cand.metrics.get(obj, 0.0)
                        if direction == "max":
                            if val + tol >= threshold:
                                filtered.append(cand)
                        else:
                            if val - tol <= threshold:
                                filtered.append(cand)
                    candidates = filtered
            if candidates:
                return self.rng.choice(candidates)
        # Default weighted tournament among 3
        contenders = self.rng.sample(self.pool, k=min(3, len(self.pool)))
        return max(contenders, key=lambda cand: cand.score(self.score_weights))

    def _trim_pool(self) -> None:
        if len(self.pool) <= self.cfg.population:
            return
        # Protect structural elites from trimming
        elite_ids: set[str] = set()
        if self.structural_elite_k > 0 and self.pool:
            scored = [(self._structural_score(c), c) for c in self.pool]
            scored.sort(key=lambda t: t[0], reverse=True)
            elite_ids = {c.ident for _, c in scored[: self.structural_elite_k]}
        excess = len(self.pool) - self.cfg.population
        removed: list[Candidate] = []
        kept: list[Candidate] = []
        for cand in self.pool:
            if cand.ident in elite_ids:
                kept.append(cand)
            elif excess > 0:
                removed.append(cand)
                excess -= 1
            else:
                kept.append(cand)
        self.pool = kept
        for candidate in removed:
            self._remove_candidate_artifacts(candidate)

    def _spawn_candidate(self) -> Candidate:
        if (
            self.mode == "live"
            and len(self.pool) >= 2
            and self.rng.random() < self.cfg.crossover_prob
        ):
            parent_a, parent_b = self.rng.sample(self.pool, 2)
            blocks, cut_a, cut_b = splice_blocks(parent_a.spec, parent_b.spec, self.rng)
            spec_data = parent_a.spec.model_dump(mode="python")
            spec_data["model"]["blocks"] = [block.model_dump(mode="python") for block in blocks]
            spec = ArchitectureSpec(**spec_data)
            child_id = self._new_id("xover")
            seed_path = self.checkpoint_dir / f"{child_id}_seed.pt"
            merge_checkpoints(
                child_spec=spec,
                cut_a=cut_a,
                cut_b=cut_b,
                parent_a_blocks=len(parent_a.spec.model.blocks),
                parent_b_blocks=len(parent_b.spec.model.blocks),
                parent_a_ckpt=parent_a.checkpoint,
                parent_b_ckpt=parent_b.checkpoint,
                out_path=seed_path,
            )
            self._parents[child_id] = [parent_a.ident, parent_b.ident]
            return Candidate(
                ident=child_id,
                spec=spec,
                parent=None,
                seed_state_path=seed_path,
            )
        parent = self._select_parent()
        name, spec = mutate(
            parent.spec, self.rng, self.mutation_weights, steps=getattr(self, "mutation_steps", 1)
        )
        child = Candidate(
            ident=self._new_id(name),
            spec=spec,
            parent=parent.ident,
            parent_checkpoint=parent.checkpoint,
        )
        self._parents[child.ident] = [parent.ident]
        return child

    def _structural_score(self, cand: Candidate) -> float:
        """Score structural richness to keep depth/MoE/selector candidates alive."""
        metrics = cand.metrics
        layers = float(metrics.get("layers") or cand.spec.model.n_layers)
        moe_blocks = float(metrics.get("moe_blocks") or 0.0)
        selector_blocks = float(metrics.get("selector_blocks") or 0.0)
        w = self.structural_elite_weights
        return (
            w.get("layers", 0.0) * layers
            + w.get("moe_blocks", 0.0) * moe_blocks
            + w.get("selector_blocks", 0.0) * selector_blocks
        )

    def frontier_table(self) -> Table:
        table = Table(title="Pareto Frontier")
        table.add_column("ID")
        table.add_column("ppl_code")
        table.add_column("long_recall")
        table.add_column("throughput")
        for cand in self.frontier.entries:
            table.add_row(
                cand.ident,
                f"{cand.metrics.get('ppl_code', 0.0):.2f}",
                f"{cand.metrics.get('long_recall', 0.0):.2f}",
                f"{cand.metrics.get('throughput', 0.0):.2f}",
            )
        return table

    def save_frontier(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.frontier.to_json(), indent=2))
        console.print(f"Frontier saved to {path}")

    def save_state(self, path: Path) -> None:
        """Persist runner state for resuming later."""
        path.parent.mkdir(parents=True, exist_ok=True)
        seed_val: int | None = None
        try:
            # random.Random does not expose the seed value directly.
            seed_val = None
        except Exception:
            seed_val = None
        state: dict[str, Any] = {
            "seed": seed_val,
            "rng_state": self.rng.getstate(),
            "counter": self.counter,
            "checkpoint_dir": str(self.checkpoint_dir),
            "init_checkpoint": str(self._init_checkpoint) if self._init_checkpoint else None,
            "objective_dir": self.objective_dir,
            "score_weights": self.score_weights,
            "pool": [c.serialize() for c in self.pool],
            "frontier": [c.ident for c in self.frontier.entries],
            "parents": self._parents,
            "history": [c.serialize() for c in self._history],
            "mutation_weights": self.mutation_weights,
            "mutation_steps": self.mutation_steps,
            "structural_elite": {
                "k": self.structural_elite_k,
                "weights": self.structural_elite_weights,
            },
        }
        path.write_text(json.dumps(state, indent=2))
        console.print(f"State saved to {path}")

    @classmethod
    def load_state(
        cls,
        path: Path,
        mode: str = "simulate",
        score_weight_overrides: dict[str, float] | None = None,
    ) -> EvolutionRunner:
        """Rehydrate a runner from a saved state manifest."""
        data = json.loads(path.read_text())
        checkpoint_dir = Path(data.get("checkpoint_dir", "runs/checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Rebuild candidates
        pool: list[Candidate] = []
        history: list[Candidate] = []
        for item in data.get("pool", []):
            pool.append(Candidate.from_json(item))
        for item in data.get("history", []):
            history.append(Candidate.from_json(item))
        # Base spec: use the first candidate's spec as base
        if not history and not pool:
            raise ValueError("State file contains no candidates.")
        base_spec = (history or pool)[0].spec.model_copy(deep=True)
        evo_cfg = base_spec.evolution
        seed_val = data.get("seed")
        if not isinstance(seed_val, int):
            seed_val = 0
        runner = cls(
            base_spec=base_spec,
            evolution_cfg=evo_cfg,
            mode=mode,
            objective_dir=data.get("objective_dir"),
            seed=seed_val,
            score_weight_overrides=score_weight_overrides or data.get("score_weights"),
        )
        runner.mutation_weights = data.get("mutation_weights")
        if "mutation_steps" in data:
            try:
                runner.mutation_steps = int(data["mutation_steps"])
            except Exception:
                pass
        elite_cfg = data.get("structural_elite") or {}
        try:
            runner.structural_elite_k = int(elite_cfg.get("k", runner.structural_elite_k))
            weights = elite_cfg.get("weights")
            if isinstance(weights, dict):
                runner.structural_elite_weights.update(
                    {k: float(v) for k, v in weights.items() if k in runner.structural_elite_weights}
                )
        except Exception:
            pass
        runner.checkpoint_dir = checkpoint_dir
        init_ckpt = data.get("init_checkpoint")
        runner._init_checkpoint = Path(init_ckpt) if init_ckpt else None
        runner.counter = int(data.get("counter", 0))
        runner.pool = pool
        runner._history = history
        runner._parents = data.get("parents", {})
        # Rebuild frontier entries by id lookup
        id_to_candidate = {c.ident: c for c in pool}
        frontier_ids = data.get("frontier", [])
        runner.frontier._entries = [
            id_to_candidate[cid] for cid in frontier_ids if cid in id_to_candidate
        ]
        # Restore RNG
        rng_state = data.get("rng_state")
        if rng_state:
            # Best-effort restore; ignore if incompatible
            if isinstance(rng_state, (list, tuple)):
                try:
                    runner.rng.setstate(tuple(rng_state))
                except Exception:
                    console.print(
                        "[yellow]Warning:[/] failed to restore RNG state; continuing with new seed."
                    )
        return runner

    def _new_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}-{self.counter}-{uuid.uuid4().hex[:4]}"

    def _remove_candidate_artifacts(self, candidate: Candidate) -> None:
        for file_path in (candidate.checkpoint, candidate.seed_state_path):
            if file_path is None:
                continue
            try:
                Path(file_path).unlink(missing_ok=True)
            except OSError:
                pass

    def save_lineage(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        nodes = []
        for cand in self._history:
            node = {
                "id": cand.ident,
                "parents": self._parents.get(cand.ident, []),
                "status": cand.status,
                "rung": cand.rung,
                "metrics": cand.metrics,
                "spec": cand.spec.model_dump(mode="python"),
            }
            nodes.append(node)
        path.write_text(json.dumps(nodes, indent=2))
        console.print(f"Lineage saved to {path}")
