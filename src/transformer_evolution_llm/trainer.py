"""Live training loop for mutated architectures."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .candidates import Candidate
from .data import TokenBatch
from .dsl import ArchitectureSpec
from .models import EvolutionModel, MoELayer, count_parameters
from .morphology import match_experts_to_parent, sort_moe_experts
from .optimizers import build_optimizer


class FullWeightTrainer:
    """Runs a short, full-weight finetune for each candidate."""

    def __init__(
        self,
        checkpoint_dir: Path = Path("runs/checkpoints"),
        device: str | None = None,
        steps: int = 50,
        eval_batches: int = 2,
        entropy_threshold: float = 0.5,
        entropy_patience: int = 3,
        instability_threshold: float = 5.0,
        no_improve_patience: int = 20,
        improvement_tolerance: float = 1e-3,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if device:
            self.device = torch.device(device)
        elif torch.backends.cuda.is_built() and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.steps = steps
        self.eval_batches = eval_batches
        self.entropy_threshold = entropy_threshold
        self.entropy_patience = entropy_patience
        self.instability_threshold = instability_threshold
        self.no_improve_patience = no_improve_patience
        self.improvement_tolerance = improvement_tolerance

    def train(
        self,
        candidate: Candidate,
        spec: ArchitectureSpec,
        batch_iter: Iterable[TokenBatch],
        seed_state_path: Path | None = None,
    ) -> tuple[dict[str, float], Path]:
        model = EvolutionModel(spec.model).to(self.device)
        sort_moe_experts(model)
        parent_state: dict[str, torch.Tensor] = {}
        if seed_state_path and seed_state_path.exists():
            parent_state = torch.load(
                seed_state_path, map_location=self.device
            )  # nosec B614 - checkpoints produced locally
            model.load_state_dict(parent_state, strict=False)
            match_experts_to_parent(model, parent_state)
        optimizer = build_optimizer(model.parameters(), spec.train)
        criterion = nn.CrossEntropyLoss()
        start_time = time.perf_counter()
        tokens_seen = 0
        iterator = iter(batch_iter)
        stop_reason = ""
        best_loss = float("inf")
        no_improve = 0
        entropy_bad = 0
        nan_or_inf = False
        max_loss_jump = 0.0
        optimizer.zero_grad()
        total_steps = max(1, self.steps)
        for step_idx in range(self.steps):
            if spec.model.recurrences:
                model.set_recurrence_steps(
                    self._recurrence_schedule(spec, step_idx, total_steps)
                )
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batch_iter)
                batch = next(iterator)
            input_ids = batch.input_ids.to(self.device)
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            # Auxiliary MoE routing losses
            aux_loss = torch.tensor(0.0, device=self.device)
            lb_coeff = float(spec.train.router_lb_coeff)
            ent_coeff = float(spec.train.router_entropy_coeff)
            if lb_coeff > 0.0 or ent_coeff > 0.0:
                for mod in model.modules():
                    if isinstance(mod, MoELayer):
                        if ent_coeff > 0.0 and hasattr(mod, "last_entropy"):
                            aux_loss = aux_loss + (-ent_coeff) * mod.last_entropy
                        if lb_coeff > 0.0 and hasattr(mod, "last_lb"):
                            aux_loss = aux_loss + lb_coeff * mod.last_lb
            loss = loss + aux_loss
            loss.backward()
            grad_total = clip_grad_norm_(model.parameters(), spec.train.clip)
            if hasattr(grad_total, "item"):
                grad_norm = float(grad_total.item())
            else:
                grad_norm = float(grad_total)
            if not math.isfinite(grad_norm):
                nan_or_inf = True
            if grad_norm > self.instability_threshold:
                stop_reason = f"high_grad({grad_norm:.2f})"
                optimizer.zero_grad()
                break
            optimizer.step()
            optimizer.zero_grad()
            tokens_seen += input_ids.numel()
            # Router entropy guard
            step_entropy = _average_router_entropy(model)
            if step_entropy is not None and step_entropy < self.entropy_threshold:
                entropy_bad += 1
                if entropy_bad >= self.entropy_patience:
                    stop_reason = f"low_entropy({step_entropy:.2f})"
                    break
            else:
                entropy_bad = 0
            current_loss = float(loss.item())
            if not math.isfinite(current_loss):
                nan_or_inf = True
            if best_loss < float("inf"):
                jump = current_loss - best_loss
                if jump > max_loss_jump:
                    max_loss_jump = jump
            if current_loss + self.improvement_tolerance < best_loss:
                best_loss = current_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.no_improve_patience:
                    stop_reason = "no_improve"
                    break
        duration = max(time.perf_counter() - start_time, 1e-6)
        throughput = tokens_seen / duration
        model.set_recurrence_steps(
            self._recurrence_schedule(spec, self.steps, total_steps)
        )
        perplexity = self._evaluate_perplexity(model, spec, batch_iter, criterion)
        checkpoint_path = self.checkpoint_dir / f"{candidate.ident}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        # Aggregate router telemetry
        router_entropy = 0.0
        router_lb = 0.0
        router_load_max = 0.0
        router_load_min = 1.0
        max_grad_norm = 0.0
        try:
            if hasattr(grad_total, "item"):
                max_grad_norm = float(grad_total.item())
            else:
                max_grad_norm = float(grad_total)
        except Exception:
            max_grad_norm = 0.0
        count = 0
        with torch.no_grad():
            for mod in model.modules():
                if isinstance(mod, MoELayer):
                    if hasattr(mod, "last_entropy"):
                        router_entropy += float(mod.last_entropy.item())
                    if hasattr(mod, "last_lb"):
                        router_lb += float(mod.last_lb.item())
                    if hasattr(mod, "last_load"):
                        load = getattr(mod, "last_load")
                        try:
                            max_val = float(load.max().item())
                            min_val = float(load.min().item())
                            router_load_max = max(router_load_max, max_val)
                            router_load_min = min(router_load_min, min_val)
                        except Exception:
                            pass
                    count += 1
        if count:
            router_entropy /= count
            router_lb /= count
        else:
            router_load_min = 0.0
        reason_code = 0.0
        if stop_reason.startswith("high_grad"):
            reason_code = 1.0
        elif stop_reason.startswith("low_entropy"):
            reason_code = 2.0
        elif stop_reason == "no_improve":
            reason_code = 3.0
        metrics = {
            "ppl_code": perplexity,
            "ppl_math": perplexity * (1.0 + 0.01 * spec.model.moe_block_count()),
            "throughput": throughput,
            "params": float(count_parameters(model)),
            "ram": float(count_parameters(model) * 2 / (1024**3)),
            "long_recall": _estimate_long_recall(spec),
            "router_entropy": router_entropy,
            "router_lb": router_lb,
            "router_load_max": router_load_max,
            "router_load_min": router_load_min,
            "max_grad_norm": max_grad_norm,
            "instability": max_grad_norm,
            "stop_reason_code": reason_code,
            "nan_seen": 1.0 if nan_or_inf else 0.0,
            "loss_spike": max(0.0, max_loss_jump),
        }
        if spec.model.recurrences:
            metrics.update(self._recurrence_evaluations(model, spec, batch_iter, criterion))
        return metrics, checkpoint_path

    def _recurrence_schedule(
        self, spec: ArchitectureSpec, step_idx: int, total_steps: int
    ) -> dict[int, int]:
        schedule: dict[int, int] = {}
        if not spec.model.recurrences:
            return schedule
        progress = step_idx / max(1, total_steps)
        for idx, cfg in enumerate(spec.model.recurrences):
            base = max(1, cfg.train_recurrence)
            target = max(1, cfg.max_train_recurrence)
            frac = cfg.curriculum_fraction
            if frac > 0 and progress < frac:
                ratio = progress / frac
                steps = int(round(base + (target - base) * ratio))
            else:
                steps = target
            schedule[idx] = max(1, steps)
        return schedule

    def _evaluate_perplexity(
        self,
        model: nn.Module,
        spec: ArchitectureSpec,
        batch_iter: Iterable[TokenBatch],
        criterion: nn.Module,
    ) -> float:
        eval_loss = 0.0
        eval_batches = 0
        with torch.no_grad():
            iterator = iter(batch_iter)
            for _ in range(self.eval_batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                input_ids = batch.input_ids.to(self.device)
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
                eval_loss += loss.item()
                eval_batches += 1
        if eval_batches == 0:
            return 1.0
        return float(torch.exp(torch.tensor(eval_loss / eval_batches)).item())

    def _recurrence_evaluations(
        self,
        model: nn.Module,
        spec: ArchitectureSpec,
        batch_iter: Iterable[TokenBatch],
        criterion: nn.Module,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if not spec.model.recurrences:
            return metrics
        rec_values = spec.model.recurrences[0].test_recurrences or [1]
        rec_values = sorted({max(1, v) for v in rec_values})
        base_value = rec_values[0]
        best_value = rec_values[-1]
        base_ppl = None
        best_ppl = None
        for value in rec_values:
            steps = dict.fromkeys(range(len(spec.model.recurrences)), value)
            model.set_recurrence_steps(steps)
            ppl = self._evaluate_perplexity(model, spec, batch_iter, criterion)
            metrics[f"ppl_code_rec_{value}"] = ppl
            if value == base_value:
                base_ppl = ppl
            if value == best_value:
                best_ppl = ppl
        if base_ppl is not None and best_ppl is not None:
            metrics["recurrence_gain"] = float(base_ppl - best_ppl)
        return metrics


def _average_router_entropy(model: nn.Module) -> float | None:
    with torch.no_grad():
        entropies = []
        for mod in model.modules():
            if isinstance(mod, MoELayer) and hasattr(mod, "last_entropy"):
                entropies.append(float(mod.last_entropy.item()))
    if not entropies:
        return None
    return sum(entropies) / len(entropies)


def _estimate_long_recall(spec: ArchitectureSpec) -> float:
    retro = 0
    for block in spec.model.blocks:
        retro += sum(1 for extra in block.extras if getattr(extra, "type", None) == "retro")
    return retro / max(1, spec.model.n_layers)
