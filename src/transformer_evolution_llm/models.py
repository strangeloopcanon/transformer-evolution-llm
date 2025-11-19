"""Torch modules assembled from the DSL for live training."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from .dsl import (
    AttentionConfig,
    BlockConfig,
    CustomModuleConfig,
    GatedModuleConfig,
    ModelConfig,
    MoEFFNConfig,
    RecurrenceConfig,
    RetroConfig,
    SSMConfig,
)
from .plugins import get_component


def _swiglu(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return F.silu(x1) * x2


ActivationLookup: dict[str, Callable[[Tensor], Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "swiglu": _swiglu,
}


class RotaryPositionalEncoding(nn.Module):
    """Minimal RoPE implementation for experimentation."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.base = base

    def forward(self, seq_len: int, device: torch.device) -> Tensor:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

    @staticmethod
    def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
        # x: (B, T, H, D)
        seq_len = x.size(1)
        rope = rope[:seq_len, :]
        rot_dim = min(rope.size(-1), x.size(-1))
        cos = rope.cos()[None, :, None, :rot_dim]
        sin = rope.sin()[None, :, None, :rot_dim]
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        x1, x2 = x_rot[..., ::2], x_rot[..., 1::2]
        cos_part = cos[..., ::2]
        sin_part = sin[..., ::2]
        rotated = torch.stack(
            (x1 * cos_part - x2 * sin_part, x1 * sin_part + x2 * cos_part),
            dim=-1,
        ).flatten(-2)
        if x_pass.numel():
            rotated = torch.cat([rotated, x_pass], dim=-1)
        return rotated


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig, dim: int):
        super().__init__()
        self.cfg = cfg
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=cfg.heads,
            batch_first=True,
        )
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.rope = (
            RotaryPositionalEncoding(cfg.head_dim, float(cfg.rope_theta or 10000.0))
            if cfg.rope
            else None
        )
        self._impl_logged = False

    def forward(self, x: Tensor) -> Tensor:
        # Optional input norm clamp to stabilize Q/K magnitudes
        if getattr(self.cfg, "qk_norm_max", None):
            max_norm = float(self.cfg.qk_norm_max)  # type: ignore[arg-type]
            eps = 1e-6
            norms = x.norm(dim=-1, keepdim=True).clamp_min(eps)
            scale = (max_norm / norms).clamp(max=1.0)
            x = x * scale
        if self.rope is not None:
            b, t, d = x.shape
            x_reshaped = x.view(b, t, self.heads, self.head_dim)
            rope = self.rope(t, x.device)
            x_rot = RotaryPositionalEncoding.apply_rotary(x_reshaped, rope)
            x = x_rot.view(b, t, d)
        attn_mask = None
        t = x.size(1)
        sparsity = getattr(self.cfg, "sparsity", "none")

        def ensure_mask() -> torch.Tensor:
            nonlocal attn_mask
            if attn_mask is None:
                attn_mask = torch.full((t, t), float("-inf"), device=x.device)
            return attn_mask

        if sparsity == "local_block":
            mask = ensure_mask()
            w = int(self.cfg.sw or self.head_dim)
            for i in range(t):
                lo = max(0, i - w)
                hi = min(t, i + w + 1)
                mask[i, lo:hi] = 0.0
            bsz = int(self.cfg.block_size or w)
            stride = int(getattr(self.cfg, "block_stride", bsz))
            for start in range(0, t, max(1, stride)):
                end = min(t, start + bsz)
                mask[:, start:end] = 0.0
        elif sparsity == "local_global":
            mask = ensure_mask()
            w = int(self.cfg.sw or self.head_dim)
            gstride = int(getattr(self.cfg, "global_stride", 0) or 0)
            for i in range(t):
                lo = max(0, i - w)
                hi = min(t, i + w + 1)
                mask[i, lo:hi] = 0.0
            if gstride and gstride > 0:
                global_idx = torch.arange(0, t, gstride, device=x.device)
                mask[:, global_idx] = 0.0
            mask[:, 0] = 0.0
        elif sparsity == "block" and getattr(self.cfg, "block_size", None):
            mask = ensure_mask()
            bsz = int(self.cfg.block_size or 0)
            stride = int(getattr(self.cfg, "block_stride", self.cfg.block_size or bsz))
            for start in range(0, t, max(1, stride)):
                end = min(t, start + bsz)
                mask[start:end, start:end] = 0.0
        elif sparsity == "dilated" and getattr(self.cfg, "dilation", None):
            mask = ensure_mask()
            dilation = max(1, int(self.cfg.dilation or 1))
            for offset in range(min(dilation, t)):
                idx = torch.arange(offset, t, dilation, device=x.device)
                mask[idx.unsqueeze(0), idx.unsqueeze(1)] = 0.0
        else:
            sliding_active = sparsity == "sliding" or (
                sparsity == "none" and getattr(self.cfg, "sw", None)
            )
            if sliding_active:
                mask = ensure_mask()
                w = int(self.cfg.sw or self.head_dim)
                for i in range(t):
                    lo = max(0, i - w)
                    hi = min(t, i + w + 1)
                    mask[i, lo:hi] = 0.0
        # Prefer CUDA SDPA kernels when available (capability, not a search knob)
        if torch.backends.cuda.is_built() and x.is_cuda:
            try:
                ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_mem_efficient=True, enable_math=True
                )
                if not self._impl_logged:
                    print("[attention] Using CUDA SDPA kernels (flash/mem-efficient enabled)")
                    self._impl_logged = True
                with ctx:
                    out, _ = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)
            except Exception:
                out, _ = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)
        else:
            out, _ = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)
        return cast(Tensor, out)


class DenseFFN(nn.Module):
    def __init__(self, dim: int, hidden: int, activation: str):
        super().__init__()
        inner = hidden * 2 if activation == "swiglu" else hidden
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner if activation != "swiglu" else hidden, dim)
        self.activation_name = activation
        self.activation: Callable[[Tensor], Tensor] = ActivationLookup.get(activation) or F.silu

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.activation(out)
        return cast(Tensor, self.fc2(out))


class Expert(nn.Module):
    def __init__(self, dim: int, hidden: int, activation: str, hops: int = 1):
        super().__init__()
        self.net = DenseFFN(dim, hidden, activation)
        self.hops = max(1, hops)

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.hops):
            x = cast(Tensor, self.net(x))
        return x


class MoELayer(nn.Module):
    def __init__(self, dim: int, cfg: MoEFFNConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(dim, cfg.n_experts)
        self.experts = nn.ModuleList()
        if cfg.experts:
            for idx in range(cfg.n_experts):
                # If fewer expert configs than n_experts, repeat the last one.
                if idx < len(cfg.experts):
                    ecfg = cfg.experts[idx]
                else:
                    ecfg = cfg.experts[-1]
                etype = getattr(ecfg, "type", "dense")
                if etype == "dense":
                    hidden = ecfg.hidden or cfg.hidden  # type: ignore[union-attr]
                    hops = getattr(ecfg, "hops", 1)
                    self.experts.append(Expert(dim, hidden, activation=ecfg.activation, hops=hops))  # type: ignore[arg-type]
                elif etype == "ssm":
                    # Wrap SSMLayer as an expert, possibly with multiple hops.
                    hops = getattr(ecfg, "hops", 1)
                    ssm_layer = SSMLayer(ecfg.ssm, dim)  # type: ignore[union-attr]

                    class _SSMExpert(nn.Module):
                        def __init__(self, layer: SSMLayer, num_hops: int) -> None:
                            super().__init__()
                            self.layer = layer
                            self.hops = max(1, num_hops)

                        def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
                            for _ in range(self.hops):
                                x = self.layer(x)
                            return x

                    self.experts.append(_SSMExpert(ssm_layer, hops))
                elif etype == "custom":
                    # Use CustomModule to build a custom expert if possible.
                    custom_cfg = CustomModuleConfig(name=ecfg.name, params=ecfg.params)  # type: ignore[union-attr]
                    self.experts.append(CustomModule(custom_cfg, dim))
                else:
                    # Fallback to a standard dense expert.
                    self.experts.append(Expert(dim, cfg.hidden, activation="swiglu"))
        else:
            self.experts = nn.ModuleList(
                Expert(dim, cfg.hidden, activation="swiglu") for _ in range(cfg.n_experts)
            )

    def forward(self, x: Tensor) -> Tensor:
        temp = float(self.cfg.router_temperature) if self.cfg.router_temperature else 1.0
        logits = self.router(x) / temp
        topk_val, topk_idx = torch.topk(logits, k=self.cfg.k, dim=-1)
        weights = torch.softmax(topk_val, dim=-1)
        outputs = torch.zeros_like(x)
        # Track simple routing stats for aux losses/metrics
        # Entropy over top-k weights (normalized to [0,1] by log(k))
        eps = 1e-8
        entropy = -(weights * (weights + eps).log()).sum(dim=-1).mean()
        self.last_entropy = entropy
        # Load-balance proxy: selection frequency deviation from uniform
        with torch.no_grad():
            b, t, _ = x.shape
            total = max(1, b * t * self.cfg.k)
            counts = torch.zeros(self.cfg.n_experts, device=x.device)
            for expert_pos in range(self.cfg.k):
                idx = topk_idx[..., expert_pos].reshape(-1)
                counts += torch.bincount(idx, minlength=self.cfg.n_experts).float()
            freq = counts / total
            uniform = 1.0 / self.cfg.n_experts
            lb = ((freq - uniform) ** 2).mean()
        self.last_lb = lb
        # Persist the last routing frequency histogram for tooling/metrics.
        self.last_load = freq
        for expert_pos in range(self.cfg.k):
            idx = topk_idx[..., expert_pos]
            weight = weights[..., expert_pos].unsqueeze(-1)
            expert_outputs = torch.zeros_like(x)
            for expert_id in range(self.cfg.n_experts):
                mask = idx == expert_id
                if mask.any():
                    expert_out = self.experts[expert_id](x[mask])
                    expert_outputs = expert_outputs.index_put((mask,), expert_out)
            outputs = outputs + expert_outputs * weight
        return outputs

    def sort_experts(self) -> None:
        norms = [expert.net.fc1.weight.norm().item() for expert in self.experts]
        permutation = sorted(range(len(norms)), key=lambda i: norms[i], reverse=True)
        self.reorder(permutation)

    def reorder(self, permutation: list[int]) -> None:
        self.experts = nn.ModuleList([self.experts[i] for i in permutation])
        with torch.no_grad():
            self.router.weight[:] = self.router.weight[permutation]
            self.router.bias[:] = self.router.bias[permutation]


class SSMLayer(nn.Module):
    def __init__(self, cfg: SSMConfig, dim: int):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=cfg.d_conv,
            padding="same",
            groups=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        # (B, T, D) -> (B, D, T)
        seq = self.conv(x.transpose(1, 2))
        return cast(Tensor, seq.transpose(1, 2))


class RetroModule(nn.Module):
    def __init__(self, cfg: RetroConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: Tensor) -> Tensor:
        window = min(self.cfg.memory_tokens, x.shape[1])
        cumsum = torch.cumsum(x, dim=1)
        padding = torch.zeros_like(x[:, :window])
        shifted = torch.cat([padding, cumsum[:, :-window]], dim=1)
        avg = (cumsum - shifted) / window
        return avg * self.cfg.gating_weight


class CustomModule(nn.Module):
    def __init__(self, cfg: CustomModuleConfig, dim: int):
        super().__init__()
        inner = cfg.params.get("dim", dim)
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.SiLU(),
            nn.Linear(inner, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))


class GatedModule(nn.Module):
    def __init__(self, cfg: GatedModuleConfig):
        super().__init__()
        weight = torch.tensor(cfg.init_weight, dtype=torch.float32)
        if cfg.learnable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight


class EvolutionBlock(nn.Module):
    def __init__(self, dim: int, cfg: BlockConfig, norm_type: str = "layernorm"):
        super().__init__()
        self.attn = MultiHeadSelfAttention(cfg.attn, dim) if cfg.attn else None
        if cfg.ffn is None:
            self.ffn: DenseFFN | MoELayer | None = None
        elif cfg.ffn.type == "moe":
            if not isinstance(cfg.ffn, MoEFFNConfig):
                msg = "MoE block requires MoEFFNConfig."
                raise TypeError(msg)
            self.ffn = MoELayer(dim, cfg.ffn)
        else:
            self.ffn = DenseFFN(dim, cfg.ffn.hidden, getattr(cfg.ffn, "activation", "silu"))
        self.ssm = SSMLayer(cfg.ssm, dim) if cfg.ssm else None
        self.norm = _norm_layer(norm_type, dim)
        self.extras = nn.ModuleList()
        for extra in cfg.extras:
            if isinstance(extra, RetroConfig):
                self.extras.append(RetroModule(extra))
            elif isinstance(extra, GatedModuleConfig):
                self.extras.append(GatedModule(extra))
            elif isinstance(extra, CustomModuleConfig):
                builder = get_component(extra.name)
                if builder is not None:
                    self.extras.append(builder(extra, dim))
                else:
                    self.extras.append(CustomModule(extra, dim))

    def forward(self, x: Tensor) -> Tensor:
        if self.attn:
            x = x + self.attn(self.norm(x))
        if self.ssm:
            x = x + self.ssm(self.norm(x))
        if self.ffn:
            x = x + self.ffn(self.norm(x))
        for extra in self.extras:
            x = x + extra(self.norm(x))  # nosec B610 - pure tensor gating, no SQL context
        return x


class EvolutionModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        vocab = cfg.emb.vocab or cfg.head.vocab
        if vocab is None:
            raise ValueError("Vocabulary must be specified in emb or head config.")
        self.embed = nn.Embedding(vocab, cfg.emb.dim)
        self.blocks = nn.ModuleList(
            [EvolutionBlock(cfg.emb.dim, block, norm_type=cfg.norm) for block in cfg.blocks]
        )
        self.norm = _norm_layer(cfg.norm, cfg.emb.dim)
        self.lm_head = nn.Linear(cfg.emb.dim, cfg.head.vocab)
        # Recurrence setup
        self._recurrence_order: list[tuple[int, RecurrenceConfig]] = sorted(
            enumerate(cfg.recurrences), key=lambda item: item[1].start
        )
        self.recurrence_adapters = nn.ModuleList()
        self.recurrence_steps: dict[int, int] = {}
        for idx, rec_cfg in self._recurrence_order:
            concat_dim = cfg.emb.dim * 2 if rec_cfg.concat_prelude else cfg.emb.dim
            adapter = RecurrenceAdapter(
                input_dim=concat_dim,
                dim=cfg.emb.dim,
                kind=rec_cfg.adapter,
            )
            self.recurrence_adapters.append(adapter)
            self.recurrence_steps[idx] = rec_cfg.train_recurrence

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        if not self._recurrence_order:
            for block in self.blocks:
                x = block(x)
        else:
            idx = 0
            order_pos = 0
            while idx < len(self.blocks):
                if (
                    order_pos < len(self._recurrence_order)
                    and idx == self._recurrence_order[order_pos][1].start
                ):
                    spec_idx, rec_cfg = self._recurrence_order[order_pos]
                    x = self._run_recurrence(spec_idx, rec_cfg, x)
                    idx = rec_cfg.end
                    order_pos += 1
                    continue
                x = self.blocks[idx](x)
                idx += 1
        x = self.norm(x)
        return cast(Tensor, self.lm_head(x))

    def set_recurrence_steps(self, steps: dict[int, int]) -> None:
        for idx, value in steps.items():
            self.recurrence_steps[idx] = max(1, int(value))

    def _run_recurrence(
        self,
        spec_idx: int,
        rec_cfg: RecurrenceConfig,
        prelude_output: Tensor,
    ) -> Tensor:
        # Determine current loop count
        end_idx = min(rec_cfg.end, len(self.blocks))
        start_idx = min(rec_cfg.start, max(0, end_idx - 1))
        if end_idx - start_idx <= 1:
            return prelude_output
        current_steps = max(1, self.recurrence_steps.get(spec_idx, rec_cfg.train_recurrence))
        state = self._init_recurrence_state(rec_cfg, prelude_output)
        adapter = self.recurrence_adapters[self._adapter_position(spec_idx)]
        for _ in range(current_steps):
            h = state
            for block_idx in range(start_idx, end_idx):
                h = self.blocks[block_idx](h)
            adapter_input = (
                torch.cat([prelude_output, h], dim=-1) if rec_cfg.concat_prelude else h
            )
            state = adapter(adapter_input, h)
        return state

    def _init_recurrence_state(self, cfg: RecurrenceConfig, reference: Tensor) -> Tensor:
        if cfg.init_state == "noise":
            return cast(Tensor, torch.randn_like(reference) * cfg.noise_std)
        return cast(Tensor, torch.zeros_like(reference))

    def _adapter_position(self, spec_idx: int) -> int:
        for pos, (cfg_idx, _) in enumerate(self._recurrence_order):
            if cfg_idx == spec_idx:
                return pos
        return 0


class RecurrenceAdapter(nn.Module):
    def __init__(self, input_dim: int, dim: int, kind: str = "linear"):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.proj = nn.Linear(input_dim, dim)
        elif kind == "gated":
            self.val = nn.Linear(input_dim, dim)
            self.gate = nn.Linear(input_dim, dim)
        else:
            raise ValueError(f"Unsupported adapter kind '{kind}'")

    def forward(self, adapter_input: Tensor, residual: Tensor) -> Tensor:
        if self.kind == "linear":
            return cast(Tensor, self.proj(adapter_input))
        gate = torch.sigmoid(self.gate(adapter_input))
        value = torch.tanh(self.val(adapter_input))
        return cast(Tensor, gate * value + (1 - gate) * residual)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def _norm_layer(norm_type: str, dim: int) -> nn.Module:
    if norm_type.lower() == "rmsnorm":
        return RMSNorm(dim)
    return nn.LayerNorm(dim)
