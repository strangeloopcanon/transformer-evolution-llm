from __future__ import annotations

from typing import cast

import torch
from torch import nn, Tensor

from transformer_evolution_llm.dsl import AttentionConfig
from transformer_evolution_llm.models import MultiHeadSelfAttention


class _DummyMHA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_q: Tensor | None = None
        self.last_mask: Tensor | None = None

    def forward(  # type: ignore[override]
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        del key, value, need_weights
        self.last_q = query
        if attn_mask is not None:
            self.last_mask = attn_mask
        return query, None


def test_qk_norm_max_clamps_queries() -> None:
    cfg = AttentionConfig(heads=2, head_dim=4, qk_norm_max=1.0)
    dim = cfg.heads * cfg.head_dim
    module = MultiHeadSelfAttention(cfg, dim)
    # Replace underlying attention with a dummy to inspect inputs.
    dummy = _DummyMHA()
    module.attn = cast(nn.MultiheadAttention, dummy)  # type: ignore[assignment]
    x = torch.randn(2, 4, dim) * 10.0
    _ = module(x)
    assert dummy.last_q is not None
    norms = dummy.last_q.norm(dim=-1)
    assert torch.all(norms <= 1.0 + 1e-4)


def test_sliding_sparsity_builds_local_window_mask() -> None:
    cfg = AttentionConfig(heads=2, head_dim=4, sw=1, sparsity="sliding")
    dim = cfg.heads * cfg.head_dim
    module = MultiHeadSelfAttention(cfg, dim)
    dummy = _DummyMHA()
    module.attn = cast(nn.MultiheadAttention, dummy)  # type: ignore[assignment]
    t = 5
    x = torch.randn(1, t, dim)
    _ = module(x)
    assert dummy.last_mask is not None
    mask = dummy.last_mask
    assert mask.shape == (t, t)
    for i in range(t):
        lo = max(0, i - 1)
        hi = min(t, i + 2)
        # Positions inside the window should be unmasked (0.0).
        window = mask[i, lo:hi]
        assert torch.all(window == 0.0)
        # Positions outside should be masked (-inf).
        if lo > 0:
            assert torch.all(torch.isneginf(mask[i, :lo]))
        if hi < t:
            assert torch.all(torch.isneginf(mask[i, hi:]))


def test_block_sparsity_masks_cross_block_attention() -> None:
    cfg = AttentionConfig(
        heads=2,
        head_dim=4,
        sparsity="block",
        block_size=2,
        block_stride=2,
    )
    dim = cfg.heads * cfg.head_dim
    module = MultiHeadSelfAttention(cfg, dim)
    dummy = _DummyMHA()
    module.attn = cast(nn.MultiheadAttention, dummy)  # type: ignore[assignment]
    t = 6
    x = torch.randn(1, t, dim)
    _ = module(x)
    assert dummy.last_mask is not None
    mask = dummy.last_mask
    assert mask.shape == (t, t)
    # Tokens 0-1, 2-3, 4-5 form separate blocks.
    # Check that token 0 cannot attend to token 2 (masked) but can attend to token 1.
    assert mask[0, 1] == 0.0
    assert torch.isneginf(mask[0, 2])


def test_dilated_sparsity_masks_every_other_token() -> None:
    cfg = AttentionConfig(
        heads=2,
        head_dim=4,
        sparsity="dilated",
        dilation=2,
    )
    dim = cfg.heads * cfg.head_dim
    module = MultiHeadSelfAttention(cfg, dim)
    dummy = _DummyMHA()
    module.attn = cast(nn.MultiheadAttention, dummy)  # type: ignore[assignment]
    t = 6
    x = torch.randn(1, t, dim)
    _ = module(x)
    assert dummy.last_mask is not None
    mask = dummy.last_mask
    # Token 0 attends only to even positions; token 1 to odd positions.
    even_positions = [0, 2, 4]
    odd_positions = [1, 3, 5]
    for j in even_positions:
        assert mask[0, j] == 0.0
    for j in odd_positions:
        assert torch.isneginf(mask[0, j])
    for j in odd_positions:
        assert mask[1, j] == 0.0
    for j in even_positions:
        assert torch.isneginf(mask[1, j])
