from __future__ import annotations

from unittest.mock import patch

import torch
from torch import Tensor, nn

from transformer_evolution_llm.dsl import AttentionConfig
from transformer_evolution_llm.models import MultiHeadSelfAttention


class CaptureModule(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.last_input: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x
        b, t, _ = x.shape
        return torch.zeros(b, t, self.out_features)


def test_qk_norm_max_clamps_queries() -> None:
    cfg = AttentionConfig(heads=2, head_dim=4, qk_norm_max=1.0)
    dim = cfg.heads * cfg.head_dim
    module = MultiHeadSelfAttention(cfg, dim)

    x = torch.randn(2, 4, dim) * 10.0

    capture_mod = CaptureModule(24)
    module.c_attn = capture_mod
    module.c_proj = CaptureModule(dim)

    with patch(
        "torch.nn.functional.scaled_dot_product_attention", return_value=torch.zeros(2, 2, 4, 4)
    ):
        _ = module(x)

    assert capture_mod.last_input is not None
    norms = capture_mod.last_input.norm(dim=-1)
    assert torch.all(norms <= 1.0 + 1e-4)


def test_sliding_sparsity_builds_local_window_mask() -> None:
    cfg = AttentionConfig(heads=2, head_dim=4, sw=1, sparsity="sliding")
    dim = cfg.heads * cfg.head_dim
    module = MultiHeadSelfAttention(cfg, dim)

    t = 5
    x = torch.randn(1, t, dim)

    with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
        mock_sdpa.return_value = torch.zeros(1, 2, t, 4)
        _ = module(x)

        args, kwargs = mock_sdpa.call_args
        mask = kwargs.get("attn_mask")
        assert mask is not None
        assert mask.shape == (t, t)

        for i in range(t):
            lo = max(0, i - 1)
            hi = min(t, i + 2)
            window = mask[i, lo:hi]
            assert torch.all(window == 0.0)
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

    t = 6
    x = torch.randn(1, t, dim)

    with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
        mock_sdpa.return_value = torch.zeros(1, 2, t, 4)
        _ = module(x)

        mask = mock_sdpa.call_args.kwargs.get("attn_mask")
        assert mask is not None
        assert mask.shape == (t, t)
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

    t = 6
    x = torch.randn(1, t, dim)

    with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
        mock_sdpa.return_value = torch.zeros(1, 2, t, 4)
        _ = module(x)

        mask = mock_sdpa.call_args.kwargs.get("attn_mask")
        assert mask is not None

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


def test_gated_attention_logic() -> None:
    """Verify that gating parameters are created and used for variants."""
    # Variant 1: Output gating, Dense
    cfg = AttentionConfig(heads=2, head_dim=4, gating_pos="output", gating_op="dense")
    dim = cfg.heads * cfg.head_dim
    module = MultiHeadSelfAttention(cfg, dim)

    assert hasattr(module, "gate_weight")
    assert module.gate_weight.shape == (2, 4, 4)
    assert module.gate_bias.shape == (2, 4)

    x = torch.randn(1, 4, dim)
    out = module(x)
    assert out.shape == (1, 4, dim)

    # Variant 2: Value gating, Diagonal
    cfg = AttentionConfig(heads=2, head_dim=4, gating_pos="value", gating_op="diagonal")
    dim = cfg.heads * cfg.head_dim
    module = MultiHeadSelfAttention(cfg, dim)

    assert hasattr(module, "gate_weight")
    assert module.gate_weight.shape == (2, 4)  # Diagonal
    assert module.gate_bias.shape == (2, 4)

    x = torch.randn(1, 4, dim)
    out = module(x)
    assert out.shape == (1, 4, dim)
