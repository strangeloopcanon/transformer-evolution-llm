# Status & Architecture Log

This document tracks the results and architectural highlights from evolutionary runs.

## Architecture Highlights (Current Frontier)

| Theme | What we learned | Where to look |
| --- | --- | --- |
| **Triple-retro loops** | Best single-block models carry three separate retro rails (short gated, mid gated, long attention) feeding the same residual; this pushes long_recall to 3 without breaking latency. | `runs/frontier_phi_loop_long.json` (`mutate_topk-30-bbb0`, `insert_custom_module-37-a0d0`) |
| **MoE + SSM hybrids** | Weighted runs keep 5–6 block stacks with dual MoE cores and mamba SSMs, but they cost ~0.5k tok/s. Still, they reach ppl ≈1.87 which beats most laptop-friendly configs. | `runs/frontier_phi_multibranch_weighted.json` (`xover-3-85e3`, `xover-13-6e49`) |
| **Checkpoint pruning works** | Old checkpoints are removed automatically when the pool ages out and again when a run completes, so long sweeps no longer eat the disk. | See log tails (`Removed old checkpoint dirs: …`) |
| **Composite objective + throughput** | Adding `ppl_per_long_recall` to the config plus a higher instability threshold surfaced non-degenerate 14-block hybrids and finally recorded throughput (~1.7 k tok/s) in `runs/frontier_phi_creative_canon.json`. | `runs/frontier_phi_creative_canon.json` (`mutate_topk-24-6bee`, `make_gqa-34-18a0`) |

---

## Evolution Log (Chronological)

| Run label | Frontier file | #candidates | Top candidate | ppl_code ↓ |
| --- | --- | --- | --- | --- |
| Nov 7 baseline | `runs/frontier_phi_live.json` | 4 | `make_gqa-23-6a85` | 1.81 |
| Nov 8 loop probe | `runs/frontier_phi_feedback.json` | 12 | `dense_to_moe-32-b1d4` | 7.86 |
| Nov 8 aggressive | `runs/frontier_phi_live_v2.json` | 2 | `xover-38-be45` | 1.33 |
| Nov 9 loop long | `runs/frontier_phi_loop_long.json` | 12 | `xover-58-d356` | 2.11 |
| Nov 10 multibranch | `runs/frontier_phi_multibranch.json` | 14 | `xover-57-aea3` | 1.78 |
| Nov 11 weighted multibranch | `runs/frontier_phi_multibranch_weighted.json` | 35 | `mutate_topk-10-4eca` | 1.78 |
| Nov 15 canonical creative (composite) | `runs/frontier_phi_creative_canon.json` | 24 | `mutate_topk-24-6bee` | 2.26×10⁴ |

---

<details>
<summary>Nov 7 – Baseline sweep</summary>

- Config `configs/live_phi_tiny.yaml` with 4 blocks (dense + MoE + SSM + MoE).
- Frontier highlights: `make_gqa-23-6a85` (best ppl/recall) vs. `toggle_gated_mix-24-2938` (faster but less recall).
- Mermaid (seed architecture):
  ```mermaid
  graph TD
    Token --> B1["Block1: GQA + Dense + Retro"]
    B1 --> B2["Block2: GQA + MoE + Custom mixer"]
    B2 --> B3["Block3: GQA + Dense + Mamba2"]
    B3 --> B4["Block4: GQA + MoE + Gated mixer"]
    B4 --> Head
  ```
</details>

<details>
<summary>Nov 8 – Loop probe (36 gens · 180 steps)</summary>

- Added feedback templates (`inject_feedback_loop`, `insert_feedback_block`).
- Frontier (12 entries) shows first self-mutated feedback block `template_mutation-7-7aad`.
- Key diagram (loop block):
  ```mermaid
  graph TD
    Attn --> Feedback["Feedback gate"]
    Feedback --> Retro["Retro gate"]
    Retro --> FFN
  ```
</details>

<details>
<summary>Nov 8 – Aggressive 48×200 run</summary>

- Config `configs/live_phi_tiny.yaml`, 48 gens, 200 steps.
- Winners collapsed to single block with dual retro memories (`xover-38-be45`, `make_gqa-22-fa88`).
</details>

<details>
<summary>Nov 9 – Loop seed (60×240, `live_phi_feedback_seed.yaml`)</summary>

- Seeded from `xover-38-be45`.
- Produced `mutate_topk-30-bbb0` (triple retro rails) and `insert_custom_module-37-a0d0` (adds trainable gate) at ppl ≈2.14 / throughput ≈2.6k tok/s.
</details>

<details>
<summary>Nov 10 – Multibranch config (72×260, no weighting)</summary>

- Config `configs/live_phi_multiblock.yaml` with long chain of dense + MoE + SSM + feedback blocks.
- Frontier (14 entries) includes `xover-57-aea3` (5-block chain with retro caches on the first two stages).
</details>

<details>
<summary>Nov 11 – Weighted multibranch run</summary>

- Same config as Nov 10 but with score weights favoring ppl/layers/MoE.
- Frontier (35 entries) demonstrates trade-off between deep hybrids (`xover-3-85e3`) and fast baselines (`mutate_topk-10-4eca`).
</details>
