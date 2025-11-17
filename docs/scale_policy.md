# Scale Policy (Laptop → Scale-Hop)

This doc captures the soft priors we use while exploring architectures on laptop-scale surrogates (~100 M parameters, 200–300 steps) and how to adapt them for a scale-hop sanity run (350 M–1 B) before committing to 1.5–8 B training.

## Priors encoded in the DSL

| Knob | Default | Rationale |
| ---- | ------- | --------- |
| `model.norm` | `layernorm` (RMSNorm optional) | LayerNorm is the baseline; RMSNorm often stabilizes larger stacks. |
| `attn.head_dim` | 64 | Keeps QK compute efficient and portable; matches most modern LLMs. |
| `ffn.hidden` | `4 × d_model` | Standard transformer width multiplier. |
| `attn.kv_groups` | 1–2 | KV compression halves KV cache without hurting quality on these surrogates. |
| `attn.sparsity` | `local_global` default window `≈ sqrt(seq_len) × window_scale` | Balances local detail and global sentinels. |
| `attn.global_stride` | `≈ sqrt(seq_len)` | One global token per √L positions keeps receptive field broad. |
| `attn.rope_theta` | 10000 (jitter allowed) | Base RoPE value; slight jitter explores length generalization. |
| `train.priors.tokens_per_param` | 4 | Roughly 4 tokens per parameter per candidate; acts as a budget soft-cap. |

All of these priors are soft: the search can deviate when it helps. The `scripts/fit_scaling.py` utility uses run history to re-fit these trends.

## Token & compute budgeting

- Default per-candidate token budget: `tokens_per_param × params`. With ~100 M params, we cap at ~400 M tokens per live run (with early stopping).
- Rung schedule (current default):
  - Rung 1: 20% of configured steps (quick filter, e.g., 60 steps).
  - Rung 2: full steps (e.g., 300). Candidates with `ppl_code > 2.5` after rung 1 stop early.
- When scaling to 350 M–1 B for a sanity hop, multiply steps/tokens by 3–4× and keep the same rung ratios.

## Scale hop checklist (350 M–1 B)

1. Export a seed (spec + checkpoint) from the latest frontier.
2. Update dimensions: increase width/heads while keeping `head_dim=64`, `ffn≈4×d`, `kv_groups=2`.
3. Increase `train.max_tokens` to match the new params (≥10× surrogate tokens).
4. Keep `local_global` window scaled with √seq_len; if moving to 2k+ context, set `window_scale` closer to 4.
5. Run `scripts/fit_scaling.py` on recent frontiers; record predicted ppl/throughput for the new params.
6. Train the 350 M–1 B model for an hour-scale budget (e.g., 2–3k steps) to validate stability and trend adherence.

Document the run (frontier + lineage + fit outputs) before moving to multi-billion parameter training.

