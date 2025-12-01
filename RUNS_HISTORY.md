# Run History

This file records what runs we have available locally and the notable historical runs referenced in README, so we don’t lose context again.

## Local artefacts

- `runs/frontier_phi_seeded128.json` / `runs/frontier_phi_seeded128_lineage.json`
  - Config: `configs/seed_xover-48-9237.yaml`, 128 generations, steps=220, eval-batches=4, device=mps, seed=42.
  - Frontier size: 25. Survivors skew shallow (2–6 layers; one with 2 MoE blocks) with retro-heavy blocks; `stop_reason=3` (early stop). Metrics: `ppl_code≈1.0`, `long_recall≈1.0–1.5`, throughput ~0.8–2.4k. Promotion rung never fired in this sweep.
- `runs/frontier_phi_gated128.json` / `runs/frontier_phi_gated128_lineage.json`
  - Config: `configs/live_phi_tiny.yaml`, 128 generations.
  - Frontier size: 1. Deep model (12 layers, 8 MoE) but unstable: `ppl_code≈3.06e4`, stop_reason=1 (instability).
- `runs/frontier_phi_entropy_v2.json` / `runs/frontier_phi_entropy_v2_lineage.json`
  - Config: `configs/seed_xover-48-9237.yaml`, 160 generations, steps=256, eval-batches=4, device=mps, seed=84.
  - Frontier size: 99. Balanced mix of shallow retro stacks and deep MoE/SSM/retro hybrids (up to 30 layers, 14 MoE, 8 SSM). Top-quality deep seeds (e.g., `xover-15-3ab4`, `xover-72-d77d`) sit near `ppl≈1.0–1.05` with throughput 200–300 tok/s; maximal structure probe `xover-119-1d12` hits `ppl≈1.05` with very low throughput (~5 tok/s). Promotion and graph-entropy objectives restored complex hydras to the frontier.

## Historical runs (not present in this checkout)

Referenced in README history (Nov sweeps):
- `runs/frontier_phi_creative_canon.json`
- `runs/frontier_phi_creative_super_recur_mps.json`
- `runs/frontier_phi_promotion_mps.json`
- Pareto/lexicase sweeps: `runs/frontier_phi_creative_unbiased.json`, `runs/frontier_phi_creative_overnight_lexi.json`

Those earlier runs surfaced diverse hybrids (MoE + SSM + retro + sparsity + recurrence), with promotion enabling deep “hydras” to reach the frontier. Keep copies of frontier/lineage/manifest artefacts for future runs.
