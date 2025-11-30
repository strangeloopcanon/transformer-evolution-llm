# Run History

This file records what runs we have available locally and the notable historical runs referenced in README, so we don’t lose context again.

## Local artefacts

- `runs/frontier_phi_seeded128.json` / `runs/frontier_phi_seeded128_lineage.json`
  - Config: `configs/seed_xover-48-9237.yaml`, 128 generations, steps=220, eval-batches=4, device=mps, seed=42.
  - Frontier size: 12. All survivors collapsed to a single block (no MoE), stop_reason=3 (early stop), `ppl_code≈1.0`, `long_recall=2.0`, throughput ~1.5–2.1k.
- `runs/frontier_phi_gated128.json` / `runs/frontier_phi_gated128_lineage.json`
  - Config: `configs/live_phi_tiny.yaml`, 128 generations.
  - Frontier size: 1. Deep model (12 layers, 8 MoE) but unstable: `ppl_code≈3.06e4`, stop_reason=1 (instability).

## Historical runs (not present in this checkout)

Referenced in README history (Nov sweeps):
- `runs/frontier_phi_creative_canon.json`
- `runs/frontier_phi_creative_super_recur_mps.json`
- `runs/frontier_phi_promotion_mps.json`
- Pareto/lexicase sweeps: `runs/frontier_phi_creative_unbiased.json`, `runs/frontier_phi_creative_overnight_lexi.json`

Those earlier runs surfaced diverse hybrids (MoE + SSM + retro + sparsity + recurrence), with promotion enabling deep “hydras” to reach the frontier. Keep copies of frontier/lineage/manifest artefacts for future runs.
