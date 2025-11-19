#!/usr/bin/env bash
set -euo pipefail

# Optional: export HF_TOKEN if your data needs it
# export HF_TOKEN=...

RUN_TAG="phi_creative_super_recur_mps"
CKPT_DIR="runs/checkpoints_${RUN_TAG}"
FRONTIER="runs/frontier_${RUN_TAG}.json"
LINEAGE="runs/frontier_${RUN_TAG}_lineage.json"
ABLAT_CKPT_DIR="runs/checkpoints_${RUN_TAG}_ablation"
SUBWAY_OUT="runs/${RUN_TAG}_subway.mmd"

echo "[run] Cleaning only this run's checkpoint dirs..."
rm -rf "${CKPT_DIR}" "${ABLAT_CKPT_DIR}"

echo "[run] Super creative recurrent sweep from live_phi_creative_recur_super.yaml..."
PYTHONPATH=src python scripts/run_live.py \
  configs/live_phi_creative_recur_super.yaml \
  --generations 120 \
  --steps 480 \
  --eval-batches 8 \
  --device mps \
  --checkpoint-dir "${CKPT_DIR}" \
  --out "${FRONTIER}" \
  --seed 4242 \
  --parent-selection lexicase

echo "[run] Fitting scaling laws on super-recur frontier..."
PYTHONPATH=src python scripts/fit_scaling.py "${FRONTIER}"

echo "[run] Running ablations on top-6 frontier entries..."
PYTHONPATH=src python scripts/run_ablation.py \
  "${FRONTIER}" \
  --top-n 6 \
  --device mps \
  --steps 120 \
  --eval-batches 2 \
  --checkpoint-dir "${ABLAT_CKPT_DIR}" \
  --ablation retro_off \
  --ablation kv_groups_to_dense \
  --ablation moe_to_dense

echo "[run] Rendering lineage subway diagram..."
PYTHONPATH=src python scripts/render_lineage_subway.py \
  "${LINEAGE}" \
  --out "${SUBWAY_OUT}" \
  --title "${RUN_TAG}"

echo "[run] Cleaning ablation checkpoints to save space..."
rm -rf "${ABLAT_CKPT_DIR}"

echo "[run] Done."
echo "  Frontier:       ${FRONTIER}"
echo "  Lineage JSON:   ${LINEAGE}"
echo "  Subway diagram: ${SUBWAY_OUT}"
