#!/usr/bin/env bash
set -euo pipefail

# Optional: export HF_TOKEN if needed for datasets
# export HF_TOKEN=...

RUN_TAG="phi_promotion_mps"
CKPT_DIR="runs/checkpoints_${RUN_TAG}"
FRONTIER="runs/frontier_${RUN_TAG}.json"
LINEAGE="runs/frontier_${RUN_TAG}_lineage.json"
SUBWAY_OUT="runs/${RUN_TAG}_subway.mmd"

echo "[run] Cleaning checkpoint dir for this run..."
rm -rf "${CKPT_DIR}"

echo "[run] Live evolution with promotion from live_phi_creative_recur_super.yaml..."
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

echo "[run] Rendering lineage subway diagram..."
PYTHONPATH=src python scripts/render_lineage_subway.py \
  "${LINEAGE}" \
  --out "${SUBWAY_OUT}" \
  --title "${RUN_TAG}"

echo "[run] Cleaning checkpoints to save space..."
rm -rf "${CKPT_DIR}"

echo "[run] Done."
echo "  Frontier:       ${FRONTIER}"
echo "  Lineage JSON:   ${LINEAGE}"
echo "  Subway diagram: ${SUBWAY_OUT}"

