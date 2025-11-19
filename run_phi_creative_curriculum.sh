#!/usr/bin/env bash
set -euo pipefail

# Optional: export HF_TOKEN if required for datasets
# export HF_TOKEN=...

BASE_TAG="phi_creative_curriculum_mps"
CYCLES=${CYCLES:-1}

for cycle in $(seq 1 "$CYCLES"); do
  RUN_TAG="${BASE_TAG}_cycle${cycle}"
  STAGE_A_CKPTS="runs/checkpoints_${RUN_TAG}_stageA"
  STAGE_A_FRONTIER="runs/frontier_${RUN_TAG}_stageA.json"
  STAGE_A_LINEAGE="runs/frontier_${RUN_TAG}_stageA_lineage.json"
  STAGE_A_SUBWAY="runs/${RUN_TAG}_stageA_subway.mmd"
  STAGE_B_OUT="runs/long_finetunes_${RUN_TAG}.json"

  echo "[run][cycle ${cycle}] Cleaning stage-specific checkpoint dirs..."
  rm -rf "${STAGE_A_CKPTS}"

  echo "[run][cycle ${cycle}] Stage A: creative evolution (fast sweep)..."
  PYTHONPATH=src python scripts/run_live.py \
    configs/live_phi_creative_recur_super.yaml \
    --generations 150 \
    --steps 480 \
    --eval-batches 8 \
    --device mps \
    --checkpoint-dir "${STAGE_A_CKPTS}" \
    --out "${STAGE_A_FRONTIER}" \
    --seed $((2025 + cycle)) \
    --parent-selection lexicase

  echo "[run][cycle ${cycle}] Stage A stats: scaling + subway..."
  PYTHONPATH=src python scripts/fit_scaling.py "${STAGE_A_FRONTIER}"
  PYTHONPATH=src python scripts/render_lineage_subway.py \
    "${STAGE_A_LINEAGE}" \
    --out "${STAGE_A_SUBWAY}" \
    --title "${RUN_TAG}_stageA"

  echo "[run][cycle ${cycle}] Stage B: long finetunes for diverse picks..."
  PYTHONPATH=src python scripts/pick_and_finetune.py \
    "${STAGE_A_FRONTIER}" \
    "${STAGE_A_CKPTS}" \
    --out "${STAGE_B_OUT}" \
    --top-n 6 \
    --steps 1800 \
    --eval-batches 8 \
    --device mps \
    --instability-threshold 60.0 \
    --no-improve-patience 400 \
    --entropy-threshold 0.2 \
    --token-multiplier 1.5

  echo "[run][cycle ${cycle}] Cleaning stage A checkpoints to save space..."
  rm -rf "${STAGE_A_CKPTS}"

  echo "[run][cycle ${cycle}] Completed."
  echo "    Stage A frontier: ${STAGE_A_FRONTIER}"
  echo "    Stage B finetunes: ${STAGE_B_OUT}"
done

echo "[run] All cycles completed (total ${CYCLES})."
