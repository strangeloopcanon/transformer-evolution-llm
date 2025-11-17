# GPU Run Playbook (nanoChat-style)

Goal: rerun the evolutionary loop on a GPU-rich machine (e.g., single 80 GB A100 or dual 4090s) to train 350 M–1 B parameter candidates quickly, inspired by Karpathy’s GPT-2 “speedrun” / nanoChat workflow.

## Hardware assumptions
- 1×A100 80 GB **or** 2×RTX 4090 (24 GB) with NVLink/PCIe.
- CUDA 12.x + FlashAttention 2 for efficient attention kernels.
- Local SSD ≥ 1 TB (checkpoints + dataset caches).

## Software stack
- Python 3.11, PyTorch 2.3+ with CUDA.
- Optional: Triton kernels (FlashAttention), bitsandbytes (if we try 8-bit opt), DeepSpeed ZeRO-1 for optimizer sharding (if VRAM tight).
- HF datasets cached locally; tokenization via `gpt2` or larger BPE (customizable via DSL).

## Workflow
1. **Seed selection**
   - Start from `configs/scalehop_xover-68-8a93.yaml` (≈400–500 M params) or widen further.
   - Ensure `model.norm=rmsnorm`, `kv_groups=2`, `sparsity` includes local_global/local_block/dilated, retro modules present.

2. **Trainer configuration**
   - Device `cuda`, `steps` per rung: 200 (rung1), 1000 (rung2) with early-stop hooks.
   - Batch size: start with 2–4 sequences (micro-batch) × gradient accumulation to reach global batch ≈ 512 tokens per step.
   - Use FlashAttention kernels automatically (torch.backends.cuda.sdp_kernel).

3. **Evolution settings**
   - Population 24, lexicase selection, priors tokens_per_param≈6.
   - Checkpoint dir on fast NVMe; `--cleanup-old-checkpoints` ON.
   - Run command (example):
     ```bash
     HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 \
     python scripts/run_live.py configs/scalehop_xover-68-8a93.yaml \
       --generations 48 --steps 1000 --eval-batches 8 \
       --device cuda --checkpoint-dir runs/checkpoints_scalehop \
       --out runs/frontier_scalehop.json --cleanup-old-checkpoints \
       --parent-selection lexicase
     ```

4. **Data**
   - Mix of news/wikitext/code still works; for scale-hop consider adding math or instruction corpora.
   - Sequence length 2048; set `window_scale` accordingly.

5. **Monitoring**
   - `nvidia-smi dmon` for VRAM/compute.
   - `tail -f runs/scalehop.log` for live mutation info.

6. **Post-run**
   - Run ablations (`scripts/run_ablation.py`) on top candidates.
   - Fit scaling script again with new data; decide final recipe for 1.5–8 B training.

## TODO when GPUs available
- Test multi-GPU (ZeRO-1 or FSDP) for larger populations or >1 B models.
- Explore Lion or Muon optimizers; record deltas vs AdamW.
- Benchmark FlashAttention vs vanilla MHA to confirm throughput gains.
