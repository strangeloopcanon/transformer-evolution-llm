"""Lightweight on-disk storage for teacher logits."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch


class TopKLogitCache:
    """Loads .npz shards with uid-indexed top-k logits."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.shards = self._discover()

    def _discover(self) -> list[tuple[Path, dict[str, int]]]:
        shards: list[tuple[Path, dict[str, int]]] = []
        for file in sorted(self.root.glob("*.npz")):
            with np.load(file, allow_pickle=True) as blob:
                if "uids" not in blob:
                    continue
                uids = blob["uids"].tolist()
            if not uids:
                continue
            mapping = {str(uid): idx for idx, uid in enumerate(uids)}
            shards.append((file, mapping))
        return shards

    def fetch(self, uids: Iterable[str], topk: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx_list = []
        logp_list = []
        for uid in uids:
            entry = self._load_entry(uid)
            if entry is None:
                continue
            idx_tensor = torch.from_numpy(entry["topk_idx"][:, :topk])
            logp_tensor = torch.from_numpy(entry["topk_logp"][:, :topk])
            idx_list.append(idx_tensor)
            logp_list.append(logp_tensor)
        if not idx_list:
            raise ValueError("No cached logits found for requested uids.")
        return torch.cat(idx_list, dim=0), torch.cat(logp_list, dim=0)

    def _load_entry(self, uid: str) -> dict[str, np.ndarray] | None:
        for path, mapping in self.shards:
            offset = mapping.get(uid)
            if offset is None:
                continue
            blob = np.load(path, allow_pickle=True)
            return {
                "topk_idx": blob["topk_idx"][offset],
                "topk_logp": blob["topk_logp"][offset],
            }
        return None
