"""Data loading helpers for lightweight laptop runs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import cycle

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from .dsl import DataConfig, DatasetShard


@dataclass
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    uids: list[str]


class DataModule:
    """Cycles through configured shards and yields token batches."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer,
            revision=cfg.hf_revision,
        )  # nosec B615 - revision pinned via config
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    class _BatchIterable:
        def __init__(self, module: DataModule, max_tokens: int | None) -> None:
            self.module = module
            self.max_tokens = max_tokens
            self._iter: Iterator[TokenBatch] | None = None

        def __iter__(self) -> Iterator[TokenBatch]:
            self._iter = self.module._batch_generator(self.max_tokens)
            return self

        def __next__(self) -> TokenBatch:
            if self._iter is None:
                self._iter = self.module._batch_generator(self.max_tokens)
            return next(self._iter)

    def batches(self, max_tokens: int | None = None) -> Iterable[TokenBatch]:
        """Return a re-iterable object so training/eval can get fresh iterators."""
        return DataModule._BatchIterable(self, max_tokens)

    def _batch_generator(self, max_tokens: int | None) -> Iterator[TokenBatch]:
        budget = max_tokens
        healing_tokens = self.cfg.healing_tokens if self.cfg.healing_shards else None
        healing_iter = self._cycle_shards(self.cfg.healing_shards) if healing_tokens else None
        main_iter = self._cycle_shards(self.cfg.shards)
        current_iter: Iterator[TokenBatch]
        if healing_iter is not None:
            current_iter = healing_iter
        else:
            current_iter = main_iter
        while True:
            try:
                batch = next(current_iter)
            except StopIteration:
                if healing_iter is not None and current_iter is healing_iter:
                    healing_iter = self._cycle_shards(self.cfg.healing_shards)
                    current_iter = healing_iter
                else:
                    main_iter = self._cycle_shards(self.cfg.shards)
                    current_iter = main_iter
                batch = next(current_iter)
            yield batch
            tokens = batch.input_ids.numel()
            if healing_tokens is not None:
                healing_tokens -= tokens
                if healing_tokens <= 0:
                    healing_tokens = None
                    healing_iter = None
                    current_iter = main_iter
                    continue
            if budget is not None:
                budget -= tokens
                if budget <= 0:
                    return

    def _shard_iter(self, shard: DatasetShard) -> Iterable[TokenBatch]:
        dataset = load_dataset(  # nosec B615 - revision pinned via config
            shard.name,
            split=shard.split,
            streaming=True,
            revision=shard.revision or self.cfg.hf_revision,
        )
        tokenizer = self.tokenizer
        for idx, sample in enumerate(dataset):
            text = (
                sample.get("text")
                or sample.get("content")
                or sample.get("question")
                or "placeholder"
            )
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=self.cfg.seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            uid = f"{shard.name}-{shard.split}-{idx}"
            yield TokenBatch(
                input_ids=encoded["input_ids"].to(dtype=torch.long),
                attention_mask=encoded["attention_mask"].to(dtype=torch.long),
                uids=[uid],
            )

    def _cycle_shards(self, shards: list[DatasetShard]) -> Iterator[TokenBatch]:
        if not shards:
            return
        for shard in cycle(shards):
            yield from self._shard_iter(shard)
