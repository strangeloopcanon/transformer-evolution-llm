"""Data loading helpers for lightweight laptop runs."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

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

    def __init__(self, cfg: DataConfig, *, seed: int = 0) -> None:
        self.cfg = cfg
        self._seed = int(seed)
        self._rng = random.Random(self._seed)  # noqa: S311  # nosec B311 - deterministic batches
        self._dataset_cache: dict[tuple[str, str, str, bool, str | None], object] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer,
            revision=cfg.hf_revision,
        )  # nosec B615 - revision pinned via config
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset deterministic shard sampling."""

        if seed is not None:
            self._seed = int(seed)
        self._rng = random.Random(self._seed)  # noqa: S311  # nosec B311 - deterministic batches

    def _load_dataset(self, shard: DatasetShard):
        """Load a dataset (streaming or cached) with backward-compatible shard.split parsing.

        Supported shard.split formats:
        - "train" (regular split)
        - "<config>" (dataset config/subset; defaults split to "train")
        - "<config>:<split>" (explicit config + split)
        """
        revision = shard.revision or self.cfg.hf_revision
        streaming = bool(getattr(self.cfg, "streaming", True))
        split_raw = str(shard.split or "train")
        cache_dir = shard.cache_path or None
        cache_key = (shard.name, split_raw, revision, streaming, cache_dir)
        cached = self._dataset_cache.get(cache_key)
        if cached is not None:
            return cached
        if ":" in split_raw:
            cfg_name, split_name = split_raw.split(":", 1)
            dataset = load_dataset(  # nosec B615 - revision pinned via config
                shard.name,
                cfg_name,
                split=split_name,
                streaming=streaming,
                revision=revision,
                cache_dir=cache_dir,
            )
            self._dataset_cache[cache_key] = dataset
            return dataset
        try:
            dataset = load_dataset(  # nosec B615 - revision pinned via config
                shard.name,
                split=split_raw,
                streaming=streaming,
                revision=revision,
                cache_dir=cache_dir,
            )
            self._dataset_cache[cache_key] = dataset
            return dataset
        except ValueError as exc:
            # Common case: configs historically used DatasetShard.split to store the dataset config
            # (e.g., wikitext-2-raw-v1). Detect and retry with a default split.
            msg = str(exc)
            if "Config name is missing" in msg or "available configs" in msg:
                dataset = load_dataset(  # nosec B615 - revision pinned via config
                    shard.name,
                    split_raw,
                    split="train",
                    streaming=streaming,
                    revision=revision,
                    cache_dir=cache_dir,
                )
                self._dataset_cache[cache_key] = dataset
                return dataset
            raise

    @dataclass
    class _ShardStream:
        module: DataModule
        shard: DatasetShard
        iterator: Iterator[TokenBatch]

        def next_batch(self) -> TokenBatch:
            try:
                return next(self.iterator)
            except StopIteration:
                # Restart the shard stream when it runs out (streaming datasets are often finite).
                self.iterator = iter(self.module._shard_iter(self.shard))
                return next(self.iterator)

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
        healing_remaining = self.cfg.healing_tokens if self.cfg.healing_shards else None

        main_streams = [
            DataModule._ShardStream(self, shard, iter(self._shard_iter(shard)))
            for shard in self.cfg.shards
        ]
        main_weights = [float(shard.weight) for shard in self.cfg.shards]
        healing_streams = (
            [
                DataModule._ShardStream(self, shard, iter(self._shard_iter(shard)))
                for shard in self.cfg.healing_shards
            ]
            if healing_remaining is not None and self.cfg.healing_shards
            else []
        )
        healing_weights = [float(shard.weight) for shard in self.cfg.healing_shards]

        def pick_stream(
            streams: list[DataModule._ShardStream], weights: list[float]
        ) -> DataModule._ShardStream:
            if len(streams) == 1:
                return streams[0]
            idx = self._rng.choices(range(len(streams)), weights=weights, k=1)[0]
            return streams[idx]

        while True:
            if healing_remaining is not None and healing_streams:
                batch = pick_stream(healing_streams, healing_weights).next_batch()
            else:
                batch = pick_stream(main_streams, main_weights).next_batch()
            yield batch
            tokens = batch.input_ids.numel()
            if healing_remaining is not None:
                healing_remaining -= tokens
                if healing_remaining <= 0:
                    healing_remaining = None
                    continue
            if budget is not None:
                budget -= tokens
                if budget <= 0:
                    return

    def _shard_iter(self, shard: DatasetShard) -> Iterable[TokenBatch]:
        dataset = self._load_dataset(shard)
        tokenizer = self.tokenizer
        batch_size = max(1, int(self.cfg.batch_size))
        uids: list[str] = []
        input_ids: list[torch.Tensor] = []
        attention_mask: list[torch.Tensor] = []
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
            # Skip samples that yield <2 non-padding tokens; they contribute no
            # next-token prediction targets after shifting.
            mask = encoded.get("attention_mask")
            if isinstance(mask, torch.Tensor) and int(mask.sum().item()) < 2:
                continue
            uids.append(f"{shard.name}-{shard.split}-{idx}")
            input_ids.append(encoded["input_ids"].to(dtype=torch.long))
            attention_mask.append(encoded["attention_mask"].to(dtype=torch.long))
            if len(uids) >= batch_size:
                yield TokenBatch(
                    input_ids=torch.cat(input_ids, dim=0),
                    attention_mask=torch.cat(attention_mask, dim=0),
                    uids=uids,
                )
                uids = []
                input_ids = []
                attention_mask = []
        if uids:
            yield TokenBatch(
                input_ids=torch.cat(input_ids, dim=0),
                attention_mask=torch.cat(attention_mask, dim=0),
                uids=uids,
            )
