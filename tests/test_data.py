from collections.abc import Iterator
from typing import Any

import torch

from transformer_evolution_llm import dsl
from transformer_evolution_llm.data import DataModule


class DummyDataset:
    def __iter__(self) -> Iterator[dict[str, str]]:
        for i in range(3):
            yield {"text": f"sample-{i}"}


class DummyTokenizer:
    pad_token = None
    eos_token = "<eos>"  # noqa: S105 - tokenizer sentinel

    def __call__(self, *_args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        length = kwargs["max_length"]
        ids = torch.zeros((1, length), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_data_module_batches(monkeypatch):
    from transformer_evolution_llm import data as data_module

    class Loader:
        @staticmethod
        def from_pretrained(_name: str, **_kwargs: Any) -> DummyTokenizer:
            return DummyTokenizer()

    monkeypatch.setattr(data_module, "AutoTokenizer", Loader)
    monkeypatch.setattr(data_module, "load_dataset", lambda *args, **kwargs: DummyDataset())

    cfg = dsl.DataConfig(
        tokenizer="hf-internal-testing/tiny-random-gpt2",
        seq_len=8,
        batch_size=1,
        workers=0,
        shards=[dsl.DatasetShard(name="dummy", split="train", weight=1.0)],
    )
    module = DataModule(cfg)
    batch = next(module.batches(max_tokens=16))
    assert batch.input_ids.shape[-1] == cfg.seq_len
