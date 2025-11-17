from pathlib import Path

from transformer_evolution_llm import api
from transformer_evolution_llm.dsl import (
    ArchitectureSpec,
    BlockConfig,
    CustomModuleConfig,
    DenseFFNConfig,
)


def test_spec_summary_counts_moe_blocks(tiny_spec: ArchitectureSpec, tmp_path: Path) -> None:
    spec = tiny_spec
    assert spec.summary()["moe_blocks"] == 0

    block = spec.model.blocks[0]
    assert isinstance(block.ffn, DenseFFNConfig)
    block.ffn = DenseFFNConfig(type="dense", hidden=4096)
    spec_path = tmp_path / "spec.yaml"
    api.save_spec(spec, spec_path)
    loaded = api.load_spec(spec_path)
    assert isinstance(loaded, ArchitectureSpec)
    assert loaded.model.n_layers == 1


def test_block_accepts_custom_extras() -> None:
    block = BlockConfig(
        ffn=None,
        extras=[CustomModuleConfig(name="wild", params={"depth": 2})],
    )
    assert block.extras[0].name == "wild"
