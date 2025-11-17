import random
from pathlib import Path

import torch

from transformer_evolution_llm.crossover import merge_checkpoints, splice_blocks


def test_merge_checkpoints(tmp_path: Path, tiny_spec):
    rng = random.Random(0)  # noqa: S311
    blocks, cut_a, cut_b = splice_blocks(tiny_spec, tiny_spec, rng)
    spec_data = tiny_spec.model_dump(mode="python")
    spec_data["model"]["blocks"] = [block.model_dump(mode="python") for block in blocks]
    child_spec = type(tiny_spec)(**spec_data)
    parent_ckpt = _save_state(tmp_path / "parent.pt", tiny_spec)
    out = merge_checkpoints(
        child_spec=child_spec,
        cut_a=cut_a,
        cut_b=cut_b,
        parent_a_blocks=len(tiny_spec.model.blocks),
        parent_b_blocks=len(tiny_spec.model.blocks),
        parent_a_ckpt=parent_ckpt,
        parent_b_ckpt=parent_ckpt,
        out_path=tmp_path / "child.pt",
    )
    assert out is not None and out.exists()


def _save_state(path: Path, spec):
    from transformer_evolution_llm.models import EvolutionModel

    model = EvolutionModel(spec.model)
    torch.save(model.state_dict(), path)
    return path
