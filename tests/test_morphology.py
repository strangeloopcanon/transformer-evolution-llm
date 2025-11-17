from transformer_evolution_llm.dsl import ArchitectureSpec, MoEFFNConfig
from transformer_evolution_llm.models import EvolutionModel
from transformer_evolution_llm.morphology import match_experts_to_parent, sort_moe_experts


def test_match_experts_to_parent(tiny_spec: ArchitectureSpec) -> None:
    spec = ArchitectureSpec(**tiny_spec.model_dump(mode="python"))
    spec.model.blocks[0].ffn = MoEFFNConfig(
        type="moe",
        hidden=1024,
        n_experts=2,
        k=1,
        capacity_factor=1.0,
        balance=0.05,
        shared=1,
    )
    model = EvolutionModel(spec.model)
    parent_state = model.state_dict()
    for block in model.blocks:
        if hasattr(block.ffn, "router"):
            block.ffn.router.weight.data *= 2  # type: ignore[union-attr]
    match_experts_to_parent(model, parent_state)
    sort_moe_experts(model)
