from pathlib import Path

from transformer_evolution_llm.api import save_spec
from transformer_evolution_llm.orchestrator import EvolutionRunner


def test_simulated_run_advances_frontier(tmp_path: Path, tiny_spec):
    cfg_path = tmp_path / "spec.yaml"
    save_spec(tiny_spec, cfg_path)
    runner = EvolutionRunner(
        base_spec=tiny_spec,
        evolution_cfg=tiny_spec.evolution,
        mode="simulate",
        seed=123,
    )
    results = runner.run(generations=2)
    assert results, "expected at least one evaluated candidate"
    assert runner.frontier.entries, "frontier should not be empty"
    out_path = tmp_path / "frontier.json"
    runner.save_frontier(out_path)
    assert out_path.exists()
