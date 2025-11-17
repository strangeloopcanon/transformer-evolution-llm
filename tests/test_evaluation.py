from transformer_evolution_llm.dsl import ArchitectureSpec
from transformer_evolution_llm.evaluation import (
    StaticChecker,
    estimate_params,
    kv_bytes_per_token,
    throughput_proxy,
)


def test_static_checker_metrics(tiny_spec: ArchitectureSpec) -> None:
    params = estimate_params(tiny_spec)
    kv = kv_bytes_per_token(tiny_spec)
    tps = throughput_proxy(tiny_spec, tiny_spec.data.seq_len)
    checker = StaticChecker()
    result = checker.run(tiny_spec)
    assert params > 0
    assert kv > 0
    assert tps > 0
    assert isinstance(result.metrics, dict)
