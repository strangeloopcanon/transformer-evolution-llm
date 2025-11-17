from pathlib import Path

import pytest

from transformer_evolution_llm.llm_live_stub import main


def test_llm_live_stub_requires_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main()


def test_llm_live_stub_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    frontier = tmp_path / "runs"
    frontier.mkdir()
    (frontier / "frontier.json").write_text("[]")
    monkeypatch.chdir(tmp_path)
    main()
