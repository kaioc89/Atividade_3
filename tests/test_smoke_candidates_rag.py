from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_smoke_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_candidates_rag.py"
    spec = importlib.util.spec_from_file_location("smoke_candidates_rag", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "provider": "remote_http",
        "dry_run": False,
        "no_audit_animation": True,
        "candidate_execution_strategy": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _target(module):
    return module.SmokeTarget(
        owner="Tests",
        model="candidate-j1",
        av3_provider="featherless",
        dataset="J1",
        range_start=1,
        range_end=10,
        selected_start=3,
        selected_end=3,
        calls=1,
    )


def test_smoke_command_omits_candidate_execution_strategy_by_default() -> None:
    module = _load_smoke_module()

    command = module.build_command(_args(), _target(module))

    assert "--candidate-execution-strategy" not in command


def test_smoke_command_passes_candidate_execution_strategy_when_provided() -> None:
    module = _load_smoke_module()

    command = module.build_command(
        _args(candidate_execution_strategy="parallel"),
        _target(module),
    )

    assert command[-2:] == ["--candidate-execution-strategy", "parallel"]


def test_smoke_command_passes_adaptive_candidate_execution_strategy() -> None:
    module = _load_smoke_module()

    command = module.build_command(
        _args(candidate_execution_strategy="adaptive"),
        _target(module),
    )

    assert command[-2:] == ["--candidate-execution-strategy", "adaptive"]
