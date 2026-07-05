"""Product modes map to model-general boundary strategies; verify stats are correct."""

from __future__ import annotations

from somatic.cluster.supervisor import boundary_strategy_for_mode
from somatic.cluster.verify import _token_stats


def test_mode_mapping() -> None:
    assert boundary_strategy_for_mode("relay") == "fp16"
    assert boundary_strategy_for_mode("exact") == "identity"
    assert boundary_strategy_for_mode("compact") == "int8_symmetric"


def test_mode_passthrough_for_advanced_strategy() -> None:
    # An unknown mode is treated as a raw strategy string (advanced use).
    assert boundary_strategy_for_mode("learned_residual5_int8") == "learned_residual5_int8"


def test_token_stats_exact_match() -> None:
    ref = [1, 2, 3, 4]
    exact, rate = _token_stats(ref, [1, 2, 3, 4])
    assert exact is True
    assert rate == 1.0


def test_token_stats_partial_and_divergent() -> None:
    ref = [1, 2, 3, 4]
    exact, rate = _token_stats(ref, [1, 2, 9, 9])
    assert exact is False
    assert rate == 0.5


def test_token_stats_empty_reference() -> None:
    exact, rate = _token_stats([], [])
    assert exact is True and rate == 1.0
