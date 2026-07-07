"""run_id sanitization and model_id validation keep untrusted input out of the shell."""

from __future__ import annotations

import pytest

from soup.cluster.hosts import safe_run_id, validate_model_id


def test_safe_run_id_strips_shell_metacharacters() -> None:
    dirty = "x; touch /tmp/pwned; echo $(whoami)"
    clean = safe_run_id(dirty)
    assert all(c.isalnum() or c == "-" for c in clean)
    assert ";" not in clean and "$" not in clean and "/" not in clean and " " not in clean


def test_safe_run_id_never_empty() -> None:
    assert safe_run_id("!!!") == "run"
    assert safe_run_id("") == "run"


def test_validate_model_id_accepts_normal_ids() -> None:
    for good in ("Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM2-135M-Instruct", "meta-llama/Llama-3.2-1B"):
        assert validate_model_id(good) == good


def test_validate_model_id_rejects_injection() -> None:
    for bad in ("x; rm -rf /", "a$(touch /tmp/x)", "a`whoami`", "a b", "a|b", "a&b"):
        with pytest.raises(ValueError):
            validate_model_id(bad)
