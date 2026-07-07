"""shell_quote_path preserves ~ expansion while keeping paths injection-safe."""

from __future__ import annotations

from soup.cluster.ssh import shell_quote_path
from soup.cluster.provision import _hf_cache_dirname


def test_tilde_preserved_when_no_special_chars() -> None:
    assert shell_quote_path("~/soup") == "~/soup"
    assert shell_quote_path("~/.local/bin/uv") == "~/.local/bin/uv"
    assert shell_quote_path("~") == "~"


def test_absolute_path_unchanged() -> None:
    assert shell_quote_path("/abs/path") == "/abs/path"


def test_spaces_after_tilde_are_quoted() -> None:
    out = shell_quote_path("~/my dir/x")
    assert out.startswith("~/")
    assert "'my dir/x'" in out


def test_injection_after_tilde_is_neutralised() -> None:
    # The metacharacters must end up inside quotes, not executable.
    for evil in ["~/a;rm -rf b", "~/a$(touch x)", "~/a`whoami`", "~/a|b", "~/a&&b"]:
        out = shell_quote_path(evil)
        assert out.startswith("~/")
        # everything after ~/ is single-quoted, so no metachar is live
        body = out[2:]
        assert body.startswith("'") and body.endswith("'")


def test_tilde_without_slash_is_fully_quoted() -> None:
    # "~evil" is not a home reference we want to expand.
    assert shell_quote_path("~evil") == "'~evil'"


def test_hf_cache_dirname() -> None:
    assert _hf_cache_dirname("Qwen/Qwen3-1.7B") == "models--Qwen--Qwen3-1.7B"
    assert _hf_cache_dirname("HuggingFaceTB/SmolLM2-135M-Instruct") == "models--HuggingFaceTB--SmolLM2-135M-Instruct"
