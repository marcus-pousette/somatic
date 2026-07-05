"""A `Host` is one machine that can hold a slice of the model.

Hosts are given on the command line as shorthands (`localhost`, `user@ip`) or in
a small TOML file. The first host is always the driver: it owns the embedding /
final-norm / lm_head and serves the OpenAI API + chat UI, in addition to holding
its own layer slice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


def safe_run_id(run_id: str) -> str:
    """Scrub a run id to a shell/path-safe slug (it flows into paths + --runtime-id)."""

    cleaned = re.sub(r"[^a-z0-9-]+", "-", str(run_id).lower()).strip("-")
    return cleaned or "run"


def validate_model_id(model_id: str) -> str:
    """Reject a model id with shell-unsafe characters (it flows into shell commands)."""

    if not _MODEL_ID_RE.match(model_id):
        raise ValueError(
            f"model id {model_id!r} contains unsupported characters; "
            "expected only letters, digits, '.', '_', '-', '/'"
        )
    return model_id

# Defaults for a freshly set-up host. `identity=None` means SSH picks the key
# itself (ssh-agent / ~/.ssh/id_*), so no personal key path is assumed.
DEFAULT_REPO = "~/somatic"
DEFAULT_PYTHON = ".venv/bin/python"
DEFAULT_IDENTITY: str | None = None


@dataclass(frozen=True)
class Host:
    ssh: str  # "localhost" | "127.0.0.1" | "user@host"
    device: str = "auto"  # auto -> mps on Apple Silicon else cpu (resolved on the host)
    identity: str | None = None
    repo: str = DEFAULT_REPO
    python: str = DEFAULT_PYTHON
    driver: bool = False
    layers: tuple[int, int] | None = None  # pin a range; None -> solver decides

    @property
    def is_local(self) -> bool:
        return self.ssh in ("localhost", "127.0.0.1", "self")

    @property
    def display(self) -> str:
        return "localhost" if self.is_local else self.ssh

    @property
    def ip(self) -> str:
        """The address the driver dials this host's worker at."""
        if self.is_local:
            return "127.0.0.1"
        return self.ssh.split("@", 1)[-1]

    def resolved_identity(self) -> str | None:
        identity = self.identity or DEFAULT_IDENTITY
        if self.is_local or not identity:
            return None
        return str(Path(identity).expanduser())

    def remote_python(self) -> str:
        """Absolute-ish python path for remote invocation (repo-relative allowed)."""
        return self.python


def _split_layer_pin(token: str) -> tuple[str, tuple[int, int] | None]:
    """Parse an optional ``:start-end`` layer pin off a host token.

    ``localhost`` -> ("localhost", None); ``user@ip:14-28`` -> ("user@ip", (14, 28)).
    The suffix is only treated as a pin when it matches ``<int>-<int>``.
    """

    import re

    match = re.search(r":(\d+)-(\d+)$", token)
    if match:
        return token[: match.start()], (int(match.group(1)), int(match.group(2)))
    return token, None


def parse_hosts(tokens: list[str]) -> list[Host]:
    """Turn ``["localhost", "you@other-machine"]`` (with optional
    ``:start-end`` layer pins) into Hosts. The first token becomes the driver.
    """

    if not tokens:
        raise ValueError("at least one host is required")
    hosts = []
    for index, token in enumerate(tokens):
        ssh, pin = _split_layer_pin(token.strip())
        hosts.append(Host(ssh=ssh, driver=(index == 0), layers=pin))
    return hosts


def coerce_hosts(hosts: list["Host | str"]) -> list[Host]:
    """Accept a mixed list of Host objects and shorthand strings (SDK convenience).

    String items may carry a ``:start-end`` layer pin, like the CLI.
    """

    out: list[Host] = []
    for index, item in enumerate(hosts):
        if isinstance(item, str):
            ssh, pin = _split_layer_pin(item.strip())
            host = Host(ssh=ssh, layers=pin)
        else:
            host = item
        out.append(replace(host, driver=(index == 0)))
    return out
