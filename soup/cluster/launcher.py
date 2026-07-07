"""`from soup import Cluster` — the one object behind the CLI, the SDK, and the UI.

    from soup import Cluster, Host
    with Cluster.launch("Qwen/Qwen3-1.7B",
                        ["localhost", "you@other-machine"]) as c:
        print(c.plan.layout)                 # [0,14)@localhost -> [14,28)@other-machine
        print(c.chat("Explain layer-split inference in one line."))
        for delta in c.stream("Now in French."):
            print(delta, end="", flush=True)
        c.openai().chat.completions.create(   # the stock OpenAI SDK, verbatim
            model=c.model, messages=[{"role": "user", "content": "hi"}])

`chat`/`stream` are synchronous — they POST to the cluster's own OpenAI endpoint,
so there is a single engine on a single event loop and the SDK reads like any
blocking client. The context-manager form tears the whole cluster down on exit.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Iterator

from soup.cluster.capacity import LaunchPlan
from soup.cluster.hosts import Host, coerce_hosts
from soup.cluster.runstate import RunRecord
from soup.cluster.supervisor import Supervisor, build_plan


def _make_run_id(model_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")[:24]
    return f"{slug}-{uuid.uuid4().hex[:6]}"


def _lan_ip() -> str:
    """This machine's LAN IP (best-effort) so a UI on another device can reach us."""

    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))  # no packets sent; just picks the egress iface
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


class Cluster:
    def __init__(self, supervisor: Supervisor, plan: LaunchPlan) -> None:
        self._sup = supervisor
        self._plan = plan

    # ---- construction ----------------------------------------------------

    @classmethod
    def launch(
        cls,
        model: str,
        hosts: list["Host | str"],
        *,
        precision: str = "bf16",
        mode: str = "relay",
        serve_host: str = "127.0.0.1",
        serve_port: int = 8000,
        headroom: float = 0.80,
        num_threads: int = 8,
        run_id: str | None = None,
        quiet: bool = False,
        skip_preflight: bool = False,
    ) -> "Cluster":
        host_objs = coerce_hosts(hosts)
        sup = Supervisor(
            run_id=run_id or _make_run_id(model),
            model_id=model,
            precision=precision,
            mode=mode,
            serve_host=serve_host,
            serve_port=serve_port,
            num_threads=num_threads,
            quiet=quiet,
        )
        plan = sup.launch(host_objs, headroom=headroom, skip_preflight=skip_preflight)
        return cls(sup, plan)

    @classmethod
    def plan(
        cls,
        model: str,
        hosts: list["Host | str"],
        *,
        precision: str = "bf16",
        headroom: float = 0.80,
        quiet: bool = True,
    ) -> LaunchPlan:
        """Dry run: preflight-free footprint + probe + fit. Launches nothing."""

        return build_plan(model, coerce_hosts(hosts), precision=precision, headroom=headroom, quiet=quiet)

    # ---- inference (over the cluster's own OpenAI endpoint) --------------

    @property
    def model(self) -> str:
        return self._sup.model_id

    @property
    def plan_(self) -> LaunchPlan:
        return self._plan

    @property
    def _local_host(self) -> str:
        # Always connect locally over the loopback (0.0.0.0 is a bind address,
        # not a usable connect address).
        return "127.0.0.1"

    @property
    def exposed(self) -> bool:
        return self._sup.serve_host == "0.0.0.0"

    @property
    def openai_base_url(self) -> str:
        return f"http://{self._local_host}:{self._sup.serve_port}/v1"

    @property
    def web_url(self) -> str:
        return f"http://{self._local_host}:{self._sup.serve_port}/"

    @property
    def advertised_base_url(self) -> str:
        """The base URL to hand to a UI — the LAN IP when exposed, else loopback."""

        host = _lan_ip() if self.exposed else self._local_host
        return f"http://{host}:{self._sup.serve_port}/v1"

    @property
    def advertised_web_url(self) -> str:
        host = _lan_ip() if self.exposed else self._local_host
        return f"http://{host}:{self._sup.serve_port}/"

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def chat(self, prompt: str, *, max_new_tokens: int = 256, system: str | None = None) -> str:
        import httpx

        resp = httpx.post(
            f"{self.openai_base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": self._messages(prompt, system),
                "max_tokens": max_new_tokens,
            },
            timeout=600.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def stream(self, prompt: str, *, max_new_tokens: int = 256, system: str | None = None) -> Iterator[str]:
        import httpx

        with httpx.stream(
            "POST",
            f"{self.openai_base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": self._messages(prompt, system),
                "max_tokens": max_new_tokens,
                "stream": True,
            },
            timeout=600.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta

    def openai(self):
        """A stock `openai.OpenAI` client pointed at this cluster."""

        import openai

        return openai.OpenAI(base_url=self.openai_base_url, api_key="soup")

    # ---- teardown --------------------------------------------------------

    def down(self) -> None:
        self._sup.teardown()

    def __enter__(self) -> "Cluster":
        return self

    def __exit__(self, *exc) -> None:
        self.down()


def teardown_run(run_id: str | None = None, *, sweep_hosts: list["Host | str"] | None = None) -> str | None:
    """Tear down a run by id (or the latest). Used by `soup down`."""

    from soup.cluster import ssh

    if sweep_hosts is not None:
        for host in coerce_hosts(sweep_hosts):
            ssh.sweep(host, "live_split_worker.py")
        return "swept"

    record = RunRecord.load(run_id) if run_id else RunRecord.load_latest()
    if record is None:
        return None
    for wrec in record.workers:
        if wrec.pid is not None:
            host = Host(ssh=wrec.ssh, identity=wrec.identity, repo=wrec.repo)
            ssh.kill_pid(host, wrec.pid, log_path=wrec.log_path)
    record.remove()
    return record.run_id
