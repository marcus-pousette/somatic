"""Bring a cluster up and take it down: spawn workers, health-gate, serve.

The Supervisor owns process lifecycle. `launch()` runs preflight → probe → fit →
spawn (local + SSH) → health-wait → start the OpenAI+UI server, and records the
run to disk before health-waiting so a worker that dies is still teardownable.
The server (and thus the ClusterEngine) lives on its own uvicorn thread; the SDK
and CLI talk to it over HTTP, so there is exactly one engine on one event loop.
"""

from __future__ import annotations

import shlex
import threading
import time
import urllib.request

from somatic.cluster import doctor, ssh
from somatic.cluster.hosts import safe_run_id, validate_model_id
from somatic.cluster.capacity import LaunchPlan, PlanEntry, fit_ranges, probe_free_ram
from somatic.cluster.errors import WorkerFailedError
from somatic.cluster.footprint import ModelFootprint, model_footprint
from somatic.cluster.hosts import Host
from somatic.cluster.runstate import RunRecord, WorkerRecord

GIB = 1024 ** 3


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(f"soup ▸ {msg}", flush=True)


# Product modes map to a boundary wire strategy the worker already understands.
# All three are model-general (no trained artifact). `exact` (identity) is byte-
# identical to full precision; `relay`/`compact` trade measured fidelity for bytes.
_MODE_STRATEGY = {
    "relay": "fp16",            # default: ~2x fewer bytes, near-exact
    "exact": "identity",        # provably identical (full-precision hidden states)
    "compact": "int8_symmetric",  # ~4x fewer bytes, more lossy
}


def boundary_strategy_for_mode(mode: str) -> str:
    """Resolve a product mode to a worker boundary strategy.

    Unknown modes pass through as a raw strategy string (advanced use).
    """

    return _MODE_STRATEGY.get(mode, mode)


def build_plan(
    model_id: str,
    hosts: list[Host],
    *,
    precision: str = "bf16",
    headroom: float = 0.80,
    local_files_only: bool = False,
    quiet: bool = True,
) -> LaunchPlan:
    """Pure planning: footprint + probe + fit. Launches nothing."""

    fp = model_footprint(model_id, precision=precision, local_files_only=local_files_only)
    _log(quiet, f"{model_id}  {fp.total_layers} layers · {fp.layer_bytes_mean/GIB:.3f} GiB/layer · head {fp.head_bytes/GIB:.2f} GiB ({precision})")
    free: dict[str, int] = {}
    for host in hosts:
        free[host.ssh] = probe_free_ram(host)
        _log(quiet, f"free RAM   {host.display}: {free[host.ssh]/GIB:.1f} GiB")
    plan = fit_ranges(fp, hosts, free, headroom=headroom)
    for e in plan.entries:
        tag = " (driver)" if e.is_driver else ""
        _log(quiet, f"plan       {host_label(e)}{tag}: [{e.layer_start},{e.layer_end}) {e.num_layers} layers, ~{plan.bytes_on(e)/GIB:.1f} GiB")
    return plan


def host_label(entry: PlanEntry) -> str:
    return entry.host.display


class Supervisor:
    def __init__(
        self,
        *,
        run_id: str,
        model_id: str,
        precision: str = "bf16",
        mode: str = "relay",
        serve_host: str = "127.0.0.1",
        serve_port: int = 8000,
        num_threads: int = 8,
        quiet: bool = False,
    ) -> None:
        self.run_id = safe_run_id(run_id)
        self.model_id = validate_model_id(model_id)
        self.precision = precision
        self.mode = mode
        self.serve_host = serve_host
        self.serve_port = serve_port
        self.num_threads = num_threads
        self.quiet = quiet
        self.record: RunRecord | None = None
        self._server = None
        self._server_thread: threading.Thread | None = None
        self._plan: LaunchPlan | None = None

    # ---- lifecycle -------------------------------------------------------

    def launch(self, hosts: list[Host], *, headroom: float = 0.80, skip_preflight: bool = False) -> LaunchPlan:
        plan = self.bring_up_workers(hosts, headroom=headroom, skip_preflight=skip_preflight)
        self._start_server(plan)
        _log(self.quiet, f"ready.  chat http://{self.serve_host}:{self.serve_port}/   api http://{self.serve_host}:{self.serve_port}/v1")
        _log(self.quiet, "stop:  soup down")
        return plan

    def bring_up_workers(self, hosts: list[Host], *, headroom: float = 0.80, skip_preflight: bool = False) -> LaunchPlan:
        """Preflight → probe → fit → spawn → health-gate, WITHOUT starting the
        server. Used by `launch` (which then serves) and by `soup verify`
        (which drives the engine directly). Tears down on any failure."""

        if not skip_preflight:
            for host in hosts:
                _log(self.quiet, f"preflight  {host.display} …")
                doctor.assert_ready(host, self.model_id)

        plan = build_plan(
            self.model_id, hosts, precision=self.precision, headroom=headroom, quiet=self.quiet
        )
        self._plan = plan

        record = RunRecord(
            run_id=self.run_id,
            model_id=self.model_id,
            precision=self.precision,
            mode=self.mode,
            serve_host=self.serve_host,
            serve_port=self.serve_port,
        )
        # Spawn every worker, recording PIDs BEFORE health-wait.
        for index, entry in enumerate(plan.entries):
            runtime_id = f"{self.run_id}-w{index}"
            log_path = self._log_path(entry, index)
            device = self._device_flag(entry.host)
            cmd = (
                f"{entry.host.remote_python()} scripts/live_split_worker.py "
                f"--model-id {shlex.quote(self.model_id)} --runtime-id {shlex.quote(runtime_id)} "
                f"--layer-start {entry.layer_start} --layer-end {entry.layer_end} "
                f"--port {entry.port} --host {'127.0.0.1' if entry.host.is_local else '0.0.0.0'} "
                f"--device {shlex.quote(device)} --precision {shlex.quote(self.precision)} --shard-loading "
                f"--num-threads {int(self.num_threads)}"
            )
            pid = ssh.launch_worker(entry.host, cmd, log_path)
            record.workers.append(
                WorkerRecord(
                    ssh=entry.host.ssh,
                    is_local=entry.host.is_local,
                    ip=entry.host.ip,
                    port=entry.port,
                    pid=pid,
                    layer_start=entry.layer_start,
                    layer_end=entry.layer_end,
                    runtime_id=runtime_id,
                    log_path=log_path,
                    identity=entry.host.resolved_identity(),
                    repo=entry.host.repo,
                )
            )
        record.save()
        self.record = record
        _log(self.quiet, "workers    launching …")

        try:
            self._wait_all_healthy(plan)
        except WorkerFailedError:
            self.teardown()
            raise
        return plan

    def _wait_all_healthy(self, plan: LaunchPlan) -> None:
        assert self.record is not None
        for index, (entry, wrec) in enumerate(zip(plan.entries, self.record.workers)):
            shard_gib = plan.bytes_on(entry) / GIB
            deadline = time.monotonic() + 30 + shard_gib * 8
            url = ssh.health_url(entry.host, entry.port)
            while True:
                if self._is_healthy(url):
                    _log(self.quiet, f"worker     {entry.host.display}:{entry.port} healthy [{entry.layer_start},{entry.layer_end})")
                    break
                if time.monotonic() > deadline:
                    tail = self._tail_log(entry.host, wrec.log_path)
                    raise WorkerFailedError(
                        f"worker on {entry.host.display} (layers [{entry.layer_start},{entry.layer_end})) "
                        f"never became healthy within {int(30 + shard_gib*8)}s.\n  last log:\n{tail}"
                    )
                if self._worker_dead(entry.host, wrec):
                    tail = self._tail_log(entry.host, wrec.log_path)
                    raise WorkerFailedError(
                        f"worker on {entry.host.display} died during startup.\n  last log:\n{tail}"
                    )
                time.sleep(1.5)

    def worker_specs(self, plan: LaunchPlan):
        """WorkerSpec list for a plan — shared by the server and `verify`."""

        from somatic.serving.cluster_engine import WorkerSpec

        return [
            WorkerSpec(f"http://{e.host.ip}:{e.port}", e.layer_start, e.layer_end)
            for e in plan.entries
        ]

    def _start_server(self, plan: LaunchPlan) -> None:
        import uvicorn

        from somatic.cluster.server import build_cluster_app
        from somatic.serving.cluster_engine import ClusterEngine

        engine = ClusterEngine(
            model_id=self.model_id,
            workers=self.worker_specs(plan),
            boundary_strategy=boundary_strategy_for_mode(self.mode),
            num_threads=self.num_threads,
        )
        status = {
            "model": self.model_id,
            "mode": self.mode,
            "workers": plan.layout,
            "entries": [
                {"host": e.host.display, "layers": [e.layer_start, e.layer_end],
                 "free_gib": round(e.free_bytes / GIB, 1), "driver": e.is_driver}
                for e in plan.entries
            ],
        }
        app = build_cluster_app(engine, model_name=self.model_id, status=status)
        config = uvicorn.Config(app, host=self.serve_host, port=self.serve_port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server_thread = threading.Thread(target=self._server.run, daemon=True)
        self._server_thread.start()

        deadline = time.monotonic() + 60
        url = f"http://{self.serve_host}:{self.serve_port}/health"
        while time.monotonic() < deadline:
            if self._is_healthy(url):
                return
            time.sleep(0.5)
        raise WorkerFailedError("OpenAI server did not become healthy")

    def teardown(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            if self._server_thread is not None:
                self._server_thread.join(timeout=8)
            self._server = None
        record = self.record or RunRecord.load(self.run_id)
        for wrec in record.workers:
            if wrec.pid is not None:
                host = Host(ssh=wrec.ssh, identity=wrec.identity, repo=wrec.repo)
                ssh.kill_pid(host, wrec.pid, log_path=wrec.log_path)
        record.remove()

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _is_healthy(url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _worker_dead(self, host: Host, wrec: WorkerRecord) -> bool:
        if wrec.pid is None:
            return False
        result = ssh.run(host, f"kill -0 {wrec.pid} 2>/dev/null && echo alive || echo dead", timeout=10.0)
        return "dead" in result.stdout

    def _tail_log(self, host: Host, log_path: str) -> str:
        result = ssh.run(host, f"tail -n 15 {shlex.quote(log_path)} 2>/dev/null", timeout=10.0)
        return "\n".join("      " + ln for ln in result.stdout.splitlines()[-15:]) or "      (no log output)"

    def _log_path(self, entry: PlanEntry, index: int) -> str:
        if entry.host.is_local:
            from somatic.cluster.runstate import STATE_ROOT

            d = STATE_ROOT / self.run_id
            d.mkdir(parents=True, exist_ok=True)
            return str(d / f"w{index}.log")
        return f".somatic/{self.run_id}-w{index}.log"

    def _device_flag(self, host: Host) -> str:
        if host.device and host.device != "auto":
            return host.device
        # Resolve 'auto' on the host: mps if torch says so, else cpu.
        result = ssh.run(
            host,
            f"{host.remote_python()} -c \"import torch; print('mps' if torch.backends.mps.is_available() else 'cpu')\"",
            timeout=25.0,
        )
        out = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "cpu"
        return "mps" if out == "mps" else "cpu"
