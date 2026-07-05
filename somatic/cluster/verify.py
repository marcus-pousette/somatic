"""`somatic verify` — prove the exactness/bytes tradeoff for YOUR model.

Runs a set of prompts through the split cluster under each boundary mode and
reports, honestly and per-model:

  * exact (identity): is it deterministic / reproducible? This is your
    provably-identical, full-precision reference.
  * relay (fp16) and compact (int8): how often is the generated token stream
    identical to the exact reference, and how many wire bytes does each cost?

This turns the research claim ("output survives boundary compression") into
something a user can run and check on their own hardware — no trained codec, any
Llama-family model. It does NOT claim the compressed modes are bit-exact; it
measures exactly how close they are.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from somatic.cluster.hosts import Host, coerce_hosts
from somatic.cluster.supervisor import Supervisor, boundary_strategy_for_mode

DEFAULT_PROMPTS = [
    "Explain what a transformer is in one sentence.",
    "What is the capital of Japan?",
    "Write a haiku about the sea.",
    "List three prime numbers.",
    "Why is the sky blue? Answer briefly.",
]

# Modes compared against the exact reference. Order matters for the report.
_COMPARE_MODES = ["relay", "compact"]


@dataclass
class ModeResult:
    mode: str
    strategy: str
    prompts_token_exact: int
    token_match_rate: float
    wire_bytes_per_boundary_token: int


@dataclass
class VerifyReport:
    model: str
    layout: str
    prompts: int
    max_new_tokens: int
    exact_deterministic: bool
    exact_wire_bytes_per_boundary_token: int
    modes: list[ModeResult] = field(default_factory=list)


def _token_stats(reference: list[int], candidate: list[int]) -> tuple[bool, float]:
    exact = reference == candidate
    if not reference:
        return exact, 1.0
    matched = sum(1 for a, b in zip(reference, candidate) if a == b)
    return exact, matched / len(reference)


def _measure_wire_bytes(model_id: str, strategies: list[str]) -> dict[str, int]:
    """Real encoded bytes for one [1,1,H] boundary tensor under each strategy."""

    import numpy as np
    from transformers import AutoConfig

    from somatic.sequence_model.boundary_adapters import (
        encode_payload_for_boundary_raw_frame,
        normalize_boundary_adapter_strategy,
    )
    from somatic.sequence_model.tensors import TensorPayload

    hidden = int(AutoConfig.from_pretrained(model_id, local_files_only=True).hidden_size)
    # A representative activation; int8/fp16/identity byte layout is value-independent.
    rng = np.random.default_rng(0)
    sample = (rng.standard_normal((1, 1, hidden)) * 0.5).astype("float32")
    payload = TensorPayload.from_numpy(sample, name="hidden_states")
    out: dict[str, int] = {}
    for strat in strategies:
        frame, _ = encode_payload_for_boundary_raw_frame(
            payload, strategy=normalize_boundary_adapter_strategy(strat)
        )
        out[strat] = frame.byte_size()
    return out


async def _run_all(engine, prompts: list[str], max_new_tokens: int) -> dict:
    await engine.start()
    try:
        exact_ref: list[list[int]] = []
        exact_rerun: list[list[int]] = []
        compare: dict[str, list[list[int]]] = {m: [] for m in _COMPARE_MODES}
        for prompt in prompts:
            msgs = [{"role": "user", "content": prompt}]
            exact_ref.append(await engine.generate_ids(
                msgs, max_new_tokens=max_new_tokens, boundary_strategy="identity"))
            exact_rerun.append(await engine.generate_ids(
                msgs, max_new_tokens=max_new_tokens, boundary_strategy="identity"))
            for mode in _COMPARE_MODES:
                strat = boundary_strategy_for_mode(mode)
                compare[mode].append(await engine.generate_ids(
                    msgs, max_new_tokens=max_new_tokens, boundary_strategy=strat))
        return {"exact_ref": exact_ref, "exact_rerun": exact_rerun, "compare": compare}
    finally:
        await engine.close()


def verify(
    model: str,
    hosts: list["Host | str"],
    *,
    prompts: list[str] | None = None,
    precision: str = "bf16",
    headroom: float = 0.80,
    max_new_tokens: int = 48,
    serve_port: int = 8000,
    quiet: bool = False,
    skip_preflight: bool = False,
) -> VerifyReport:
    from somatic.cluster.launcher import _make_run_id
    from somatic.serving.cluster_engine import ClusterEngine

    prompts = prompts or DEFAULT_PROMPTS
    host_objs = coerce_hosts(hosts)
    sup = Supervisor(
        run_id=_make_run_id(model), model_id=model, precision=precision,
        mode="exact", serve_port=serve_port, quiet=quiet,
    )
    plan = sup.bring_up_workers(host_objs, headroom=headroom, skip_preflight=skip_preflight)
    try:
        engine = ClusterEngine(
            model_id=model, workers=sup.worker_specs(plan),
            boundary_strategy="identity", num_threads=sup.num_threads,
        )
        results = asyncio.run(_run_all(engine, prompts, max_new_tokens))
    finally:
        sup.teardown()

    strategies = ["identity"] + [boundary_strategy_for_mode(m) for m in _COMPARE_MODES]
    wire = _measure_wire_bytes(model, strategies)

    exact_deterministic = all(
        r == d for r, d in zip(results["exact_ref"], results["exact_rerun"])
    )
    report = VerifyReport(
        model=model,
        layout=plan.layout,
        prompts=len(prompts),
        max_new_tokens=max_new_tokens,
        exact_deterministic=exact_deterministic,
        exact_wire_bytes_per_boundary_token=wire["identity"],
    )
    for mode in _COMPARE_MODES:
        strat = boundary_strategy_for_mode(mode)
        exacts = 0
        rates = []
        for ref, cand in zip(results["exact_ref"], results["compare"][mode]):
            is_exact, rate = _token_stats(ref, cand)
            exacts += int(is_exact)
            rates.append(rate)
        report.modes.append(ModeResult(
            mode=mode,
            strategy=strat,
            prompts_token_exact=exacts,
            token_match_rate=sum(rates) / len(rates) if rates else 1.0,
            wire_bytes_per_boundary_token=wire[strat],
        ))
    return report
