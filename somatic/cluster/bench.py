"""`somatic bench` — a reproducible, honest throughput number for a split model.

Most distributed-inference tools publish no single-stream tok/s at all. This one
does: it times real decoding across your cluster and, when the model fits on the
driver, also measures the memory-bandwidth *frontier* (the single-machine floor)
so you can see how close the split runs to the physical limit.

The reported numbers are wall-clock, warmed-up, and synced — no marketing peak.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from somatic.cluster.hosts import Host, coerce_hosts
from somatic.cluster.supervisor import Supervisor, boundary_strategy_for_mode

_BENCH_PROMPT = "Explain in detail how a computer network routes packets across the internet."


@dataclass
class BenchReport:
    model: str
    layout: str
    mode: str
    prefill_tok_s: float
    decode_tok_s: float
    decode_ms_per_token: float
    frontier_tok_s: float | None  # single-machine memory-bound floor (if measurable)
    pct_of_frontier: float | None
    warmup: int
    steps: int


def _sync(device: str) -> None:
    import torch

    if "mps" in device:
        torch.mps.synchronize()
    elif "cuda" in device:
        torch.cuda.synchronize()


def _measure_frontier(model_id: str, warmup: int, steps: int) -> float | None:
    """Synced single-machine decode floor. Returns tok/s, or None if it won't fit."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, local_files_only=True
        ).to(device).eval()
    except Exception:
        return None
    try:
        ids = tok(_BENCH_PROMPT, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            _sync(device)
            for _ in range(warmup):
                out = model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                _sync(device)
            t0 = time.perf_counter()
            for _ in range(steps):
                out = model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                _sync(device)
            per_token = (time.perf_counter() - t0) / steps
        return 1.0 / per_token
    finally:
        del model
        if device == "mps":
            torch.mps.empty_cache()


async def _time_split(engine, mode: str, warmup: int, steps: int) -> tuple[float, float]:
    strategy = boundary_strategy_for_mode(mode)
    tok_ = engine._tokenizer
    text = tok_.apply_chat_template(
        [{"role": "user", "content": _BENCH_PROMPT}], add_generation_prompt=True, tokenize=False
    )
    prompt_ids = [int(t) for t in tok_(text, add_special_tokens=False)["input_ids"]]
    sid = "bench"
    caches = [(await w.create_qwen_cache(sequence_id=sid)).cache_id for w in engine._worker_clients]

    hidden = engine._embed_payload(prompt_ids)
    t0 = time.perf_counter()
    for w, ch in zip(engine._worker_clients, caches):
        res, _ = await w.prefill_qwen_shard_binary(
            tensor=hidden, cache_id=ch, sequence_id=sid, position_start=0,
            boundary_adapter_strategy=strategy)
        hidden = res.tensor
    tok = engine._next_token(hidden)
    prefill_tok_s = len(prompt_ids) / (time.perf_counter() - t0)
    pos = len(prompt_ids)

    decode_times = []
    for step in range(warmup + steps):
        s = time.perf_counter()
        payload = engine._embed_payload([tok])
        for w, ch in zip(engine._worker_clients, caches):
            res, _ = await w.decode_qwen_shard_binary(
                tensor=payload, cache_id=ch, sequence_id=sid, position_start=pos,
                boundary_adapter_strategy=strategy)
            payload = res.tensor
        tok = engine._next_token(payload)
        pos += 1
        if step >= warmup:
            decode_times.append(time.perf_counter() - s)
    decode_per_token = sum(decode_times) / len(decode_times)
    return prefill_tok_s, 1.0 / decode_per_token


def benchmark(
    model: str,
    hosts: list["Host | str"],
    *,
    mode: str = "relay",
    warmup: int = 6,
    steps: int = 40,
    measure_frontier: bool = True,
    headroom: float = 0.80,
    quiet: bool = False,
    skip_preflight: bool = False,
) -> BenchReport:
    from somatic.cluster.launcher import _make_run_id
    from somatic.serving.cluster_engine import ClusterEngine

    host_objs = coerce_hosts(hosts)
    sup = Supervisor(run_id=_make_run_id(model), model_id=model, precision="bf16",
                     mode=mode, quiet=quiet)
    plan = sup.bring_up_workers(host_objs, headroom=headroom, skip_preflight=skip_preflight)
    try:
        engine = ClusterEngine(model_id=model, workers=sup.worker_specs(plan),
                               boundary_strategy=boundary_strategy_for_mode(mode))
        prefill_ts, decode_ts = asyncio.run(_run(engine, mode, warmup, steps))
    finally:
        sup.teardown()

    frontier = _measure_frontier(model, warmup, steps) if measure_frontier else None
    pct = (100.0 * (1.0 / frontier) / (1.0 / decode_ts)) if frontier else None
    return BenchReport(
        model=model, layout=plan.layout, mode=mode,
        prefill_tok_s=prefill_ts, decode_tok_s=decode_ts,
        decode_ms_per_token=1000.0 / decode_ts,
        frontier_tok_s=frontier, pct_of_frontier=pct,
        warmup=warmup, steps=steps,
    )


async def _run(engine, mode, warmup, steps):
    await engine.start()
    try:
        return await _time_split(engine, mode, warmup, steps)
    finally:
        await engine.close()
