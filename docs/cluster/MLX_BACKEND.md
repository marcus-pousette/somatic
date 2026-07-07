# MLX backend (Apple Silicon) — faster split inference

An alternative compute backend that runs each machine's transformer layers with
Apple's **MLX** instead of PyTorch/MPS. On Apple Silicon it's **measured ~2× faster
than the PyTorch path single-machine and beats llama.cpp**, and it dominates once
split across machines. Same idea as the main product — a model too big for one
machine, split across the machines you have — just a faster engine.

Status: **lab-stage, Apple-only.** It's a separate command (`mlx_run.py`), not yet
folded into `soup run` (the CLI runs in a Python without MLX). See caveats below.

## Measured (2 Macs — M3 Pro 36 GB + M1 Pro 32 GB, 5 GHz WiFi, bf16)

| model | config | decode tok/s |
|-------|--------|-------------:|
| Qwen3-1.7B | 1 machine, MLX | **35** (llama.cpp 30, PyTorch path 17) |
| Qwen3-1.7B | 2 machines, MLX split | **~19** (llama.cpp RPC 5.5, PyTorch split 7.8) |
| Qwen3-14B | 2 machines, MLX split | **~4.2** (PyTorch split 2.9, llama.cpp single 1.4) |

MLX's edge shrinks as the model grows (both converge to the memory-bandwidth floor)
but it wins at every size tested. Full receipts: [../../benchmarks/2026-07-05-head-to-head.md](../../benchmarks/2026-07-05-head-to-head.md).

## Requirements

- Apple Silicon Macs, key-based SSH between them.
- The model in each machine's Hugging Face cache (same as the PyTorch path).

## 1. Provision (once per machine)

Creates `~/mlxenv` (the MLX runtime) on each host. Idempotent — re-running is a fast
"ready ✓".

```bash
python scripts/mlx_provision.py \
    --driver --host you@other-mac=<ip> --identity ~/.ssh/<key>
```

`--driver` sets up this machine (adds the serving deps); each `--host ssh_target=ip`
sets up a remote worker.

## 2. Run

Auto-fits the layer split by each machine's free RAM (proportional to capacity), then
launches, generates, and tears the workers down.

```bash
PYTHONPATH=. ~/mlxenv/bin/python scripts/mlx_run.py \
    --model Qwen/Qwen3-14B \
    --host you@other-mac=<ip> --identity ~/.ssh/<key>
```

Manual split instead of auto-fit: `--local 0:20 --worker you@mac=<ip>:5601:20:40`.

## Speculative decoding (`--draft`) — amortize the network

Split decoding is network-bound: every token pays the full hop chain. With
`--draft`, a small same-family model held whole on the driver proposes `k`
tokens and **one** multi-token pass through the split target verifies them all —
the expensive chain runs ~3× less often for the same greedy output. Each pass
carries more network as machines are added, so the amortization *should* matter
more with more machines — measured here on three; a clean two-machine
comparison is still open.

```bash
PYTHONPATH=. ~/mlxenv/bin/python scripts/mlx_run.py \
    --model Qwen/Qwen3-14B --draft Qwen/Qwen3-1.7B \
    --host you@mac-b=<ip> --host you@mac-c=<ip> --identity ~/.ssh/<key>
```

Measured (3 Macs — M3 Pro 36 GB + M1 Pro 32 GB + M1 Max 32 GB, home WiFi, bf16,
Qwen3-14B split 14|13|13, no swap; ranges across runs — WiFi and background load
move the absolutes):

| Qwen3-14B, 3 machines | decode tok/s | speedup |
|---|---:|---:|
| plain greedy | 3.5–4.4 | — |
| + Qwen3-0.6B draft, k=6 | ~6.0 | ~1.45× |
| + Qwen3-1.7B draft, k=6 | 5.3–6.4 | **1.5–1.6×** |

- The draft must share the target's tokenizer (same family — a Qwen drafts a
  Qwen). The engine refuses mismatched vocabularies at startup.
- Rollback needs a rewindable KV cache (the standard `KVCache` — Qwen, most
  Llama-family models). Sliding-window / SSM caches (gemma-3, gpt-oss, mamba)
  are refused at startup — or fail loudly mid-run if a sliding window wraps —
  rather than silently corrupting output.
- **Exactness**: greedy speculative decoding only accepts tokens the target
  itself emits — measured byte-identical to plain greedy at 14B. On a small
  target (1.7B) the batched verify pass can flip a rare bf16 near-tie (top-2
  logits within ~2 ULPs), giving an equally coherent alternative continuation.
- **Verify-bound**: the k+1-token verify pass costs ~1.6× a single-token pass,
  so k beyond ~6 stops helping, and a *stronger* draft that accepts more per
  pass (1.7B) beats a cheaper one (0.6B) despite costing more to run.
- Splitting a model that fits one machine is still slower than not splitting —
  spec-decode narrows that gap, it doesn't reverse it.

## 3. Serve (OpenAI API + chat UI)

Add `--serve` to turn it into a real server — streaming `/v1/chat/completions`,
`/v1/models`, and the built-in chat page. Point Open WebUI / the `openai` SDK / curl at it.

```bash
PYTHONPATH=. ~/mlxenv/bin/python scripts/mlx_run.py \
    --model Qwen/Qwen3-14B --host you@other-mac=<ip> --identity ~/.ssh/<key> \
    --serve --serve-port 8000
# → chat http://127.0.0.1:8000/   api http://127.0.0.1:8000/v1
```

Workers self-terminate when the driver exits (even a hard kill), so there are no
orphaned processes.

## How it works

- **Driver** (`soup/serving/mlx_engine.py`, `MLXClusterEngine`): holds embed / norm /
  lm_head + its own layer range, chains the hidden state through remote workers, samples.
- **Workers** (`scripts/mlx_split_worker.py`): each holds a contiguous layer range, over a
  length-framed **bit-exact bf16** socket (`TCP_NODELAY`).
- **Shard-only loading**: `mlx_lm.load(lazy=True)` — each node materialises only the layers
  it runs (a 14B worker holds ~3 GB, not 30). The rest stay memory-mapped.
- **Speculative decoding** (`--draft`): the draft loops on-device on the driver, one
  k+1-token verify pass runs through the whole chain, and every KV cache rolls back by
  the reject count — locally via `trim_prompt_cache`, remotely via a 4-byte trim frame
  (unambiguous next to hidden frames, which are multiples of 2·dim bytes). One
  host↔GPU sync per step.
- **Serving**: `AsyncMLXEngine` adapts the sync engine to the existing async OpenAI app, so
  the entire OpenAI surface + chat UI is reused unchanged.

## Caveats

- **Apple-only.** MLX is Apple Silicon; CUDA machines would need a different fast backend.
- **Not in `soup run` yet.** The `soup` CLI runs in a Python that can't load MLX, so
  this is a separate `mlx_run.py` for now.
- **Not yet: disk-shard loading.** Each node currently needs the full model files on disk
  (only its layers ever load into RAM). `mlx_lm.pipeline_load` can fetch only a node's shards
  — a planned follow-up.
