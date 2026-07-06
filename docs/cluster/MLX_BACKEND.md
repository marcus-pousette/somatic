# MLX backend (Apple Silicon) — faster split inference

An alternative compute backend that runs each machine's transformer layers with
Apple's **MLX** instead of PyTorch/MPS. On Apple Silicon it's **measured ~2× faster
than the PyTorch path single-machine and beats llama.cpp**, and it dominates once
split across machines. Same idea as the main product — a model too big for one
machine, split across the machines you have — just a faster engine.

Status: **lab-stage, Apple-only.** It's a separate command (`mlx_run.py`), not yet
folded into `somatic run` (the CLI runs in a Python without MLX). See caveats below.

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

- **Driver** (`somatic/serving/mlx_engine.py`, `MLXClusterEngine`): holds embed / norm /
  lm_head + its own layer range, chains the hidden state through remote workers, samples.
- **Workers** (`scripts/mlx_split_worker.py`): each holds a contiguous layer range, over a
  length-framed **bit-exact bf16** socket (`TCP_NODELAY`).
- **Shard-only loading**: `mlx_lm.load(lazy=True)` — each node materialises only the layers
  it runs (a 14B worker holds ~3 GB, not 30). The rest stay memory-mapped.
- **Serving**: `AsyncMLXEngine` adapts the sync engine to the existing async OpenAI app, so
  the entire OpenAI surface + chat UI is reused unchanged.

## Caveats

- **Apple-only.** MLX is Apple Silicon; CUDA machines would need a different fast backend.
- **Not in `somatic run` yet.** The `somatic` CLI runs in a Python that can't load MLX, so
  this is a separate `mlx_run.py` for now.
- **Not yet: disk-shard loading.** Each node currently needs the full model files on disk
  (only its layers ever load into RAM). `mlx_lm.pipeline_load` can fetch only a node's shards
  — a planned follow-up.
