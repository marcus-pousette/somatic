# Benchmarks — running big LLMs across home machines, honestly

Most tools that split an LLM across machines publish **no reproducible
single-stream tok/s at all.** This page does — and explains, with the roofline,
*why the numbers are what they are*. Run it yourself:

```bash
somatic bench Qwen/Qwen3-1.7B --host localhost --host you@other-machine
```

## The one physics fact that governs everything

Batch-1 decoding is **memory-bandwidth bound**: to generate each token, the
hardware must read *every weight once*. So the floor is:

```
time / token  ≥  model_bytes / memory_bandwidth
```

**Splitting a model across machines does NOT speed up a single stream.** In a
layer-split, machine 2 sits idle while machine 1 works — they run *sequentially*
per token. So per-token time = `total_bytes / one machine's bandwidth`.
**Distributing buys you capacity (run a model too big for one box), not speed.**
Three machines ≈ one machine's decode speed. This is a wall, not a bug — and it's
why a single big-memory machine beats a pool for anything that fits on it.

We call the single-machine floor the **frontier**. `somatic bench` reports what
% of it your split achieves.

## Measured here (Apple Silicon, ~80 GB/s base Mac, bf16, synced, warmed-up)

| model | config | decode tok/s | frontier | % of frontier |
|-------|--------|-------------|----------|---------------|
| Qwen3-1.7B | 1 machine (split runtime) | **17.1** | 27.4 | **62%** |
| Qwen3-1.7B | 2 machines | ~15 | 27.4 | ~55% |
| Qwen3-14B (29.5 GB) | 2 Macs — *doesn't fit either alone* | runs (capacity demo) | — | — |

Two honest takeaways: (1) the split runtime reaches ~55–62% of the physical floor
— there's real overhead (transport + head), but it's ~1.5× the floor, not 10×;
(2) bandwidth is destiny — the same 1.7B floor is ~27 tok/s on this base Mac and
would be ~60 tok/s on a 200 GB/s Mac. You don't engineer past bandwidth.

## ⚠️ These are NOT a head-to-head comparison (yet)

Be clear about what's below. The Somatic row above is **measured** on a small
model (1.7B) on a modest Mac. The rows below are competitors' **reported** figures
for **70B on 3 machines** on *different* hardware. **You cannot directly compare
them** — different models, different machines, different bandwidth.

The only thing that transfers across all of them is the **roofline logic** above
(memory-bandwidth physics is hardware-agnostic). A *real* comparison requires
running Somatic **and** a competitor on the **same model and the same machines** —
that's the next step for this doc (tracked as an open item), not something these
tables establish. Until then, read the rows below as "what others report," and the
Somatic row as "what we measured on our rig."

## The landscape (reported numbers, cited — not measured by us)

Single-stream tok/s on **70B-class** models on comparable consumer hardware,
unless noted. Reported figures from public sources; treat as indicative.

| tool | topology | hardware | reported tok/s | source |
|------|----------|----------|----------------|--------|
| **exo** | pipeline / layer | consumer Macs, LAN | 5–8 (3 nodes, 70B); demo-grade, hangs reported | exo-explore/exo, issue #553, Medium deep-dives |
| **prima.cpp** | pipelined-ring | mixed home cluster | 1.48 (70B); 26 (32B w/ spec-decode) | arXiv 2504.08791, ICLR 2026 |
| **distributed-llama** | tensor-parallel | 4× Raspberry Pi 5 | 13 (Qwen3-30B-A3B, Q4) | b4rtaz/distributed-llama |
| **llama.cpp RPC** | master-worker layer | any, 10GbE | 5.9 (72B decode; 47% loss vs local) | llama.cpp RPC benchmarks |
| **MLX + JACCL** (Apple) | TP over Thunderbolt-5 | 4× M4 Ultra | 26–28 (235B–1T) | Apple WWDC 2026 s233; case studies |
| **Petals** | BitTorrent / WAN | consumer GPUs | 6 (70B) — **dormant since 2023** | bigscience/petals |
| **Shard** (leyten) | pipeline + spec-decode / WAN | prosumer GPUs, multi-state | ~30 (744B) | leyten/shard |

**Single-machine baselines** (the honest alternative to splitting):

| hardware | 70B int4 | note |
|----------|----------|------|
| M4 Max MacBook Pro (546 GB/s) | 20–28 tok/s | *faster than any home pool*, one laptop |
| M3 Ultra Mac Studio (800 GB/s) | 10–16 tok/s | one machine, no setup |
| Cloud API | 50–100+ tok/s | pennies/token |

## What to conclude

- **Somatic isn't the fastest — nothing splitting across a home LAN can be.** The
  point of splitting is running a model that fits *no single machine you own*.
- The honest differentiator here is **a published, reproducible number and the
  roofline that explains it** — which the rest of the category does not provide.
- If a model fits one machine, run it on one machine. If it doesn't, and you have
  several machines and a privacy/offline requirement, splitting at reading speed
  is a real option — and now you can measure exactly what you'll get before you
  commit.

*Numbers here are wall-clock, warmed-up, and synced (MPS/CUDA are asynchronous;
un-synced timers under-report). Re-run `somatic bench` on your own hardware —
that's the whole point.*
