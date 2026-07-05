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

**Splitting does NOT speed up a stream that already fits on one machine.** In a
layer-split, machine 2 sits idle while machine 1 works — they run *sequentially*
per token, so per-token time = `total_bytes / one machine's bandwidth`. Three
machines ≈ one machine's decode speed. For a model that fits comfortably on one
box, **a single big-memory machine beats a pool** — distributing buys capacity,
not speed.

**But that floor assumes the weights fit in fast memory.** When a model is too big
to sit comfortably on one machine, that machine falls *off* the roofline — the GPU
thrashes against its working-set cap and decode craters far below the bandwidth
floor. That's the regime where splitting wins on *speed*, not just capacity — and
it's measured below (14B: one cramped machine 1.43 tok/s, split across two 2.88).

We call the single-machine bandwidth floor the **frontier**. `somatic bench`
reports what % of it your split achieves.

## Measured head-to-head — Somatic vs llama.cpp, on one real rig

The first apples-to-apples comparison: **same models, same two machines, same
precision (F16, 2 bytes/weight), same warmed + synced method**, both runtimes.
Rig: **Mac A** (M3 Pro, 36 GB) + **Mac B** (M1 Pro, 32 GB), over **2.4 GHz WiFi**.
Full raw outputs and commands:
[benchmarks/2026-07-05-head-to-head.md](benchmarks/2026-07-05-head-to-head.md).

| model | runtime | config | decode tok/s |
|-------|---------|--------|-------------:|
| **1.7B** — *fits one machine* | llama.cpp | 1 machine | **30.2** |
| 1.7B | Somatic | 1 machine (split runtime) | 17.1 |
| 1.7B | **llama.cpp** | **2 machines (RPC, 5 GHz)** | **5.47** |
| 1.7B | **Somatic** | **2 machines (5 GHz)** | **7.79** |
| **14B** — *too big for one* | llama.cpp | 1 machine (memory-pressured) | **1.43** |
| 14B | Somatic | 2 machines | **2.88** |
| 14B | llama.cpp | 2 machines (RPC) | not run (~72 min weight upload) |

*(2-machine rows measured with the worker Mac under background load — a remote-desktop
session + a VM — so absolutes are conservative; it hits both runtimes, so the ratio
is what matters.)*

This isn't a clean win — it's a map of where each tool belongs:

1. **If the model fits one machine, use one machine.** llama.cpp's mature Metal
   kernels do 1.7B at 30 tok/s; Somatic's split runtime does 17. Our per-token
   overhead on a single box is real. Splitting a model that already fits is pointless.
2. **But splitting is where the two runtimes diverge sharply.** Going from one
   machine to a 2-machine split, **llama.cpp drops 5.5× (30 → 5.5)** while **Somatic
   drops only 2.2× (17 → 7.8)** — so when both actually split, **Somatic is faster
   (7.8 vs 5.5).** llama.cpp's RPC is a *generic* remote-tensor backend: it
   synchronizes many ops across the network per token. Somatic is *purpose-built*
   for the pipeline — one ~4 KB activation per boundary per token. That gap is the
   real, measured differentiator.
3. **The reason to split at all is a model that won't fit.** On the 36 GB Mac, 14B
   F16 (27.5 GiB vs a ~29.4 GiB Metal cap) falls off the memory cliff — llama.cpp on
   one machine drops to **1.43 tok/s**. Somatic across both machines runs it at
   **2.88** (2×), and models bigger than any one machine become runnable at all.
4. **Loading, too, favors the purpose-built path.** llama.cpp RPC ships every remote
   layer's weights over the network at load; on 2.4 GHz WiFi that stalled entirely,
   and even on 5 GHz the 14B upload is ~72 min. Somatic loads each shard from that
   host's *own disk* — nothing to ship — so a split *starts* immediately on ordinary
   home WiFi.

**Somatic's honest niche:** run a model too big for any one machine you own, over the
home network you already have — where its purpose-built pipeline both *starts* faster
(local shard load) and *runs* faster under a split (2.2× vs 5.5× slowdown) than the
standard tool's generic RPC.

## The wider landscape (others' reported numbers — not measured by us)

Unlike the table above, these are **not** our measurements — they're figures other
projects report for **70B-class** models on their own (different) hardware. Kept
for context; treat as indicative, not comparable to our rig.

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

- **Somatic isn't the fastest runtime.** On anything that fits one machine, a
  mature single-machine engine beats it — measured, 30 vs 17 tok/s at 1.7B. It
  doesn't try to be.
- **Its value shows up exactly when a model won't fit one machine:** the split ran
  14B at 2× a cramped single machine, and shard-local loading *started* a split
  over home WiFi where llama.cpp's weight-shipping RPC couldn't even load. That
  narrow case — a model too big for any one box, on the network you already have —
  is the whole point.
- **And the numbers are published, reproducible, and explained by the roofline**,
  which the rest of the category mostly doesn't provide.

*Numbers here are wall-clock, warmed-up, and synced (MPS/CUDA are asynchronous;
un-synced timers under-report). Re-run `somatic bench` on your own hardware —
that's the whole point.*
