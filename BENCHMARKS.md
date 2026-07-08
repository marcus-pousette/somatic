# Benchmarks — running big LLMs across home machines, honestly

Most tools that split an LLM across machines publish **no reproducible
single-stream tok/s at all.** This page does — and explains, with the roofline,
*why the numbers are what they are*. Run it yourself:

```bash
soup bench Qwen/Qwen3-1.7B --host localhost --host you@other-machine
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

We call the single-machine bandwidth floor the **frontier**. `soup bench`
reports what % of it your split achieves.

## Measured head-to-head — Computer Soup vs llama.cpp, on one real rig

The first apples-to-apples comparison: **same models, same two machines, same
precision (F16, 2 bytes/weight), same warmed + synced method**, both runtimes.
Rig: **Mac A** (M3 Pro, 36 GB) + **Mac B** (M1 Pro, 32 GB), over **2.4 GHz WiFi**.
Full raw outputs and commands:
[benchmarks/2026-07-05-head-to-head.md](benchmarks/2026-07-05-head-to-head.md).

| model | runtime | config | decode tok/s |
|-------|---------|--------|-------------:|
| **1.7B** — *fits one machine* | **Computer Soup (MLX backend)** | 1 machine | **35.2** |
| 1.7B | llama.cpp | 1 machine | 30.2 |
| 1.7B | Computer Soup (PyTorch backend) | 1 machine (split runtime) | 17.1 |
| 1.7B | **Computer Soup (MLX backend)** | **2 machines (5 GHz)** | **13–19** |
| 1.7B | Computer Soup (PyTorch backend) | 2 machines (5 GHz) | 7.79 |
| 1.7B | llama.cpp | 2 machines (RPC, 5 GHz) | 5.47 |
| **14B** — *too big for one* | **Computer Soup (MLX backend)** | **2 machines** | **2.0–4.3** |
| 14B | Computer Soup (PyTorch backend) | 2 machines | 2.88 |
| 14B | llama.cpp | 1 machine (memory-pressured) | 1.43 |
| 14B | llama.cpp | 2 machines (RPC) | not run (~72 min weight upload) |

*(2-machine rows measured on a lived-in home rig — the worker Mac runs a remote-desktop
session + a VM — so absolutes are conservative and vary with background load; MLX split
rows are the measured range across sessions. The load hits every runtime equally, and
same-session ratios are what matter: at 1.7B the MLX split's **worst** run still beat
llama.cpp's RPC split 2.4×. The 14B MLX range is RAM-sensitive — each shard wants
~15 GB resident; when the rig has less free, mmap pages evict and it slides toward
the low end.)*

What the table shows:

1. **The compute engine matters as much as the split.** Computer Soup's original PyTorch/MPS
   backend loses to llama.cpp single-machine (17 vs 30) — its per-token overhead is
   real. The **MLX backend** ([docs/cluster/MLX_BACKEND.md](docs/cluster/MLX_BACKEND.md),
   Apple Silicon) removes it: **35 tok/s single-machine, faster than llama.cpp** — MLX
   runs at the memory-bandwidth floor.
2. **Splitting is where the architectures diverge sharply.** Going from one machine to
   a 2-machine split, **llama.cpp drops 5.5× (30 → 5.5)** while Computer Soup's pipeline
   drops ~2× (MLX: 35 → 13–19; PyTorch: 17 → 7.8). llama.cpp's RPC is a *generic*
   remote-tensor backend that synchronizes many ops across the network per token;
   Computer Soup is *purpose-built* — one ~4 KB activation per boundary per token. Split
   for split, the MLX backend's **worst** measured run beat llama.cpp's RPC 2.4×.
3. **The reason to split at all is a model that won't fit.** On the 36 GB Mac, 14B
   F16 (27.5 GiB vs a ~29.4 GiB Metal cap) falls off the memory cliff — llama.cpp on
   one machine drops to **1.43 tok/s**. Split across both machines it decodes at
   2.0–4.3 tok/s (1.4–3× that, depending on free RAM), and models bigger than any
   one machine become runnable at all.
4. **Loading, too, favors the purpose-built path.** llama.cpp RPC ships every remote
   layer's weights over the network at load; on 2.4 GHz WiFi that stalled entirely,
   and even on 5 GHz the 14B upload is ~72 min. Computer Soup loads each shard from that
   host's *own disk* — nothing to ship — so a split *starts* immediately on ordinary
   home WiFi. (The MLX backend's lazy shard load means a 14B worker holds ~3 GB
   resident, not 30.)

**Computer Soup's honest niche:** run a model too big for any one machine you own, over the
home network you already have — where it starts faster (local shard load), runs faster
under a split (~2× vs 5.5× slowdown), and with the MLX backend is the fastest thing
we measured on Apple Silicon, single-machine or split.

## Speculative decoding — beating the split's own roofline

A layer-split is memory-bandwidth bound: every token pays the full hop chain. **Speculative
decoding** (`--draft`) breaks that — a small same-family draft model on the driver proposes
`k` tokens and **one** multi-token pass through the split target verifies them all, so the
expensive chain runs fewer times for the *same greedy output*. It's the one lever that beats
the per-token roofline, and the home-cluster category (exo) doesn't have it. Measured on a
**14B, three-machine bf16 split** (M3 Pro + M1 Pro + M1 Max, home WiFi, no swap); receipt:
[benchmarks/2026-07-07-spec-decode-3machine.md](benchmarks/2026-07-07-spec-decode-3machine.md):

| Qwen3-14B, 3 machines | decode tok/s | speedup |
|---|---:|---:|
| plain greedy | 3.5–4.4 | — |
| + Qwen3-0.6B draft, k=6 | ~6.0 | ~1.45× |
| + Qwen3-1.7B draft, k=6 | **5.3–6.4** | **1.5–1.6×** |

Output is **byte-identical to plain greedy** (greedy spec-decode only accepts tokens the
target itself emits). It's *verify-bound* — the k+1-token pass costs ~1.6× a single-token
pass — so the win grows with how network-dominated the pass is, and a *stronger* draft
(1.7B) beats a cheaper one (0.6B) by triggering fewer of the expensive verify passes. On a
model that fits one machine, spec-decode is ~a wash (the target pass is too cheap to
amortize); it pays off exactly in the split-a-big-model regime this tool is for.

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

- **Computer Soup isn't the fastest runtime.** On anything that fits one machine, a
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
un-synced timers under-report). Re-run `soup bench` on your own hardware —
that's the whole point.*
