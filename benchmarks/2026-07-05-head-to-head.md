# Head-to-head receipt — Somatic vs llama.cpp, 2026-07-05

First real apples-to-apples run: **same models, same two machines, same precision
(F16 = 2 bytes/weight), same warmed+synced methodology**, both runtimes. Raw
outputs below so the numbers in [../BENCHMARKS.md](../BENCHMARKS.md) are auditable.

## Rig

| host | chip | RAM | Metal working-set cap | role |
|------|------|-----|----------------------|------|
| Mac A | Apple M3 Pro | 36 GB | 30150 MB (~29.4 GiB) | driver + worker |
| Mac B | Apple M1 Pro | 32 GB | ~25.5 GiB free | worker |

Network: **802.11n, 2.4 GHz, 20 MHz** (`system_profiler SPAirPortDataType`).
Mac A→B ping: `min/avg/max/stddev = 52/70/107/22 ms`, 0% loss. Bulk throughput
measured ~3–4 MB/s. Effective decode bandwidth on Mac A (from the 1.7B frontier,
27.4 tok/s × 3.4 GB): ~93 GB/s.

## llama.cpp (build 2da6686, Metal, `llama-bench`, F16 GGUF converted from the HF weights)

```
# 1.7B single machine (Mac A)
qwen3 1.7B F16 | 3.78 GiB | 2.03 B | MTL,BLAS | pp128 1427.05 ± 9.48 | tg128 30.17 ± 0.29

# 1.7B split over loopback RPC (Mac A local rpc-server, no network)  -ts 14,14
qwen3 1.7B F16 | 3.78 GiB | 2.03 B | MTL,BLAS,RPC | pp64 1274 | tg64 34.6

# 14B single machine (Mac A) — 27.51 GiB model vs 29.4 GiB Metal cap → memory-pressured
qwen3 14B F16 | 27.51 GiB | 14.77 B | MTL,BLAS | pp128 112.60 ± 37.32 | tg128 1.43 ± 0.15

# 14B split over RPC to Mac B (2.4 GHz WiFi)  -ts 16,24
FAILED TO LOAD — weight upload to Mac B stalled (RSS flat ~6 GB, never reached ~17 GB);
rpc-server churned accept/close. 3 attempts (14B, 1.7B, 1.7B-patient) all stalled.
Loopback RPC works instantly → cause is the WiFi link, not the build.
```

## Somatic (bf16, `somatic bench`, warmed + synced)

```
# 1.7B, 1 machine (split runtime) : 17.1 tok/s   (62% of 27.4 frontier)
# 1.7B, 2 machines               : ~15 tok/s
# 14B,  2 machines (Mac A [0,16) + Mac B [16,40)) : 2.88 tok/s (347 ms/token), prefill ~3 tok/s
```

## What each comparison shows

- **Model that fits one machine (1.7B):** llama.cpp single = **30.2**, Somatic split
  runtime = **17.1**. llama.cpp's mature Metal kernels beat our split runtime by ~1.75×.
  If a model fits one machine, run it on one machine.
- **Model too big for one machine (14B):** on Mac A alone llama.cpp falls off the
  memory-bandwidth cliff to **1.43** (27.5 GiB against a 29.4 GiB cap). Somatic split
  across both machines = **2.88**, ~2× faster — and splitting is the only way to run
  models larger than any single machine at all.
- **Split-vs-split at 14B (the truly fair test): not measurable here.** llama.cpp RPC
  ships every remote layer's weights over the network at load; our 2.4 GHz WiFi
  couldn't sustain it. Somatic loads each shard from that host's own disk, so it never
  makes that transfer — a real practical edge on home networks, **not** a claim that
  Somatic's compute is faster than llama.cpp's (the 1.7B numbers say it isn't).

## Update 2026-07-06 — 5 GHz retry (network fixed)

Moved both Macs to the same **5 GHz** channel (Ch 44); latency 52–107 ms → **6 ms**,
0% loss. Bulk throughput still ~3.9 MB/s (home WiFi cap), but stable latency fixed the
RPC handshake that stalled on 2.4 GHz. Worker Mac (M1 Pro) was under background load
(Chrome Remote Desktop + a VM + the Claude app, load avg ~6) — depresses absolutes on
*both* runtimes equally, so the ratio is the signal.

```
# llama.cpp 1.7B split over RPC to Mac B (5 GHz), -ts 14,14 — LOADED + RAN this time
qwen3 1.7B F16 | RPC | pp64 ~218 | tg64  5.47   (10m21s wall; ~1.9 GB upload dominated)

# Somatic 1.7B split across the same 2 Macs (5 GHz), relay mode, same session
Somatic 1.7B  [0,14)@Mac A -> [14,28)@Mac B :  decode 7.79 tok/s
```

Single → 2-machine-split slowdown (the load-robust signature):
- **llama.cpp: 30.2 → 5.47  = 5.5× slower** (generic remote-tensor RPC, heavy per-token sync)
- **Somatic:   17.1 → 7.79  = 2.2× slower** (purpose-built pipeline, ~4 KB/boundary/token)

So split-vs-split at 1.7B, same conditions: **Somatic 7.79 vs llama.cpp 5.47 (~1.4×).**
14B split-vs-split still not run (llama.cpp would need ~72 min to upload ~17 GB at this
WiFi's bulk rate).

## Reproduce

```bash
# llama.cpp
python llama.cpp/convert_hf_to_gguf.py <hf_snapshot> --outfile m.gguf --outtype f16
llama.cpp/build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052            # on Mac B
llama.cpp/build/bin/llama-bench -m m.gguf -rpc <macB>:50052 -ngl 99 -ts 16,24 -p 128 -n 128 -r 3
# Somatic
somatic bench Qwen/Qwen3-14B --host localhost --host you@macB
```
