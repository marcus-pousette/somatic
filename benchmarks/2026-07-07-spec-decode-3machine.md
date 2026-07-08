# Speculative decoding receipt — Computer Soup, 3 machines, 2026-07-07

Distributed **speculative decoding** (`soup`'s MLX backend, `--draft`): a small
same-family draft model runs whole on the driver and proposes `k` tokens; **one**
multi-token pass through the split target verifies them all, so the expensive
cross-machine chain runs fewer times for the *same greedy output*. This is the
lever a plain layer-split can't pull — and the one the rest of the home-cluster
category (exo) doesn't have. Raw numbers below back the summary in
[../BENCHMARKS.md](../BENCHMARKS.md).

## Rig

| host | chip | RAM | role |
|------|------|-----|------|
| Mac A | Apple M3 Pro | 36 GB | driver (embed/head + its layer range + draft model) |
| Mac B | Apple M1 Pro | 32 GB | worker |
| Mac C | Apple M1 Max | 32 GB | worker |

Network: home WiFi (5 GHz). Target **Qwen3-14B, bf16**, split **14 | 13 | 13**
(driver + two workers), each node ~9 GB resident so **no swap**. Draft models:
Qwen3-0.6B and Qwen3-1.7B (bf16), same tokenizer family as the target.

## Measured (warmed, greedy)

```
# 14B, 3 machines, plain greedy baseline
plain greedy                        : 3.5–4.4 tok/s

# + speculative decoding, k=6
+ Qwen3-0.6B draft, k=6             : ~6.0 tok/s   (~1.45×)
+ Qwen3-1.7B draft, k=6            : 5.3–6.4 tok/s (1.5–1.6×)

# production engine (MLXClusterEngine), same 14B split, 1.7B draft, one clean session
plain 3.49  ->  spec k=4 5.27 (1.51×)  /  spec k=6 5.33 (1.53×)   [byte-exact both]
```

## What it shows

- **Speedup 1.5–1.6× on a 14B three-machine split, byte-identical output.** Greedy
  speculative decoding only ever accepts tokens the target itself would emit, so the
  result is the same token stream as plain greedy (verified byte-exact at 14B).
- **A stronger draft wins.** The 1.7B draft (higher acceptance, ~4.3 accepted tokens
  per verify pass) beats the 0.6B draft (~3.2/pass) despite costing more to run,
  because it triggers fewer of the expensive verify passes. `k≈6` is the plateau.
- **It's verify-bound, not draft-bound.** Step decomposition: draft 67–195 ms, the
  distributed verify pass ~400 ms (≈1.6× a single-token pass, since it pushes k+1
  tokens through two workers). That verify cost is the ceiling on this rig — which is
  also why the win *grows* with how network-dominated the pass is.
- **On a small target it's a wash, correctly.** 1.7B target single-machine spec-decode
  ≈ 0.95× — the target pass is too cheap to amortize the draft cost. Spec-decode pays
  off exactly when the target pass is expensive (big model, split across machines).

## Reproduce

```bash
PYTHONPATH=. ~/mlxenv/bin/python scripts/mlx_run.py \
    --model Qwen/Qwen3-14B --draft Qwen/Qwen3-1.7B --num-draft 6 \
    --host you@mac-b=<ip> --host you@mac-c=<ip> --identity ~/.ssh/<key>
```

Rare caveat: on a *small* target (1.7B), the batched verify pass can flip a bf16
near-tie (top-2 logits within ~2 ULPs) into an equally coherent alternative token —
the same numerical near-tie that affects any batched-vs-sequential forward, not a
cache bug. At 14B the runs were byte-identical.
