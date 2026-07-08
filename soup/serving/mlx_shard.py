"""Shared MLX shard primitives for the MLX backend (Apple Silicon).

The MLX backend runs each node's transformer layers with Apple's MLX instead of
PyTorch/MPS — measured ~2× faster single-machine and it *beats* llama.cpp, while
`mlx_lm.load(lazy=True)` gives shard-only loading for free (a node materialises
only the layers it actually runs; the rest stay mmap'd and never hit RAM).

This module is import-guarded: it pulls in `mlx`/`mlx_lm`, which are optional and
Apple-only, so nothing here is imported unless the MLX backend is selected.

Wire format between driver and worker is a length-framed bf16 hidden-state tensor
(`[4-byte big-endian length][payload]`, bit-exact); a zero-length frame is a reset,
and a 4-byte frame is a KV-cache trim (speculative-decoding rollback; unambiguous
because a hidden frame is a multiple of 2*dim bytes and dim is always > 2).
"""

from __future__ import annotations

import socket
import struct

import numpy as np


# ----- length-framed socket transport -------------------------------------------------

def send_frame(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> bytes | None:
    """Return the next frame's payload bytes, ``b''`` for a reset, ``None`` on close."""
    header = _recv_exactly(sock, 4)
    if header is None:
        return None
    (n,) = struct.unpack(">I", header)
    return b"" if n == 0 else _recv_exactly(sock, n)


def set_nodelay(sock: socket.socket) -> None:
    # Small per-token frames must not wait on Nagle's algorithm.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


# ----- hidden-state (de)serialisation -------------------------------------------------

def hidden_to_bytes(hidden) -> bytes:
    """MLX bf16 (1, L, dim) -> contiguous bytes (2*dim per token), *exact*.

    Reinterpret the bf16 bits as uint16 (no dtype conversion) — same 2 bytes/elem
    as fp16 but lossless, so the boundary is bit-exact (fp16 would clip bf16's
    wider exponent range). Also skips the fp16 conversion kernel each hop.
    """
    import mlx.core as mx

    bits = hidden.astype(mx.bfloat16).view(mx.uint16)
    mx.eval(bits)
    return np.ascontiguousarray(np.array(bits)).tobytes()


def bytes_to_hidden(payload: bytes, dim: int):
    """Exact bf16 bytes (uint16 bit-pattern) -> MLX bf16 (1, L, dim)."""
    import mlx.core as mx

    seq = len(payload) // (dim * 2)
    arr = np.frombuffer(payload, dtype=np.uint16).reshape(1, seq, dim)
    return mx.array(arr).view(mx.bfloat16)


# ----- lazy sharded model -------------------------------------------------------------

class ShardModel:
    """A lazily-loaded MLX model that materialises only the parts it is asked to run.

    Load the whole model with ``lazy=True`` (weights are mmap-backed, not resident),
    then use only ``embed``/``norm``/``head`` (driver) or ``run_layers`` over a range
    (worker). Unused layers are never evaluated, so they never become resident.
    """

    def __init__(self, model_id: str):
        import mlx.core as mx
        from mlx_lm import load

        self.model_id = model_id
        self.model, self.tokenizer = load(model_id, lazy=True)
        inner = self.model.model
        # The wire carries activations, so dim must be the ACTIVATION width. Do
        # not read embed_tokens.weight.shape[1]: for a quantized model that weight
        # is bit-packed (e.g. 8-bit packs 4 values per uint32), so its shape is
        # dim / packing-factor. Measure the true width from a one-token embed.
        self.dim = int(inner.embed_tokens(mx.array([[0]])).shape[-1])
        self.n_layers = len(inner.layers)

    # -- driver head --------------------------------------------------------------
    def embed(self, token_ids: list[int]):
        import mlx.core as mx

        return self.model.model.embed_tokens(mx.array([token_ids]))

    def head_logits(self, hidden):
        """Final norm + (tied or untied) lm_head on the last position -> logits row."""
        inner = self.model.model
        h = inner.norm(hidden[:, -1:, :])
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head(h)[0, -1]
        return inner.embed_tokens.as_linear(h)[0, -1]

    def argmax_token(self, hidden) -> int:
        import mlx.core as mx

        return int(mx.argmax(self.head_logits(hidden)).item())

    # -- verify pass (speculative decoding) -----------------------------------------
    def embed_array(self, ids):
        """Embed a (1, L) MLX id array without a host round-trip."""
        return self.model.model.embed_tokens(ids)

    def argmax_tokens(self, hidden):
        """Final norm + lm_head over ALL positions -> (L,) argmax ids, on-device."""
        import mlx.core as mx

        inner = self.model.model
        h = inner.norm(hidden)
        if hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(h)
        else:
            logits = inner.embed_tokens.as_linear(h)
        return mx.argmax(logits[0], axis=-1)

    # -- worker layer range -------------------------------------------------------
    def make_cache(self, start: int, end: int):
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(self.model)[start:end]

    def run_layers(self, hidden, cache, start: int, end: int):
        from mlx_lm.models import base

        layers = self.model.model.layers[start:end]
        mask = base.create_attention_mask(hidden, cache)
        for layer, c in zip(layers, cache):
            hidden = layer(hidden, mask, c)
        return hidden


# ----- tree speculation: control frames, tree attention, cache compaction -------------
#
# Control frames ride the same length-framed socket as hidden frames. Hidden frames
# are always an EVEN number of bytes (2 bytes/element), so control frames are padded
# to an ODD length — the two can never be confused — and carry a 4-byte magic.

TREE_MAGIC = b"TREE"  # tree-verify metadata; the next frame is the tree's hidden states
COMP_MAGIC = b"COMP"  # compaction: keep these tree-local indices, drop the rest


def _pad_odd(payload: bytes) -> bytes:
    return payload if len(payload) % 2 == 1 else payload + b"\x00"


def encode_tree_meta(depths: list[int], parents: list[int]) -> bytes:
    n = len(depths)
    out = TREE_MAGIC + struct.pack(">I", n)
    out += b"".join(struct.pack(">Hi", d, p) for d, p in zip(depths, parents))
    return _pad_odd(out)


def encode_compact(indices: list[int]) -> bytes:
    out = COMP_MAGIC + struct.pack(">I", len(indices))
    out += b"".join(struct.pack(">H", i) for i in indices)
    return _pad_odd(out)


def decode_control(frame: bytes):
    """Return ("tree", depths, parents) / ("compact", indices) / None."""
    if len(frame) % 2 == 0 or len(frame) < 8:
        return None
    magic, n = frame[:4], struct.unpack(">I", frame[4:8])[0]
    if magic == TREE_MAGIC:
        depths, parents = [], []
        for i in range(n):
            d, p = struct.unpack_from(">Hi", frame, 8 + 6 * i)
            depths.append(d)
            parents.append(p)
        return ("tree", depths, parents)
    if magic == COMP_MAGIC:
        return ("compact", list(struct.unpack_from(f">{n}H", frame, 8)))
    return None


def ancestor_mask(parents: list[int]):
    """(N, N) bool rows: node i attends to node j iff j is an ancestor-or-self of i."""
    n = len(parents)
    rows = [[False] * n for _ in range(n)]
    for i in range(n):
        j = i
        while j != -1:
            rows[i][j] = True
            j = parents[j]
    return rows


def tree_attention(attn, x, rope_offsets, mask_rows, cache):
    """One attention layer over a block of tree tokens.

    rope_offsets: per-token ABSOLUTE rope position (applied per equal-offset
    group — mx.fast.rope only takes a scalar). mask_rows: (L, cache_len_after_
    update) bool mx array — row i is True where token i may attend. Requires a
    Llama/Qwen-shaped attention module (q/k/v/o proj + rope; q/k_norm optional).
    """
    import mlx.core as mx
    from mlx_lm.models.base import scaled_dot_product_attention

    B, L, _ = x.shape
    q = attn.q_proj(x).reshape(B, L, attn.n_heads, -1)
    k = attn.k_proj(x).reshape(B, L, attn.n_kv_heads, -1)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

    # rope(x, offset=k) assigns CONSECUTIVE positions k, k+1, ... along the
    # sequence axis — so tokens can only share a call if their positions form a
    # consecutive run in index order. Same-depth siblings do NOT (they need the
    # SAME position), so split into maximal consecutive runs (a chain is one
    # run; branched trees get one call per run).
    q_out = mx.zeros_like(q)
    k_out = mx.zeros_like(k)
    i = 0
    while i < L:
        j = i + 1
        while j < L and rope_offsets[j] == rope_offsets[j - 1] + 1:
            j += 1
        q_out[:, :, i:j, :] = attn.rope(q[:, :, i:j, :], offset=rope_offsets[i])
        k_out[:, :, i:j, :] = attn.rope(k[:, :, i:j, :], offset=rope_offsets[i])
        i = j
    k, v = cache.update_and_fetch(k_out, v)

    add_mask = mx.where(mask_rows, 0.0, -mx.inf).astype(q_out.dtype)
    o = scaled_dot_product_attention(
        q_out, k, v, cache=cache, scale=attn.scale, mask=add_mask
    )
    return attn.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


def tree_block_inputs(depths: list[int], parents: list[int], base_offset: int):
    """(rope_offsets, mask_rows builder input) for a SELF-CONTAINED tree block:
    every node attends to the whole committed prefix [0, base_offset) plus its
    ancestors-and-self inside the block."""
    import mlx.core as mx

    rows = ancestor_mask(parents)
    n = len(parents)
    if base_offset:
        prefix = mx.ones((n, base_offset), dtype=mx.bool_)
        mask_rows = mx.concatenate([prefix, mx.array(rows)], axis=1)
    else:
        mask_rows = mx.array(rows)
    return [base_offset + d for d in depths], mask_rows


def run_layers_tree(model, hidden, cache, start: int, end: int,
                    depths: list[int], parents: list[int]):
    """Tree counterpart of ShardModel.run_layers over layers [start, end) for a
    self-contained tree block (worker side). Cache offsets = committed length."""
    layers = model.model.layers[start:end]
    if not layers:
        return hidden
    # identical for every layer (all caches sit at the same committed offset)
    offs, mask_rows = tree_block_inputs(depths, parents, cache[0].offset)
    for layer, c in zip(layers, cache):
        x = hidden
        a = tree_attention(layer.self_attn, layer.input_layernorm(x), offs, mask_rows, c)
        h = x + a
        hidden = h + layer.mlp(layer.post_attention_layernorm(h))
    return hidden


def compact_cache_block(cache, block_start: int, keep: list[int]) -> None:
    """Gather tree-block columns `keep` (tree-local) to [block_start, ...) and trim."""
    import mlx.core as mx

    if not keep:
        for c in cache:
            c.offset = block_start
        return
    cols = mx.array([block_start + i for i in keep])
    m = len(keep)
    for c in cache:
        k = mx.take(c.keys, cols, axis=2)
        v = mx.take(c.values, cols, axis=2)
        # Materialise the gather BEFORE the overlapping write-back: the lazy
        # take reads the same buffer the slice-assign may update in place
        # (donation), and non-contiguous `keep` makes that a real read/write
        # race — silent KV corruption that only surfaces tokens later.
        mx.eval(k, v)
        c.keys[..., block_start:block_start + m, :] = k
        c.values[..., block_start:block_start + m, :] = v
        c.offset = block_start + m


def trim_cache(cache, n: int) -> None:
    """Roll a KV cache back by ``n`` positions (speculative-decoding reject path).

    Loud on failure: sliding-window / SSM caches can't rewind, and mlx_lm signals
    that by returning a short trim count instead of raising. A silent under-trim
    would corrupt every subsequent token, so refuse instead.
    """
    from mlx_lm.models.cache import trim_prompt_cache

    if n <= 0 or not cache:
        return
    trimmed = trim_prompt_cache(cache, n)
    if trimmed != n:
        raise RuntimeError(
            f"KV cache rollback failed: needed {n}, trimmed {trimmed} — this "
            "model's cache type can't rewind (sliding-window/SSM), so it cannot "
            "run speculative decoding"
        )
