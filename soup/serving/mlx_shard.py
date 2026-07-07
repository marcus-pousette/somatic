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
        from mlx_lm import load

        self.model_id = model_id
        self.model, self.tokenizer = load(model_id, lazy=True)
        inner = self.model.model
        self.dim = int(inner.embed_tokens.weight.shape[1])
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
