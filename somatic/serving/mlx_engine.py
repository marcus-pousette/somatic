"""MLX backend engine — the driver side of an MLX split cluster.

Same shape as `ClusterEngine`: the driver holds the head (embed / norm / lm_head),
an ordered partition of the transformer layers is spread across machines, and to
generate we embed the prompt, chain the hidden state through every layer range in
order, apply norm + lm_head, and greedily sample — one token at a time.

Difference from the PyTorch `ClusterEngine`: compute is MLX (measured ~2× faster
per node, and it beats llama.cpp), the driver runs its *own* first layer range
in-process (no socket hop), and remote ranges are reached over a length-framed
fp16 socket. `mlx_lm.load(lazy=True)` means each node materialises only the layers
it runs — shard-only loading, for free.

One generation at a time (home cluster); workers hold per-sequence KV caches.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Callable, Sequence

from somatic.serving import mlx_shard as ms


@dataclass(frozen=True)
class RemoteRange:
    """A remote machine holding transformer layers [layer_start, layer_end)."""

    host: str
    port: int
    layer_start: int
    layer_end: int


class MLXClusterEngine:
    def __init__(
        self,
        *,
        model_id: str,
        remote_workers: Sequence[RemoteRange],
        local_range: tuple[int, int] = (0, 0),
    ) -> None:
        # local_range = layers the driver runs in-process (usually (0, k)); the
        # remote_workers cover the rest, contiguous and in order.
        self.model_id = model_id
        self.local_start, self.local_end = local_range
        self.remote_workers = sorted(remote_workers, key=lambda w: w.layer_start)
        self.shard: ms.ShardModel | None = None
        self._sockets: list[socket.socket] = []
        self._eos_ids: set[int] = set()
        self._turn = 0

    @property
    def layout(self) -> str:
        parts = []
        if self.local_end > self.local_start:
            parts.append(f"[{self.local_start},{self.local_end})@driver")
        parts += [
            f"[{w.layer_start},{w.layer_end})@{w.host}:{w.port}" for w in self.remote_workers
        ]
        return " -> ".join(parts)

    def start(self) -> None:
        self.shard = ms.ShardModel(self.model_id)
        self._eos_ids = self._resolve_eos_ids()
        for w in self.remote_workers:
            sock = socket.create_connection((w.host, w.port), timeout=600.0)
            ms.set_nodelay(sock)
            self._sockets.append(sock)

    def close(self) -> None:
        for sock in self._sockets:
            try:
                sock.close()
            except OSError:
                pass
        self._sockets.clear()

    def _resolve_eos_ids(self) -> set[int]:
        tok = self.shard.tokenizer
        eos: set[int] = set()
        tid = getattr(tok, "eos_token_id", None)
        if isinstance(tid, int):
            eos.add(tid)
        for name in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>"):
            try:
                got = tok.convert_tokens_to_ids(name)
            except Exception:
                got = None
            if isinstance(got, int) and got >= 0:
                eos.add(got)
        return eos

    # -- one hop to a remote layer range ------------------------------------------
    def _remote(self, sock: socket.socket, hidden):
        ms.send_frame(sock, ms.hidden_to_bytes(hidden))
        resp = ms.recv_frame(sock)
        if resp is None:
            raise ConnectionError("worker closed the connection mid-generation")
        return ms.bytes_to_hidden(resp, self.shard.dim)

    def _reset_remotes(self) -> None:
        for sock in self._sockets:
            ms.send_frame(sock, b"")
            ms.recv_frame(sock)

    def _forward(self, hidden, local_cache) -> int:
        """Embed-output hidden -> chain local + remote ranges -> next token id."""
        if self.local_end > self.local_start:
            hidden = self.shard.run_layers(
                hidden, local_cache, self.local_start, self.local_end
            )
        for sock in self._sockets:
            hidden = self._remote(sock, hidden)
        return self.shard.argmax_token(hidden)

    def generate_ids(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        on_token: Callable[[str], None] | None = None,
    ) -> list[int]:
        self._turn += 1
        text = self.shard.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt_ids = [int(t) for t in self.shard.tokenizer.encode(text)]

        local_cache = self.shard.make_cache(self.local_start, self.local_end)
        self._reset_remotes()

        # Prefill the whole prompt through the chain, then the decode loop.
        token = self._forward(self.shard.embed(prompt_ids), local_cache)
        generated = [token]
        printed = ""
        for _ in range(max_new_tokens - 1):
            if generated[-1] in self._eos_ids:
                break
            token = self._forward(self.shard.embed([generated[-1]]), local_cache)
            generated.append(token)
            if on_token is not None:
                visible = [t for t in generated if t not in self._eos_ids]
                full = self.shard.tokenizer.decode(visible)
                if len(full) > len(printed):
                    on_token(full[len(printed):])
                    printed = full
        return [t for t in generated if t not in self._eos_ids]

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        ids = self.generate_ids(messages, **kwargs)
        return self.shard.tokenizer.decode(ids)


class AsyncMLXEngine:
    """Async adapter so the sync `MLXClusterEngine` can back the async OpenAI app.

    The engine uses blocking sockets and MLX, so it runs on a single dedicated
    worker thread (start / generate / close all on the same thread, so the model
    and sockets are only ever touched from there). Generations are serialised
    (one at a time — a home cluster, and workers hold one KV cache each), and the
    sync per-token callback is marshalled back onto the event loop for streaming.
    This exposes exactly the interface `build_openai_app` expects, so the whole
    OpenAI surface + chat UI is reused unchanged.
    """

    def __init__(self, engine: MLXClusterEngine) -> None:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        self._engine = engine
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()

    @property
    def layout(self) -> str:
        return self._engine.layout

    async def start(self) -> None:
        import asyncio

        await asyncio.get_running_loop().run_in_executor(self._pool, self._engine.start)

    async def close(self) -> None:
        import asyncio

        await asyncio.get_running_loop().run_in_executor(self._pool, self._engine.close)
        self._pool.shutdown(wait=False)

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        on_token=None,
    ) -> str:
        import asyncio

        loop = asyncio.get_running_loop()

        sync_cb = None
        if on_token is not None:
            def sync_cb(delta: str) -> None:
                # Called on the worker thread; hand the delta to the event loop.
                asyncio.run_coroutine_threadsafe(on_token(delta), loop).result()

        def work() -> str:
            ids = self._engine.generate_ids(
                messages, max_new_tokens=max_new_tokens, on_token=sync_cb
            )
            return self._engine.shard.tokenizer.decode(ids)

        async with self._lock:  # one generation at a time
            return await loop.run_in_executor(self._pool, work)
