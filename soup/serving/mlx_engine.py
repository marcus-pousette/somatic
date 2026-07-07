"""MLX backend engine — the driver side of an MLX split cluster.

Same shape as `ClusterEngine`: the driver holds the head (embed / norm / lm_head),
an ordered partition of the transformer layers is spread across machines, and to
generate we embed the prompt, chain the hidden state through every layer range in
order, apply norm + lm_head, and greedily sample — one token at a time.

Difference from the PyTorch `ClusterEngine`: compute is MLX (measured ~2× faster
per node, and it beats llama.cpp), the driver runs its *own* first layer range
in-process (no socket hop), and remote ranges are reached over a length-framed
bit-exact bf16 socket. `mlx_lm.load(lazy=True)` means each node materialises only the layers
it runs — shard-only loading, for free.

One generation at a time (home cluster); workers hold per-sequence KV caches.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Callable, Sequence

from soup.serving import mlx_shard as ms


@dataclass(frozen=True)
class RemoteRange:
    """A remote machine holding transformer layers [layer_start, layer_end)."""

    host: str
    port: int
    layer_start: int
    layer_end: int


def spec_trims(k: int, n_acc: int) -> tuple[int, int, bool]:
    """Cache bookkeeping after a verify pass accepted ``n_acc`` of ``k`` drafts.

    Returns ``(target_trim, draft_trim, draft_catch_up)``. The target consumed
    ``k + 1`` tokens (current + k drafts) but only ``n_acc + 1`` were kept, so it
    rolls back ``k - n_acc``. The draft consumed current + the first ``k - 1``
    drafts; on a reject it rolls back to just past the last accepted token, and
    on a full accept it must still consume the final draft token to catch up.
    """
    if not 0 <= n_acc <= k:
        raise ValueError(f"n_acc must be in [0, {k}], got {n_acc}")
    if n_acc < k:
        return k - n_acc, max(k - n_acc - 1, 0), False
    return 0, 0, True


class MLXClusterEngine:
    def __init__(
        self,
        *,
        model_id: str,
        remote_workers: Sequence[RemoteRange],
        local_range: tuple[int, int] = (0, 0),
        draft_model_id: str | None = None,
        num_draft: int = 6,
    ) -> None:
        # local_range = layers the driver runs in-process (usually (0, k)); the
        # remote_workers cover the rest, contiguous and in order. A draft model
        # (same family/tokenizer, runs whole on the driver) turns generation into
        # greedy speculative decoding — same tokens as plain greedy, fewer chain
        # passes (exactness scope: docs/cluster/MLX_BACKEND.md).
        if num_draft < 1:
            raise ValueError(f"num_draft must be >= 1, got {num_draft}")
        self.model_id = model_id
        self.local_start, self.local_end = local_range
        self.remote_workers = sorted(remote_workers, key=lambda w: w.layer_start)
        self.draft_model_id = draft_model_id
        self.num_draft = num_draft
        self.shard: ms.ShardModel | None = None
        self._draft = None
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
        if self.draft_model_id:
            from mlx_lm import load

            self._draft, _ = load(self.draft_model_id)  # small, runs whole on the driver
            target_vocab = int(self.shard.model.model.embed_tokens.weight.shape[0])
            draft_vocab = int(self._draft.model.embed_tokens.weight.shape[0])
            if draft_vocab != target_vocab:
                raise ValueError(
                    f"draft {self.draft_model_id} (vocab {draft_vocab}) is not "
                    f"token-compatible with {self.model_id} (vocab {target_vocab}) — "
                    "the draft must share the target's tokenizer (same model family)"
                )
            # Rollback needs rewindable caches; SSM/never-trimmable ones fail here
            # instead of corrupting output later. (Sliding-window caches pass while
            # unwrapped — trim_cache still refuses loudly if one wraps mid-run.)
            from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache

            for who, model, mid in (
                ("target", self.shard.model, self.model_id),
                ("draft", self._draft, self.draft_model_id),
            ):
                if not can_trim_prompt_cache(make_prompt_cache(model)):
                    raise ValueError(
                        f"{who} {mid}: its KV cache can't roll back "
                        "(sliding-window/SSM) — speculative decoding unsupported"
                    )
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

    def _reset_remotes(self) -> None:
        for sock in self._sockets:
            ms.send_frame(sock, b"")
            ms.recv_frame(sock)

    def _chain(self, hidden, local_cache):
        """Embed-output hidden -> local range + every remote range, in order.

        Between remote hops the payload is relayed as raw bytes — worker N's
        output IS worker N+1's input, and the bf16 wire format is a lossless
        bit-pattern, so decoding to an array in between is pure waste.
        """
        if self.local_end > self.local_start:
            hidden = self.shard.run_layers(
                hidden, local_cache, self.local_start, self.local_end
            )
        if not self._sockets:
            return hidden
        payload = ms.hidden_to_bytes(hidden)
        token_bytes = 2 * self.shard.dim
        for sock in self._sockets:
            ms.send_frame(sock, payload)
            payload = ms.recv_frame(sock)
            if payload is None:
                raise ConnectionError("worker closed the connection mid-generation")
            if not payload or len(payload) % token_bytes:
                # Never relay a bad frame onward: b'' would reset every
                # downstream cache and a ragged length would kill the next
                # worker's process — fail loudly at the hop that produced it.
                raise ConnectionError(
                    f"worker returned a malformed frame ({len(payload)} bytes, "
                    f"expected a positive multiple of {token_bytes})"
                )
        return ms.bytes_to_hidden(payload, self.shard.dim)

    def _forward(self, hidden, local_cache) -> int:
        """Embed-output hidden -> chain local + remote ranges -> next token id."""
        return self.shard.argmax_token(self._chain(hidden, local_cache))

    def _trim(self, n: int, local_cache) -> None:
        """Roll every KV cache in the chain back by ``n`` (reject path).

        Fire-and-forget to the workers — no reply frame, and TCP ordering
        guarantees the trim lands before the next hidden frame.
        """
        if n <= 0:
            return
        ms.trim_cache(local_cache, n)
        for sock in self._sockets:
            ms.send_frame(sock, struct.pack(">I", n))

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

        if self._draft is not None:
            generated = self._spec_generate(
                prompt_ids, local_cache, max_new_tokens, on_token
            )
            return [t for t in generated if t not in self._eos_ids]

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

    def _spec_generate(
        self,
        prompt_ids: list[int],
        local_cache,
        max_new_tokens: int,
        on_token: Callable[[str], None] | None,
    ) -> list[int]:
        """Greedy speculative decoding — accepts only the target's own argmax picks.

        Each step: the local draft proposes ``k`` tokens (all on-device), ONE
        multi-token pass through the split target verifies current + k drafts,
        the matching prefix is accepted, and every KV cache rolls back by the
        reject count. The only host<->GPU sync per step is the accept count.
        Measured byte-identical to the plain loop at 14B; on small targets the
        batched verify pass can flip a rare bf16 near-tie (docs/cluster/
        MLX_BACKEND.md has the exactness scope).
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        k = self.num_draft
        draft = self._draft
        dcache = make_prompt_cache(draft)

        # Prefill both: target caches the prompt via the chain, draft locally.
        hidden = self._chain(self.shard.embed(prompt_ids), local_cache)
        token = self.shard.argmax_token(hidden)
        draft(mx.array([prompt_ids]), cache=dcache)

        generated = [token]
        printed = ""
        cur = mx.array(token, dtype=mx.uint32)  # same dtype argmax_tokens yields
        done = token in self._eos_ids
        while not done and len(generated) < max_new_tokens:
            # 1) draft k tokens, no host sync
            drafts, c = [], cur.reshape(1, 1)
            for _ in range(k):
                d = mx.argmax(draft(c, cache=dcache)[0, -1])
                drafts.append(d)
                c = d.reshape(1, 1)
            darr = mx.stack(drafts)

            # 2) one verify pass of [cur, d1..dk] through the whole chain.
            #    (Drafting the next chunk during this pass's network wait was
            #    tried and measured NEUTRAL-to-negative on the real rig: the
            #    win only lands on full-accept steps ~20% of the time, and the
            #    queued draft work delays the verify head-compute — don't.)
            verify_ids = mx.concatenate([cur.reshape(1), darr]).reshape(1, k + 1)
            hidden = self._chain(self.shard.embed_array(verify_ids), local_cache)
            tnext = self.shard.argmax_tokens(hidden)  # (k+1,) greedy targets

            # 3) accepted prefix length — the single sync per step
            matches = (tnext[:k] == darr).astype(mx.int32)
            n_acc = int(mx.sum(mx.cumprod(matches)).item())

            # 4) roll back the reject count everywhere, keep the draft in step
            target_trim, draft_trim, catch_up = spec_trims(k, n_acc)
            self._trim(target_trim, local_cache)
            ms.trim_cache(dcache, draft_trim)
            if catch_up:
                draft(darr[-1].reshape(1, 1), cache=dcache)

            # 5) the accepted drafts equal tnext[:n_acc]; tnext[n_acc] is free
            for t in tnext[: n_acc + 1].tolist():
                generated.append(int(t))
                if int(t) in self._eos_ids or len(generated) >= max_new_tokens:
                    done = True
                    break
            cur = tnext[n_acc]  # stays on-device; unused when done

            if on_token is not None:
                visible = [t for t in generated if t not in self._eos_ids]
                full = self.shard.tokenizer.decode(visible)
                if len(full) > len(printed):
                    on_token(full[len(printed):])
                    printed = full
        return generated

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
