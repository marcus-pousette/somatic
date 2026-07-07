"""The one generation path for the split-model product.

A `ClusterEngine` holds the driver-side head modules (embedding / final norm /
lm_head, shard-loaded so the driver never needs the whole model) and an ordered
list of worker clients, each holding a contiguous slice of the transformer
layers. To generate, it embeds the prompt, chains the hidden states through
every worker in order, applies the final norm + lm_head, and greedily samples
the next token — repeating one token at a time, streaming each as it lands.

It is model-general: layers, embeddings and the rotary cache are discovered via
the generic `AutoModelForCausalLM` backbone, so any Llama-family HF decoder LM
(embed_tokens / layers / norm / rotary_emb) works with no code change. And it is
N-worker: the chain is a plain loop, so two machines and ten machines are the
same code.

Requests are serialised with an async lock — a home cluster runs one generation
at a time, and the workers hold per-sequence KV caches that must not interleave.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence


@dataclass(frozen=True)
class WorkerSpec:
    """One machine's slice of the model: where it lives, which layers it holds."""

    base_url: str
    layer_start: int
    layer_end: int


def parse_worker(spec: str) -> WorkerSpec:
    """Parse a ``URL:layer_start:layer_end`` string.

    Split from the right so the URL's own colons (scheme, host:port) survive.
    """

    head, start, end = spec.rsplit(":", 2)
    return WorkerSpec(base_url=head, layer_start=int(start), layer_end=int(end))


def even_layer_ranges(total_layers: int, machines: int) -> list[tuple[int, int]]:
    """An even contiguous split of ``total_layers`` across ``machines``."""

    base, extra = divmod(total_layers, machines)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for index in range(machines):
        span = base + (1 if index < extra else 0)
        ranges.append((cursor, cursor + span))
        cursor += span
    return ranges


class ClusterEngine:
    def __init__(
        self,
        *,
        model_id: str,
        workers: Sequence[WorkerSpec],
        boundary_strategy: str = "fp16",
        driver_precision: str = "fp32",
        driver_device: str | None = None,
        num_threads: int = 8,
        local_files_only: bool = True,
    ) -> None:
        if not workers:
            raise ValueError("ClusterEngine requires at least one worker")
        self.model_id = model_id
        self.workers = sorted(workers, key=lambda w: w.layer_start)
        self.boundary_strategy = boundary_strategy
        self.driver_precision = driver_precision
        self.driver_device = driver_device  # None -> auto (mps if available else cpu)
        self.num_threads = num_threads
        self.local_files_only = local_files_only

        self._tokenizer = None
        self._device = "cpu"
        self._head_dtype = None
        self._embed = None
        self._norm = None
        self._lm_head = None
        self._eos_ids: set[int] = set()
        self._clients: list[Any] = []
        self._worker_clients: list[Any] = []
        self._lock = asyncio.Lock()
        self._turn = 0

    @property
    def layout(self) -> str:
        return " -> ".join(
            f"[{w.layer_start},{w.layer_end})@{w.base_url}" for w in self.workers
        )

    async def start(self) -> None:
        import httpx
        import torch
        from transformers import AutoTokenizer

        from somatic.sequence_model.qwen_real import load_head_modules_shard
        from somatic.sequence_model.remote import SequenceWorkerClient

        torch.set_num_threads(max(int(self.num_threads), 1))
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, local_files_only=self.local_files_only
        )
        # Run the head (embed / norm / lm_head) on the GPU when available: the
        # lm_head is a hidden x vocab matmul done every token, and fp32-on-CPU is
        # ~20% of per-token latency. MPS/fp32 keeps the numerics, just faster.
        if self.driver_device is not None:
            self._device = self.driver_device
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self._embed, self._norm, self._lm_head, _hidden = load_head_modules_shard(
            model_id=self.model_id,
            precision=self.driver_precision,
            device=self._device,
            local_files_only=self.local_files_only,
        )
        self._head_dtype = next(self._lm_head.parameters()).dtype
        self._eos_ids = self._resolve_eos_ids()
        for spec in self.workers:
            client = httpx.AsyncClient(base_url=spec.base_url, timeout=600.0)
            self._clients.append(client)
            self._worker_clients.append(
                SequenceWorkerClient(base_url=spec.base_url, client=client)
            )

    async def close(self) -> None:
        for client in self._clients:
            await client.aclose()
        self._clients.clear()
        self._worker_clients.clear()

    def _resolve_eos_ids(self) -> set[int]:
        tokenizer = self._tokenizer
        eos_ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            eos_ids.add(int(tokenizer.eos_token_id))
        for name in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(name)
            except Exception:
                tid = None
            if isinstance(tid, int) and tid >= 0:
                eos_ids.add(tid)
        return eos_ids

    def _prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        text = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return [
            int(t)
            for t in self._tokenizer(text, add_special_tokens=False)["input_ids"]
        ]

    def _embed_payload(self, token_ids: list[int]):
        import torch

        from somatic.sequence_model.tensors import TensorPayload

        with torch.no_grad():
            ids = torch.tensor([token_ids], device=self._device)
            hidden = self._embed(ids).detach().to("cpu").float().numpy()
        return TensorPayload.from_numpy(hidden, name="hidden_states")

    def _next_token(self, hidden_payload) -> int:
        import torch

        with torch.no_grad():
            final = torch.from_numpy(hidden_payload.to_numpy()).to(
                self._device, dtype=self._head_dtype
            )
            logits = self._lm_head(self._norm(final))[:, -1, :].float()
            return int(torch.argmax(logits, dim=-1)[0].to("cpu"))

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        on_token: Callable[[str], Awaitable[None]] | Callable[[str], None] | None = None,
        boundary_strategy: str | None = None,
    ) -> str:
        """Generate a reply for a full chat `messages` list (decoded text).

        `boundary_strategy` overrides the wire codec for this call (used by
        `soup verify` to compare modes on the same workers). If `on_token` is
        given it receives each incremental text delta as tokens are produced.
        """

        ids = await self.generate_ids(
            messages,
            max_new_tokens=max_new_tokens,
            on_token=on_token,
            boundary_strategy=boundary_strategy,
        )
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    async def generate_ids(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        on_token: Callable[[str], Awaitable[None]] | Callable[[str], None] | None = None,
        boundary_strategy: str | None = None,
    ) -> list[int]:
        """Generate and return the raw (visible) token ids — the exact sequence
        `soup verify` compares across boundary strategies."""

        strategy = boundary_strategy or self.boundary_strategy
        async with self._lock:
            self._turn += 1
            sequence_id = f"cluster-{self._turn}"
            caches = []
            for worker in self._worker_clients:
                handle = await worker.create_qwen_cache(sequence_id=sequence_id)
                caches.append(handle.cache_id)

            prompt_ids = self._prompt_ids(messages)
            generated: list[int] = []
            printed = ""

            async def emit(new_text: str) -> None:
                nonlocal printed
                delta = new_text[len(printed):]
                if delta and on_token is not None:
                    result = on_token(delta)
                    if asyncio.iscoroutine(result):
                        await result
                printed = new_text

            # Prefill the prompt through the whole chain.
            hidden = self._embed_payload(prompt_ids)
            for worker, cache in zip(self._worker_clients, caches, strict=True):
                result, _ = await worker.prefill_qwen_shard_binary(
                    tensor=hidden,
                    cache_id=cache,
                    sequence_id=sequence_id,
                    position_start=0,
                    boundary_adapter_strategy=strategy,
                )
                hidden = result.tensor
            generated.append(self._next_token(hidden))
            position = len(prompt_ids)

            for _ in range(max_new_tokens - 1):
                if generated[-1] in self._eos_ids:
                    break
                step = self._embed_payload([generated[-1]])
                for worker, cache in zip(self._worker_clients, caches, strict=True):
                    result, _ = await worker.decode_qwen_shard_binary(
                        tensor=step,
                        cache_id=cache,
                        sequence_id=sequence_id,
                        position_start=position,
                        boundary_adapter_strategy=strategy,
                    )
                    step = result.tensor
                generated.append(self._next_token(step))
                position += 1
                visible = [t for t in generated if t not in self._eos_ids]
                await emit(self._tokenizer.decode(visible, skip_special_tokens=True))

            return [t for t in generated if t not in self._eos_ids]
