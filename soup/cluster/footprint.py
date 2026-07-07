"""Exact per-layer / head byte footprint of a model, read without downloading weights.

The launcher needs to know, for any model, how many bytes each transformer layer
costs and how many the driver-side head (embed / final-norm / lm_head) costs, so
it can fit a layer split onto machines by their real free RAM. We read this
straight from the safetensors headers — the leading JSON of each shard carries
every tensor's dtype and byte offsets — so no tensor bodies are fetched or loaded.

Model-general by construction: it buckets tensors by the standard Llama-family
names (`model.layers.{i}.*`, `model.embed_tokens`, `model.norm`, `lm_head`) that
Qwen / Llama / Mistral / Gemma / Phi / SmolLM all share.
"""

from __future__ import annotations

import glob
import json
import struct
from dataclasses import dataclass
from pathlib import Path

# bytes per element for the dtypes a safetensors header may declare.
_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}

_PRECISION_BYTES = {"fp32": 4, "float32": 4, "fp16": 2, "float16": 2, "bf16": 2, "bfloat16": 2}


@dataclass(frozen=True)
class ModelFootprint:
    model_id: str
    total_layers: int
    per_layer_bytes: list[int]
    head_bytes: int
    disk_dtype: str
    precision: str  # the dtype the workers/driver will actually hold

    @property
    def layer_bytes_mean(self) -> float:
        return sum(self.per_layer_bytes) / self.total_layers if self.total_layers else 0.0

    @property
    def total_bytes(self) -> int:
        return sum(self.per_layer_bytes) + self.head_bytes


def _read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


def _scale_for_precision(disk_dtype: str, precision: str) -> float:
    """If the workers hold a different dtype than what's on disk, scale bytes."""

    disk = _DTYPE_BYTES.get(disk_dtype.upper())
    target = _PRECISION_BYTES.get(precision.lower())
    if not disk or not target:
        return 1.0
    return target / disk


def model_footprint(
    model_id: str,
    *,
    precision: str = "bf16",
    local_files_only: bool = False,
) -> ModelFootprint:
    """Resolve a model's byte footprint from its safetensors headers.

    Only config + index (+ shard headers) are read — never the weight bodies.
    `precision` is the dtype the cluster will actually hold; byte costs are
    rescaled from the on-disk dtype accordingly.
    """

    from huggingface_hub import snapshot_download

    # Index + config are kilobytes; for single-file models we still only read the
    # leading header of each .safetensors, not the tensor bodies.
    root = Path(
        snapshot_download(
            model_id,
            local_files_only=local_files_only,
            allow_patterns=[
                "*.safetensors.index.json",
                "config.json",
                "*.safetensors",
            ],
        )
    )
    return footprint_from_dir(root, model_id=model_id, precision=precision)


def footprint_from_dir(root: Path, *, model_id: str, precision: str = "bf16") -> ModelFootprint:
    """Compute a footprint from a resolved model directory (config + shards).

    Split out from `model_footprint` so it can be unit-tested against a fixture
    without touching the network or the HF cache.
    """

    config = json.loads((root / "config.json").read_text())
    total_layers = int(config["num_hidden_layers"])
    tied = bool(config.get("tie_word_embeddings", False))

    shard_files = sorted(glob.glob(str(root / "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"no safetensors shards found for {model_id} under {root}")

    per_layer = [0] * total_layers
    head = 0
    disk_dtype = "BF16"
    for shard in shard_files:
        header = _read_safetensors_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            offsets = meta.get("data_offsets")
            if not offsets:
                continue
            nbytes = offsets[1] - offsets[0]
            disk_dtype = meta.get("dtype", disk_dtype)
            if name.startswith("model.layers."):
                layer_index = int(name.split(".")[2])
                if 0 <= layer_index < total_layers:
                    per_layer[layer_index] += nbytes
            elif "embed_tokens" in name or name == "model.norm.weight" or name.endswith(".model.norm.weight"):
                head += nbytes
            elif "lm_head" in name:
                # A tied model shares lm_head with the embedding — the driver
                # holds one copy, so don't count it twice.
                if not tied:
                    head += nbytes

    scale = _scale_for_precision(disk_dtype, precision)
    per_layer = [int(round(b * scale)) for b in per_layer]
    head = int(round(head * scale))

    return ModelFootprint(
        model_id=model_id,
        total_layers=total_layers,
        per_layer_bytes=per_layer,
        head_bytes=head,
        disk_dtype=disk_dtype,
        precision=precision,
    )
