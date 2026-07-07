"""Footprint reads exact bytes from safetensors headers and honours tie-weights."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from soup.cluster.footprint import footprint_from_dir


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, int]]) -> None:
    """Write a safetensors file whose header declares tensors of given byte sizes.

    tensors: name -> (dtype, nbytes). Bodies are zero-filled; only offsets matter.
    """

    header: dict = {}
    cursor = 0
    for name, (dtype, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": [nbytes], "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * cursor)


def _make_model(root: Path, *, layers: int, tie: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({
        "num_hidden_layers": layers,
        "tie_word_embeddings": tie,
    }))
    tensors: dict[str, tuple[str, int]] = {}
    for i in range(layers):
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = ("BF16", 1000)
        tensors[f"model.layers.{i}.mlp.up_proj.weight"] = ("BF16", 3000)  # 4000 / layer
    tensors["model.embed_tokens.weight"] = ("BF16", 5000)
    tensors["model.norm.weight"] = ("BF16", 100)
    tensors["lm_head.weight"] = ("BF16", 5000)
    _write_safetensors(root / "model.safetensors", tensors)


def test_per_layer_and_total(tmp_path: Path) -> None:
    _make_model(tmp_path, layers=6, tie=False)
    fp = footprint_from_dir(tmp_path, model_id="fixture", precision="bf16")
    assert fp.total_layers == 6
    assert fp.per_layer_bytes == [4000] * 6
    assert fp.layer_bytes_mean == 4000
    # untied: head = embed + norm + lm_head
    assert fp.head_bytes == 5000 + 100 + 5000
    assert fp.total_bytes == 6 * 4000 + 10100


def test_tie_weight_drops_lm_head(tmp_path: Path) -> None:
    _make_model(tmp_path, layers=4, tie=True)
    fp = footprint_from_dir(tmp_path, model_id="fixture", precision="bf16")
    # tied: lm_head shares the embedding, so it must NOT be counted twice.
    assert fp.head_bytes == 5000 + 100


def test_precision_rescale(tmp_path: Path) -> None:
    _make_model(tmp_path, layers=2, tie=False)
    bf16 = footprint_from_dir(tmp_path, model_id="fixture", precision="bf16")
    fp32 = footprint_from_dir(tmp_path, model_id="fixture", precision="fp32")
    # fp32 holds twice the bytes of the on-disk bf16.
    assert fp32.per_layer_bytes[0] == bf16.per_layer_bytes[0] * 2
    assert fp32.head_bytes == bf16.head_bytes * 2
