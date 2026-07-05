"""Lossless recompression layer for boundary-wire payloads.

A thin, byte-exact wrapper applied AFTER an encoder produces its payload
bytes. It shrinks the payload further with a stdlib lossless codec (zlib or
lzma) without touching any fidelity gate: ``decode(encode(x))`` is
byte-identical to the pre-lossless payload, so downstream decoding and every
token/argmax/top-k/NLL gate are unchanged.

Scope: only compressible payload kinds (int8 / trainable / sparse-codebook).
fp16 and identity payloads are near-incompressible (high-entropy mantissa
bits) and are skipped so the header flag never claims a non-win. The layer
also refuses to grow a payload — if compression does not help, it returns the
original bytes uncompressed and records ``lossless_payload_codec = None``.
"""

from __future__ import annotations

import bz2
import lzma
import zlib
from typing import Any

# Payload kinds whose bytes carry quantization structure worth compressing.
# fp16/identity are excluded on purpose (measured ~4% headroom on real
# activations vs ~30-43% for int8/mixed).
LOSSLESS_ELIGIBLE_STRATEGIES: frozenset[str] = frozenset(
    {
        "int8_symmetric",
        "learned_int8",
        "learned_per_boundary_int8",
        "learned_residual4_int8",
        "learned_residual4_sparse24_int8",
        "learned_residual5_int8",
        "learned_residual6_int8",
        "learned_residual8_int8",
        "sparse_codebook",
        "trainable_autoencoder_source_latent",
    }
)

_COMPRESSORS = {
    "zlib": lambda raw: zlib.compress(raw, 6),
    "lzma": lambda raw: lzma.compress(raw, preset=1),
    "bz2": lambda raw: bz2.compress(raw, 6),
}
_DECOMPRESSORS = {
    "zlib": zlib.decompress,
    "lzma": lzma.decompress,
    "bz2": bz2.decompress,
}

# Header keys used to signal the lossless layer to the decode side.
LOSSLESS_CODEC_HEADER_KEY = "lossless_payload_codec"
LOSSLESS_UNCOMPRESSED_NBYTES_HEADER_KEY = "lossless_uncompressed_nbytes"


def strategy_is_lossless_eligible(strategy: str | None) -> bool:
    return strategy is not None and str(strategy) in LOSSLESS_ELIGIBLE_STRATEGIES


def compress_wire_payload(
    payload: bytes,
    *,
    codec: str,
) -> tuple[bytes, dict[str, Any]]:
    """Losslessly compress ``payload``; fail closed if it does not shrink.

    Returns (bytes, header_delta). ``header_delta`` carries the codec name and
    the original byte count when compression won, or a null codec when it did
    not (so the decode side inflates only real compressed payloads).
    """
    if codec not in _COMPRESSORS:
        raise ValueError(f"unsupported lossless payload codec: {codec}")
    compressed = _COMPRESSORS[codec](payload)
    if len(compressed) >= len(payload):
        return payload, {
            LOSSLESS_CODEC_HEADER_KEY: None,
            LOSSLESS_UNCOMPRESSED_NBYTES_HEADER_KEY: None,
        }
    return compressed, {
        LOSSLESS_CODEC_HEADER_KEY: codec,
        LOSSLESS_UNCOMPRESSED_NBYTES_HEADER_KEY: len(payload),
    }


def decompress_wire_payload(
    payload: bytes,
    *,
    header: dict[str, Any],
) -> bytes:
    """Inflate a payload the encode side compressed; pass through otherwise."""
    codec = header.get(LOSSLESS_CODEC_HEADER_KEY)
    if codec is None:
        return payload
    if codec not in _DECOMPRESSORS:
        raise ValueError(f"unsupported lossless payload codec: {codec}")
    restored = _DECOMPRESSORS[str(codec)](payload)
    expected = header.get(LOSSLESS_UNCOMPRESSED_NBYTES_HEADER_KEY)
    if expected is not None and len(restored) != int(expected):
        raise ValueError(
            "lossless payload inflated to the wrong byte count: "
            f"expected {int(expected)}, got {len(restored)}"
        )
    return restored


__all__ = [
    "LOSSLESS_ELIGIBLE_STRATEGIES",
    "LOSSLESS_CODEC_HEADER_KEY",
    "LOSSLESS_UNCOMPRESSED_NBYTES_HEADER_KEY",
    "strategy_is_lossless_eligible",
    "compress_wire_payload",
    "decompress_wire_payload",
]
