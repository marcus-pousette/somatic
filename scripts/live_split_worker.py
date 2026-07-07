"""Standalone sequence-worker process for SOMATIC-LIVE-SPLIT-01.

Loads a contiguous layer slice of a causal LM and serves the existing
sequence-worker FastAPI app (binary prefill/decode endpoints with KV cache
and boundary-adapter wire strategies) over real HTTP.

Usage (one process per worker):
    uv run python scripts/live_split_worker.py \
        --model-id Qwen/Qwen3-0.6B --runtime-id worker-a \
        --layer-start 0 --layer-end 14 --port 8801
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument(
        "--data-port",
        type=int,
        default=None,
        help="Optional persistent raw-TCP data plane for per-step decode frames (control plane stays HTTP).",
    )
    parser.add_argument(
        "--shard-loading",
        action="store_true",
        help="Load only this worker's layer-slice weights (product path for models too large for one machine).",
    )
    parser.add_argument(
        "--head",
        action="store_true",
        help="Keep final norm + lm_head on this worker and serve the data-plane decode_head op (top-k logits reply instead of the hidden tensor).",
    )
    parser.add_argument(
        "--learned-codec-artifact",
        default=None,
        help="Optional hash-bound learned wire codec artifact JSON for worker-side learned_int8 data-plane decode.",
    )
    parser.add_argument(
        "--learned-codec-results",
        default=None,
        help="Compatibility path: optional training-binding results.json for worker-side learned_int8 data-plane decode.",
    )
    args = parser.parse_args()

    import torch

    torch.set_num_threads(max(int(args.num_threads), 1))
    import uvicorn

    from soup.sequence_model.interfaces import ResourceProfile
    from soup.sequence_model.qwen_real import QwenWorkerLayerRuntime
    from soup.servers.sequence_worker import create_app

    head_norm = None
    lm_head = None
    if args.head:
        from transformers import AutoConfig, AutoModelForCausalLM

        from soup.sequence_model.qwen_real import (
            QwenLayerRange,
            QwenShardManifest,
            _qwen_backbone,
            _torch_dtype,
        )

        config = AutoConfig.from_pretrained(
            args.model_id, trust_remote_code=True, local_files_only=True
        )
        full_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=_torch_dtype(torch, device=args.device, precision=args.precision),
            device_map=None,
            trust_remote_code=True,
            local_files_only=True,
        )
        full_model.to(args.device)
        full_model.eval()
        for parameter in full_model.parameters():
            parameter.requires_grad_(False)
        manifest = QwenShardManifest(
            shard_id=f"{args.model_id}:shard-{args.layer_start}-{args.layer_end}",
            runtime_id=args.runtime_id,
            model_id=args.model_id,
            weight_uri=args.model_id,
            precision=args.precision,
            device=args.device,
            layer_range=QwenLayerRange(
                start=int(args.layer_start), end=int(args.layer_end)
            ),
            roles=["layer_range", "final_norm", "lm_head"],
            metadata={
                "loader": "live-split-worker-head-v0",
                "total_layers": int(getattr(config, "num_hidden_layers", 0)),
                "hidden_size": int(getattr(config, "hidden_size", 0)),
                "local_files_only": True,
            },
        )
        head_norm = _qwen_backbone(full_model).norm
        lm_head = full_model.get_output_embeddings()
        runtime = QwenWorkerLayerRuntime.from_model(
            model=full_model, manifest=manifest, device=args.device
        )
        del full_model
    elif args.shard_loading:
        # Load ONLY this worker's layer-slice weights (the product path:
        # lets a model too large for one machine run across several). The
        # driver owns embed/norm/lm_head in the relay topology, so a plain
        # layer worker needs no head roles.
        runtime = QwenWorkerLayerRuntime.from_pretrained_shard(
            model_id=args.model_id,
            runtime_id=args.runtime_id,
            layer_start=int(args.layer_start),
            layer_end=int(args.layer_end),
            precision=args.precision,
            device=args.device,
            local_files_only=True,
        )
    else:
        runtime = QwenWorkerLayerRuntime.from_pretrained(
            model_id=args.model_id,
            runtime_id=args.runtime_id,
            layer_start=int(args.layer_start),
            layer_end=int(args.layer_end),
            precision=args.precision,
            device=args.device,
            local_files_only=True,
        )
    learned_int8_codec = None
    if args.learned_codec_artifact or args.learned_codec_results:
        from soup.sequence_model.boundary_compression.wire_codec_runtime import (
            load_learned_int8_wire_codec,
            load_learned_int8_wire_codec_from_artifact,
        )

        if args.learned_codec_artifact:
            learned_int8_codec = load_learned_int8_wire_codec_from_artifact(
                args.learned_codec_artifact
            )
        else:
            learned_int8_codec = load_learned_int8_wire_codec(
                args.learned_codec_results
            )
    app = create_app(
        resource_profile=ResourceProfile(
            runtime_id=args.runtime_id,
            backend="transformers",
            memory_gb=8,
            gpu_kind=args.device if args.device != "cpu" else None,
            supported_precisions=[args.precision],
            metadata={
                "model_id": args.model_id,
                "layer_start": int(args.layer_start),
                "layer_end": int(args.layer_end),
                "device": args.device,
                "precision": args.precision,
                "data_port": args.data_port,
                "head_enabled": bool(args.head),
                "learned_codec_artifact_loaded": bool(args.learned_codec_artifact),
                "learned_codec_results_loaded": bool(args.learned_codec_results),
            },
        ),
        qwen_shard_runtime=runtime,
    )

    if args.data_port is not None:
        import asyncio
        import json as jsonlib
        import struct
        import threading

        import numpy as np
        from soup.sequence_model.boundary_compression.wire_codec_runtime import (
            decode_boundary_wire_payload,
        )

        def _decode_request_array(header: dict, payload: bytes) -> np.ndarray:
            return decode_boundary_wire_payload(
                header,
                payload,
                learned_int8_codec=learned_int8_codec,
            )

        async def _handle(reader, writer):
            try:
                while True:
                    header_len_bytes = await reader.readexactly(4)
                    header_len = struct.unpack(">I", header_len_bytes)[0]
                    header = jsonlib.loads(await reader.readexactly(header_len))
                    payload = await reader.readexactly(int(header["nbytes"]))
                    array = _decode_request_array(header, payload)
                    result = runtime.decode_array(
                        array,
                        name="hidden_states",
                        metadata={},
                        cache_id=header["cache_id"],
                        sequence_id=header["sequence_id"],
                        position_start=int(header["position_start"]),
                    )
                    if header.get("op") == "decode_head":
                        if head_norm is None or lm_head is None:
                            raise RuntimeError(
                                "decode_head requested but worker launched without --head"
                            )
                        with torch.no_grad():
                            hidden = torch.from_numpy(
                                np.ascontiguousarray(
                                    result.tensor.to_numpy(), dtype=np.float32
                                )
                            ).to(runtime.device)
                            hidden = hidden.to(
                                next(lm_head.parameters()).dtype
                            )
                            logits = (
                                lm_head(head_norm(hidden))[:, -1, :].float().cpu()
                            )
                        top_k = int(header.get("top_k", 8))
                        top_values, top_ids = torch.topk(logits, top_k, dim=-1)
                        reply = jsonlib.dumps(
                            {
                                "op": "decode_head",
                                "nbytes": 0,
                                "top_values": [
                                    float(v) for v in top_values[0].tolist()
                                ],
                                "top_ids": [int(i) for i in top_ids[0].tolist()],
                            }
                        ).encode()
                        writer.write(struct.pack(">I", len(reply)) + reply)
                        await writer.drain()
                        continue
                    out = np.ascontiguousarray(
                        result.tensor.to_numpy(), dtype=np.float32
                    )
                    reply = jsonlib.dumps(
                        {"shape": list(out.shape), "nbytes": out.nbytes}
                    ).encode()
                    writer.write(struct.pack(">I", len(reply)) + reply + out.tobytes())
                    await writer.drain()
            except (asyncio.IncompleteReadError, ConnectionResetError):
                pass
            finally:
                writer.close()

        def _serve_data_plane():
            async def _main():
                server = await asyncio.start_server(
                    _handle, args.host, int(args.data_port)
                )
                async with server:
                    await server.serve_forever()

            asyncio.run(_main())

        threading.Thread(target=_serve_data_plane, daemon=True).start()
        print(f"data plane listening on {args.host}:{args.data_port}", flush=True)

    uvicorn.run(app, host=args.host, port=int(args.port), log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
