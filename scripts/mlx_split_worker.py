"""MLX split worker — holds a contiguous layer range [start, end) and runs it.

Deployed to each machine alongside ``mlx_shard.py`` (both need only ``mlx``/
``mlx_lm`` + numpy — no torch, no full somatic install). The driver
(`MLXClusterEngine`) chains hidden states through one of these per machine.

    python mlx_split_worker.py --model Qwen/Qwen3-14B --start 20 --end 40 --port 5599

A worker loads the whole model lazily but only ever runs its own layers, so only
those materialise (shard-only loading). One sequence at a time (home cluster); a
zero-length frame resets the KV cache for a new sequence.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

# mlx_shard.py is deployed next to this script (mlx-only, no somatic dependency).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx_shard as ms  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    print(f"mlx-worker: loading {args.model} (lazy) ...", flush=True)
    shard = ms.ShardModel(args.model)
    end = args.end if args.end > 0 else shard.n_layers
    print(
        f"mlx-worker: ready — layers [{args.start},{end}) of {shard.n_layers}, "
        f"dim {shard.dim}, {args.host}:{args.port}",
        flush=True,
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)

    # One driver owns this worker for its lifetime. Exit when the driver's real
    # connection closes — even a hard kill, since the OS closes the socket and
    # recv returns None — so teardown is automatic, no orphaned worker. Bare TCP
    # health probes (connect then close with no frames) are ignored, not fatal.
    while True:
        conn, _ = srv.accept()
        ms.set_nodelay(conn)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # detect a dead driver
        cache = shard.make_cache(args.start, end)
        served_any = False
        try:
            while True:
                frame = ms.recv_frame(conn)
                if frame is None:
                    break
                served_any = True
                if len(frame) == 0:  # reset: new sequence
                    cache = shard.make_cache(args.start, end)
                    ms.send_frame(conn, b"")
                    continue
                hidden = ms.bytes_to_hidden(frame, shard.dim)
                hidden = shard.run_layers(hidden, cache, args.start, end)
                ms.send_frame(conn, ms.hidden_to_bytes(hidden))
        finally:
            conn.close()
        if served_any:  # the driver's working connection closed -> we're done
            break
    print("mlx-worker: driver disconnected — exiting", flush=True)


if __name__ == "__main__":
    main()
