"""Product serving layer: talk to a model split across N machines.

`ClusterEngine` owns the driver side (embed / final-norm / lm_head, shard-loaded)
and chains hidden states through an ordered list of layer-slice workers to
generate a reply. Both the interactive CLI (`scripts/cluster_chat.py`) and the
OpenAI-compatible HTTP server (`scripts/cluster_openai_server.py`) run on top of
it, so there is exactly one generation path to keep correct.
"""

from soup.serving.cluster_engine import ClusterEngine, WorkerSpec, parse_worker

__all__ = ["ClusterEngine", "WorkerSpec", "parse_worker"]
