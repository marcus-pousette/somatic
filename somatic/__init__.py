"""Computer Soup (Python package `somatic`) — Somatic Runtime and Somatic Latent Graph.

Product surface — run a model too big for one machine, split across the machines you have:

    from somatic import Cluster, Host
    with Cluster.launch("Qwen/Qwen3-1.7B", ["localhost", "user@other-mac"]) as c:
        print(c.chat("Hello!"))
"""

__all__ = ["__version__", "Cluster", "Host"]

__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy so `import somatic` stays cheap (no torch/transformers import cost).
    if name in ("Cluster", "Host"):
        from somatic.cluster import Cluster, Host

        return {"Cluster": Cluster, "Host": Host}[name]
    raise AttributeError(f"module 'somatic' has no attribute {name!r}")
