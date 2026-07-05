from __future__ import annotations

from typing import Any

from somatic.sequence_model.interfaces import ModelGraph, ModelManifest, ModelUnit


def qwen_transformer_manifest(
    *,
    model_id: str = "Qwen/Qwen3-0.6B",
    num_layers: int = 28,
    hidden_size: int = 1024,
    default_precision: str = "bf16",
) -> ModelManifest:
    """Return a Qwen-family Transformer manifest expressed through generic model units."""
    units: list[ModelUnit] = [
        ModelUnit(
            unit_id="embedding",
            kind="embedding",
            ordinal=0,
            activation_bytes=hidden_size * 2,
            required_memory_gb=0.25,
        )
    ]
    ordinal = 1
    for layer in range(num_layers):
        units.append(
            ModelUnit(
                unit_id=f"layer_{layer:02d}_attention",
                kind="attention_block",
                ordinal=ordinal,
                activation_bytes=hidden_size * 2,
                required_memory_gb=0.05,
                metadata={"layer": layer},
            )
        )
        ordinal += 1
        units.append(
            ModelUnit(
                unit_id=f"layer_{layer:02d}_mlp",
                kind="mlp",
                ordinal=ordinal,
                activation_bytes=hidden_size * 2,
                required_memory_gb=0.05,
                metadata={"layer": layer},
            )
        )
        ordinal += 1
    units.extend(
        [
            ModelUnit(unit_id="final_norm", kind="norm", ordinal=ordinal, activation_bytes=hidden_size * 2),
            ModelUnit(unit_id="lm_head", kind="lm_head", ordinal=ordinal + 1, activation_bytes=hidden_size * 2),
        ]
    )
    return ModelManifest(
        model_id=model_id,
        display_name="Qwen family Transformer demo manifest",
        architecture_family="transformer",
        tokenizer_id=model_id,
        weight_uri=model_id,
        supported_precisions=["bf16", "fp16", "fp32", "int8", "int4"],
        default_precision=default_precision,  # type: ignore[arg-type]
        adapter_compatibility=["transformers", "qwen-family"],
        graph=ModelGraph(graph_id=f"{model_id}:generic-transformer", units=units),
        metadata={
            "demo_path": "first compatibility target, not a runtime assumption",
            "hidden_size": hidden_size,
        },
    )


def qwen_next_hybrid_manifest(
    *,
    model_id: str = "Qwen/Qwen3-Next",
    num_layers: int = 16,
    hidden_size: int = 2048,
    default_precision: str = "bf16",
) -> ModelManifest:
    """Sketch a Qwen-Next-style hybrid graph with DeltaNet, attention, and MoE units."""
    units: list[ModelUnit] = [
        ModelUnit(
            unit_id="embedding",
            kind="embedding",
            ordinal=0,
            activation_bytes=hidden_size * 2,
            required_memory_gb=0.5,
        )
    ]
    ordinal = 1
    for layer in range(num_layers):
        sequence_kind = "attention_block" if layer % 4 == 0 else "deltanet_block"
        units.append(
            ModelUnit(
                unit_id=f"layer_{layer:02d}_{sequence_kind}",
                kind=sequence_kind,  # type: ignore[arg-type]
                ordinal=ordinal,
                activation_bytes=hidden_size * 2,
                required_memory_gb=0.08,
                metadata={"layer": layer},
            )
        )
        ordinal += 1
        units.append(
            ModelUnit(
                unit_id=f"layer_{layer:02d}_moe",
                kind="moe",
                ordinal=ordinal,
                activation_bytes=hidden_size * 2,
                required_memory_gb=0.12,
                metadata={"layer": layer},
            )
        )
        ordinal += 1
    units.extend(
        [
            ModelUnit(unit_id="final_norm", kind="norm", ordinal=ordinal, activation_bytes=hidden_size * 2),
            ModelUnit(unit_id="lm_head", kind="lm_head", ordinal=ordinal + 1, activation_bytes=hidden_size * 2),
        ]
    )
    return ModelManifest(
        model_id=model_id,
        display_name="Qwen-Next-style hybrid demo manifest",
        architecture_family="hybrid",
        tokenizer_id=model_id,
        weight_uri=model_id,
        supported_precisions=["bf16", "fp16", "fp32", "int8", "int4"],
        default_precision=default_precision,  # type: ignore[arg-type]
        adapter_compatibility=["transformers", "qwen-family", "hybrid-sequence"],
        graph=ModelGraph(graph_id=f"{model_id}:generic-hybrid", units=units),
        metadata={
            "demo_path": "hybrid graph used to keep the runtime architecture-neutral",
            "hidden_size": hidden_size,
        },
    )


def qwen_manifest_from_config(*, model_id: str, config: Any, default_precision: str = "bf16") -> ModelManifest:
    """Build a generic Qwen-family manifest from a Transformers config-like object."""
    num_layers = int(_config_value(config, "num_hidden_layers", "n_layer", default=28))
    hidden_size = int(_config_value(config, "hidden_size", "n_embd", default=1024))
    model_type = str(_config_value(config, "model_type", default="qwen")).lower()
    architectures = [str(value).lower() for value in _config_value(config, "architectures", default=[]) or []]
    has_moe = bool(_config_value(config, "num_experts", "num_local_experts", "moe_intermediate_size", default=0))
    if "next" in model_id.lower() or "deltanet" in model_type or any("next" in item for item in architectures):
        manifest = qwen_next_hybrid_manifest(
            model_id=model_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            default_precision=default_precision,
        )
    else:
        manifest = qwen_transformer_manifest(
            model_id=model_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            default_precision=default_precision,
        )
    if has_moe:
        manifest.metadata["config_has_moe"] = True
    manifest.metadata["source"] = "transformers-config"
    manifest.metadata["model_type"] = model_type
    return manifest


def _config_value(config: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(config, dict) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    if hasattr(config, "to_dict"):
        values = config.to_dict()
        for name in names:
            if name in values:
                return values[name]
    return default
