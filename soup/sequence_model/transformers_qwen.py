from __future__ import annotations

import importlib.util
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from soup.sequence_model.interfaces import ModelManifest, Precision
from soup.sequence_model.qwen import qwen_manifest_from_config


QwenSmokeStatus = Literal["ready", "skipped", "failed"]


class QwenDependencyStatus(BaseModel):
    available: bool
    missing: list[str] = Field(default_factory=list)
    installed: list[str] = Field(default_factory=list)


class QwenGenerationResult(BaseModel):
    status: QwenSmokeStatus
    model_id: str
    prompt: str
    generated_text: str = ""
    new_text: str = ""
    tokens_generated: int = 0
    elapsed_seconds: float = 0.0
    manifest: ModelManifest | None = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransformersQwenAdapter:
    name = "transformers-qwen"

    def __init__(self) -> None:
        self._model_cache: dict[tuple[str, str, bool, Precision], tuple[Any, Any, ModelManifest]] = {}

    def dependency_status(self) -> QwenDependencyStatus:
        packages = ["transformers", "accelerate", "safetensors"]
        installed = [name for name in packages if importlib.util.find_spec(name) is not None]
        missing = [name for name in packages if name not in installed]
        return QwenDependencyStatus(available=not missing, installed=installed, missing=missing)

    def inspect_manifest(
        self,
        model_id: str = "Qwen/Qwen3-0.6B",
        *,
        default_precision: Precision = "bf16",
        local_files_only: bool = False,
    ) -> ModelManifest:
        self._require_dependencies()
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
        manifest = qwen_manifest_from_config(model_id=model_id, config=config, default_precision=default_precision)
        manifest.adapter_compatibility = sorted(set(manifest.adapter_compatibility + ["transformers-qwen"]))
        manifest.metadata["real_weight_adapter"] = self.name
        return manifest

    def generate(
        self,
        *,
        model_id: str = "Qwen/Qwen3-0.6B",
        prompt: str,
        max_new_tokens: int = 24,
        device: str = "cpu",
        local_files_only: bool = False,
        default_precision: Precision = "bf16",
    ) -> QwenGenerationResult:
        dependency_status = self.dependency_status()
        if not dependency_status.available:
            return QwenGenerationResult(
                status="skipped",
                model_id=model_id,
                prompt=prompt,
                reason=f"missing optional dependencies: {', '.join(dependency_status.missing)}",
                metadata={"install": "uv sync --extra qwen"},
            )

        try:
            import torch

            started = time.perf_counter()
            tokenizer, model, manifest = self._load_model(
                model_id=model_id,
                device=device,
                local_files_only=local_files_only,
                default_precision=default_precision,
            )
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            prompt_token_count = int(encoded["input_ids"].shape[-1])
            tokens_generated = max(int(output.shape[-1]) - prompt_token_count, 0)
            elapsed = time.perf_counter() - started
            return QwenGenerationResult(
                status="ready",
                model_id=model_id,
                prompt=prompt,
                generated_text=generated_text,
                new_text=generated_text[len(prompt) :].strip() if generated_text.startswith(prompt) else generated_text,
                tokens_generated=tokens_generated,
                elapsed_seconds=elapsed,
                manifest=manifest,
                metadata={"adapter": self.name, "device": device, "precision": default_precision, "cached_model": "true"},
            )
        except Exception as exc:
            return QwenGenerationResult(
                status="failed",
                model_id=model_id,
                prompt=prompt,
                reason=str(exc),
                metadata={"adapter": self.name, "device": device, "local_files_only": str(local_files_only)},
            )

    def _require_dependencies(self) -> None:
        status = self.dependency_status()
        if not status.available:
            raise RuntimeError(f"missing optional dependencies: {', '.join(status.missing)}")

    def _load_model(
        self,
        *,
        model_id: str,
        device: str,
        local_files_only: bool,
        default_precision: Precision,
    ) -> tuple[Any, Any, ModelManifest]:
        key = (model_id, device, local_files_only, default_precision)
        if key in self._model_cache:
            return self._model_cache[key]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        manifest = self.inspect_manifest(model_id, default_precision=default_precision, local_files_only=local_files_only)
        dtype = _torch_dtype(torch, device=device, precision=default_precision)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=None,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model.to(device)
        model.eval()
        self._model_cache[key] = (tokenizer, model, manifest)
        return self._model_cache[key]


def _torch_dtype(torch: Any, *, device: str, precision: Precision) -> Any:
    if precision == "bf16" and device != "cpu":
        return torch.bfloat16
    if precision == "fp16" and device != "cpu":
        return torch.float16
    return torch.float32
