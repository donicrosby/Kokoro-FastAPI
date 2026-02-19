"""Model configuration for Kokoro V1.

This module provides model-specific configuration settings that complement the application-level
settings in config.py. While config.py handles general application settings (API, paths, etc.),
this module focuses on memory management and model file paths.
"""

from pydantic import BaseModel, Field


class KokoroV1Config(BaseModel):
    """Kokoro V1 configuration."""

    languages: list[str] = ["en"]

    class Config:
        frozen = True


class PyTorchConfig(BaseModel):
    """PyTorch backend configuration."""

    memory_threshold: float = Field(0.8, description="Memory threshold for cleanup")
    retry_on_oom: bool = Field(True, description="Whether to retry on OOM errors")

    class Config:
        frozen = True


class ModelConfig(BaseModel):
    """Kokoro V1 model configuration (ONNX backend)."""

    # General settings
    cache_voices: bool = Field(True, description="Whether to cache voice tensors")
    voice_cache_size: int = Field(2, description="Maximum number of cached voices")

    # ONNX model and voices (primary)
    kokoro_onnx_file: str = Field(
        "v1_0/kokoro-v1.0.onnx",
        description="Kokoro ONNX model filename",
    )
    kokoro_onnx_voices_file: str = Field(
        "v1_0/voices-v1.0.bin",
        description="Kokoro ONNX voices bundle filename",
    )

    # Backend config (legacy / optional)
    pytorch_gpu: PyTorchConfig = Field(default_factory=PyTorchConfig)

    class Config:
        frozen = True


# Global instance
model_config = ModelConfig()
