"""Kokoro ONNX backend with controlled resource management."""

import os
import tempfile
from typing import AsyncGenerator, Optional, Tuple, Union

import numpy as np
from kokoro_onnx import Kokoro
from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.lang import lang_code_to_onnx
from .base import AudioChunk, BaseModelBackend

# kokoro-onnx indexes voice by len(tokens); max token length is 510 (512 - 2 pad)
MAX_PHONEME_LENGTH = 511


def _ensure_voice_shape(arr: np.ndarray) -> np.ndarray:
    """Ensure voice array is (MAX_PHONEME_LENGTH, 1, 256) for kokoro-onnx indexing."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == 256:
        arr = np.tile(arr.reshape(1, 1, -1), (MAX_PHONEME_LENGTH, 1, 1))
    elif arr.ndim == 2 and arr.shape in ((1, 256), (256, 1)):
        arr = arr.reshape(1, 256) if arr.shape[0] == 256 else arr
        arr = np.tile(arr.reshape(1, 1, -1), (MAX_PHONEME_LENGTH, 1, 1))
    elif arr.ndim != 3 or arr.shape[1:] != (1, 256) or arr.shape[0] < MAX_PHONEME_LENGTH:
        if arr.ndim == 3 and arr.shape[1:] == (1, 256) and arr.shape[0] >= MAX_PHONEME_LENGTH:
            pass
        else:
            # Replicate first row/slice to fill (MAX_PHONEME_LENGTH, 1, 256)
            if arr.ndim == 3:
                row = arr[0:1]
            else:
                row = arr.reshape(1, 1, -1) if arr.size == 256 else arr.reshape(1, -1)
                if row.shape[-1] != 256:
                    row = row.reshape(1, 1, -1)
            arr = np.tile(row, (MAX_PHONEME_LENGTH, 1, 1))
    return np.asarray(arr, dtype=np.float32)


def _load_voice_array(voice_path: str) -> np.ndarray:
    """Load voice array from .npy or .pt file. Prefer numpy; torch only for .pt."""
    ext = os.path.splitext(voice_path)[1].lower()
    if ext == ".npy":
        arr = np.load(voice_path, allow_pickle=False)
        return _ensure_voice_shape(arr)
    if ext == ".pt":
        try:
            import torch
            data = torch.load(voice_path, map_location="cpu", weights_only=True)
            if hasattr(data, "numpy"):
                arr = data.numpy()
            else:
                arr = np.array(data, dtype=np.float32)
            return _ensure_voice_shape(arr)
        except ImportError:
            raise RuntimeError(
                "Loading .pt voice files requires torch. Install with: pip install kokoro-fastapi[cpu] (or [gpu])"
            )
    raise ValueError(f"Unsupported voice file extension: {ext}")


class KokoroV1(BaseModelBackend):
    """Kokoro ONNX backend with controlled resource management."""

    def __init__(self):
        """Initialize backend with environment-based configuration."""
        super().__init__()
        self._device = "cpu"  # ONNX Runtime manages provider
        self._model: Optional[Kokoro] = None

    async def load_model(self, path: str) -> None:
        """Load ONNX model and voices bundle.

        Args:
            path: Ignored; paths come from model_config.

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            onnx_path = await paths.get_onnx_model_path()
            voices_path = await paths.get_onnx_voices_path()

            logger.info("Loading Kokoro ONNX model")
            logger.info(f"Model path: {onnx_path}")
            logger.info(f"Voices path: {voices_path}")

            self._model = Kokoro(onnx_path, voices_path)

        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load Kokoro ONNX model: {e}") from e

    def _resolve_voice_spec(
        self,
        voice: Union[str, Tuple[str, Union[np.ndarray, str]]],
    ) -> Union[str, np.ndarray]:
        """Resolve voice to either a built-in name (str) or a numpy array (custom/combined)."""
        voice_path: str
        voice_name: str

        if isinstance(voice, tuple):
            voice_name, voice_data = voice
            if isinstance(voice_data, str):
                if self._model and voice_data in self._model.get_voices():
                    return voice_data
                voice_path = voice_data
                voice_name = os.path.splitext(os.path.basename(voice_path))[0]
            else:
                # In-memory array (numpy or tensor): save to temp .npy and load with shape handling
                if hasattr(voice_data, "numpy"):
                    arr = voice_data.numpy()
                else:
                    arr = np.asarray(voice_data, dtype=np.float32)
                fd, voice_path = tempfile.mkstemp(suffix=".npy", prefix="voice_")
                os.close(fd)
                np.save(voice_path, arr)
                return _load_voice_array(voice_path)
        else:
            voice_path = voice
            voice_name = os.path.splitext(os.path.basename(voice_path))[0]

        if not self._model:
            raise RuntimeError("Model not loaded")

        if voice_name in self._model.get_voices():
            return voice_name

        return _load_voice_array(voice_path)

    async def generate_from_tokens(
        self,
        tokens: str,
        voice: Union[str, Tuple[str, Union[np.ndarray, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio from phoneme string.

        Args:
            tokens: Phoneme string to synthesize
            voice: Voice name/path or (name, path/array)
            speed: Speed multiplier
            lang_code: Optional; unused when is_phonemes=True

        Yields:
            Generated audio chunks (float32)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        voice_spec = self._resolve_voice_spec(voice)

        logger.debug(
            f"Generating audio from tokens: '{tokens[:100]}{'...' if len(tokens) > 100 else ''}'"
        )
        async for samples, _sample_rate in self._model.create_stream(
            tokens,
            voice=voice_spec,
            speed=speed,
            is_phonemes=True,
        ):
            if samples is not None and len(samples) > 0:
                yield samples
            else:
                logger.warning("No audio in chunk")

    async def generate(
        self,
        text: str,
        voice: Union[str, Tuple[str, Union[np.ndarray, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate audio from text. Word timestamps not supported with ONNX."""

        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        voice_spec = self._resolve_voice_spec(voice)

        if lang_code:
            pipeline_lang_code = lang_code
        elif settings.default_voice_code:
            pipeline_lang_code = settings.default_voice_code
        elif isinstance(voice, tuple):
            pipeline_lang_code = voice[0][0].lower() if voice[0] else "a"
        else:
            name = os.path.splitext(os.path.basename(voice))[0]
            pipeline_lang_code = name[0].lower() if name else "a"
        lang = lang_code_to_onnx(pipeline_lang_code)

        logger.debug(
            f"Generating audio for text with lang '{lang}': '{text[:100]}{'...' if len(text) > 100 else ''}'"
        )
        async for samples, _sample_rate in self._model.create_stream(
            text,
            voice=voice_spec,
            speed=speed,
            lang=lang,
            is_phonemes=False,
        ):
            if samples is not None and len(samples) > 0:
                yield AudioChunk(audio=samples, word_timestamps=None)
            else:
                logger.warning("No audio in chunk")

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._model is not None:
            del self._model
            self._model = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Get device string (ONNX uses CPU/cuda via provider)."""
        return self._device

    def get_voices(self) -> list[str]:
        """Return list of built-in voice names from the loaded model."""
        if not self._model:
            return []
        return self._model.get_voices()

    def get_onnx_provider(self) -> Optional[str]:
        """Return the ONNX Runtime execution provider currently in use (primary), or None if not loaded."""
        if not self._model or not hasattr(self._model, "sess"):
            return None
        sess = self._model.sess
        if not hasattr(sess, "get_providers"):
            return None
        providers = sess.get_providers()
        return providers[0] if providers else None
