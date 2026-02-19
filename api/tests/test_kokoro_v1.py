from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from api.src.inference.kokoro_v1 import KokoroV1, _ensure_voice_shape


@pytest.fixture
def kokoro_backend():
    """Create a KokoroV1 instance for testing."""
    return KokoroV1()


def test_initial_state(kokoro_backend):
    """Test initial state of KokoroV1."""
    assert not kokoro_backend.is_loaded
    assert kokoro_backend._model is None
    assert kokoro_backend.device == "cpu"


def test_ensure_voice_shape_1d():
    """(256,) -> (511, 1, 256)."""
    arr = np.zeros(256, dtype=np.float32)
    out = _ensure_voice_shape(arr)
    assert out.shape == (511, 1, 256)
    assert out.dtype == np.float32


def test_ensure_voice_shape_2d():
    """(1, 256) -> (511, 1, 256)."""
    arr = np.zeros((1, 256), dtype=np.float32)
    out = _ensure_voice_shape(arr)
    assert out.shape == (511, 1, 256)


@pytest.mark.asyncio
async def test_load_model_validation(kokoro_backend):
    """Test model loading raises when Kokoro constructor fails."""
    with patch("api.src.inference.kokoro_v1.paths.get_onnx_model_path", new_callable=AsyncMock, return_value="/nonexistent.onnx"):
        with patch("api.src.inference.kokoro_v1.paths.get_onnx_voices_path", new_callable=AsyncMock, return_value="/nonexistent.bin"):
            with patch("api.src.inference.kokoro_v1.Kokoro", side_effect=FileNotFoundError("missing")):
                with pytest.raises(FileNotFoundError, match="missing"):
                    await kokoro_backend.load_model("")


@pytest.mark.asyncio
async def test_load_model_success(kokoro_backend):
    """Test model loading with mocked paths and Kokoro (path-based when HF not set)."""
    mock_instance = MagicMock()
    mock_kokoro_class = MagicMock(return_value=mock_instance)
    with patch("api.src.inference.kokoro_v1.paths.get_onnx_model_path", new_callable=AsyncMock, return_value="/tmp/kokoro.onnx"):
        with patch("api.src.inference.kokoro_v1.paths.get_onnx_voices_path", new_callable=AsyncMock, return_value="/tmp/voices.bin"):
            with patch("api.src.inference.kokoro_v1.Kokoro", mock_kokoro_class):
                with patch("api.src.inference.kokoro_v1.settings") as mock_settings:
                    mock_settings.hf_model_repo = None
                    await kokoro_backend.load_model("")
    assert kokoro_backend.is_loaded
    assert kokoro_backend._model is mock_instance
    mock_kokoro_class.assert_called_once_with("/tmp/kokoro.onnx", "/tmp/voices.bin")


@pytest.mark.asyncio
async def test_load_model_uses_from_pretrained_when_hf_repo_set(kokoro_backend):
    """When settings.hf_model_repo is set, load_model calls _load_from_pretrained and does not use paths."""
    mock_instance = MagicMock()
    with patch("api.src.inference.kokoro_v1.settings") as mock_settings:
        mock_settings.hf_model_repo = "onnx-community/Kokoro-82M-v1.0-ONNX"
        mock_settings.hf_model_filename = "onnx/model.onnx"
        mock_settings.hf_voices_subdir = "voices"
        mock_settings.hf_voices_filename = None
        mock_settings.hf_revision = None
        mock_settings.hf_cache_dir = None
        with patch.object(kokoro_backend, "_load_from_pretrained", new_callable=AsyncMock) as mock_load_hf:
            await kokoro_backend.load_model("")
    mock_load_hf.assert_awaited_once()
    assert kokoro_backend._model is None  # _load_from_pretrained was mocked, so model not set
    # Paths should not have been called (no get_onnx_model_path / get_onnx_voices_path)


@pytest.mark.asyncio
async def test_load_from_pretrained_builds_kwargs_and_calls_from_pretrained(kokoro_backend):
    """_load_from_pretrained builds correct kwargs and calls Kokoro.from_pretrained in executor."""
    mock_model = MagicMock()
    from_pretrained_called = []

    def capture_from_pretrained(**kwargs):
        from_pretrained_called.append(kwargs)
        return mock_model

    async def fake_run_in_executor(executor, fn):
        return fn()

    with patch("api.src.inference.kokoro_v1.settings") as mock_settings:
        mock_settings.hf_model_repo = "onnx-community/Kokoro-82M-v1.0-ONNX"
        mock_settings.hf_model_filename = "onnx/model.onnx"
        mock_settings.hf_voices_subdir = "voices"
        mock_settings.hf_voices_filename = None
        mock_settings.hf_revision = "main"
        mock_settings.hf_cache_dir = "/cache"
        with patch("api.src.inference.kokoro_v1.Kokoro") as mock_kokoro_class:
            mock_kokoro_class.from_pretrained = capture_from_pretrained
            with patch("api.src.inference.kokoro_v1.asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(side_effect=fake_run_in_executor)
                await kokoro_backend._load_from_pretrained()
    assert len(from_pretrained_called) == 1
    kwargs = from_pretrained_called[0]
    assert kwargs["repo_id"] == "onnx-community/Kokoro-82M-v1.0-ONNX"
    assert kwargs["model_filename"] == "onnx/model.onnx"
    assert kwargs["voices_subdir"] == "voices"
    assert "voices_filename" not in kwargs
    assert kwargs["revision"] == "main"
    assert kwargs["cache_dir"] == "/cache"
    assert kokoro_backend._model is mock_model


@pytest.mark.asyncio
async def test_load_from_pretrained_with_voices_filename(kokoro_backend):
    """_load_from_pretrained passes voices_filename when subdir is not set."""
    from_pretrained_called = []

    def capture_from_pretrained(**kwargs):
        from_pretrained_called.append(kwargs)
        return MagicMock()

    async def fake_run_in_executor(executor, fn):
        return fn()

    with patch("api.src.inference.kokoro_v1.settings") as mock_settings:
        mock_settings.hf_model_repo = "user/repo"
        mock_settings.hf_model_filename = "model.onnx"
        mock_settings.hf_voices_subdir = None
        mock_settings.hf_voices_filename = "data/voices.bin"
        mock_settings.hf_revision = None
        mock_settings.hf_cache_dir = None
        with patch("api.src.inference.kokoro_v1.Kokoro") as mock_kokoro_class:
            mock_kokoro_class.from_pretrained = capture_from_pretrained
            with patch("api.src.inference.kokoro_v1.asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(side_effect=fake_run_in_executor)
                await kokoro_backend._load_from_pretrained()
    assert len(from_pretrained_called) == 1
    kwargs = from_pretrained_called[0]
    assert kwargs["voices_filename"] == "data/voices.bin"
    assert "voices_subdir" not in kwargs


def test_unload(kokoro_backend):
    """Test model unloading."""
    kokoro_backend._model = MagicMock()
    assert kokoro_backend.is_loaded
    kokoro_backend.unload()
    assert not kokoro_backend.is_loaded
    assert kokoro_backend._model is None


@pytest.mark.asyncio
async def test_generate_validation(kokoro_backend):
    """Test generate raises when model not loaded."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        async for _ in kokoro_backend.generate("test", "voice"):
            pass


@pytest.mark.asyncio
async def test_generate_from_tokens_validation(kokoro_backend):
    """Test generate_from_tokens raises when model not loaded."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        async for _ in kokoro_backend.generate_from_tokens("test tokens", "voice"):
            pass


@pytest.mark.asyncio
async def test_generate_uses_create_stream_and_yields_chunks_without_timestamps(kokoro_backend):
    """Test generate uses create_stream and yields AudioChunk with word_timestamps=None."""
    mock_kokoro = MagicMock()
    chunk_audio = np.zeros(1600, dtype=np.float32)

    async def fake_stream(*args, **kwargs):
        yield (chunk_audio, 24000)

    mock_kokoro.create_stream = MagicMock(side_effect=fake_stream)
    mock_kokoro.get_voices.return_value = ["af_heart", "af_bella"]
    kokoro_backend._model = mock_kokoro

    chunks = []
    async for chunk in kokoro_backend.generate("hello", "af_heart"):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].word_timestamps is None
    assert np.array_equal(chunks[0].audio, chunk_audio)
    mock_kokoro.create_stream.assert_called_once()
    call_kw = mock_kokoro.create_stream.call_args[1]
    assert call_kw["is_phonemes"] is False
    assert "lang" in call_kw


@pytest.mark.asyncio
async def test_generate_from_tokens_uses_create_stream(kokoro_backend):
    """Test generate_from_tokens uses create_stream with is_phonemes=True."""
    mock_kokoro = MagicMock()
    chunk_audio = np.ones(800, dtype=np.float32)

    async def fake_stream(*args, **kwargs):
        yield (chunk_audio, 24000)

    mock_kokoro.create_stream = MagicMock(side_effect=fake_stream)
    mock_kokoro.get_voices.return_value = ["af_bella"]
    kokoro_backend._model = mock_kokoro

    chunks = []
    async for chunk in kokoro_backend.generate_from_tokens("test phonemes", "af_bella"):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], chunk_audio)
    mock_kokoro.create_stream.assert_called_once()
    call_kw = mock_kokoro.create_stream.call_args[1]
    assert call_kw["is_phonemes"] is True


def test_get_voices_empty_when_not_loaded(kokoro_backend):
    """Test get_voices returns [] when model not loaded."""
    assert kokoro_backend.get_voices() == []


def test_get_voices_delegates_to_model(kokoro_backend):
    """Test get_voices returns model.get_voices()."""
    mock_kokoro = MagicMock()
    mock_kokoro.get_voices.return_value = ["af_heart", "af_bella"]
    kokoro_backend._model = mock_kokoro
    assert kokoro_backend.get_voices() == ["af_heart", "af_bella"]
