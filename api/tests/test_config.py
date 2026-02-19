"""Tests for application config (Settings) and HF model validation."""

import pytest

from api.src.core.config import Settings


def test_settings_hf_defaults():
    """HF-related settings default to None or expected values when not set."""
    s = Settings()
    assert s.hf_model_repo is None
    assert s.hf_model_filename == "onnx/model.onnx"
    assert s.hf_voices_subdir is None
    assert s.hf_voices_filename is None
    assert s.hf_revision is None
    assert s.hf_cache_dir is None


def test_settings_hf_repo_with_voices_subdir_valid():
    """When HF_MODEL_REPO is set, exactly one of voices_subdir or voices_filename is valid (subdir)."""
    s = Settings(
        hf_model_repo="onnx-community/Kokoro-82M-v1.0-ONNX",
        hf_voices_subdir="voices",
        hf_voices_filename=None,
    )
    assert s.hf_model_repo == "onnx-community/Kokoro-82M-v1.0-ONNX"
    assert s.hf_voices_subdir == "voices"
    assert s.hf_voices_filename is None


def test_settings_hf_repo_with_voices_filename_valid():
    """When HF_MODEL_REPO is set, exactly one of voices_subdir or voices_filename is valid (filename)."""
    s = Settings(
        hf_model_repo="user/repo",
        hf_voices_subdir=None,
        hf_voices_filename="voices/voices.bin",
    )
    assert s.hf_voices_filename == "voices/voices.bin"
    assert s.hf_voices_subdir is None


def test_settings_hf_repo_with_both_voices_invalid():
    """When HF_MODEL_REPO is set, both voices_subdir and voices_filename raises ValueError."""
    with pytest.raises(ValueError, match="exactly one of HF_VOICES_SUBDIR or HF_VOICES_FILENAME"):
        Settings(
            hf_model_repo="user/repo",
            hf_voices_subdir="voices",
            hf_voices_filename="voices.bin",
        )


def test_settings_hf_repo_with_neither_voices_invalid():
    """When HF_MODEL_REPO is set, neither voices_subdir nor voices_filename raises ValueError."""
    with pytest.raises(ValueError, match="exactly one of HF_VOICES_SUBDIR or HF_VOICES_FILENAME"):
        Settings(
            hf_model_repo="user/repo",
            hf_voices_subdir=None,
            hf_voices_filename=None,
        )


def test_settings_hf_repo_empty_string_treated_as_unset():
    """Empty hf_model_repo does not trigger voices validation (no error)."""
    s = Settings(
        hf_model_repo="",
        hf_voices_subdir=None,
        hf_voices_filename=None,
    )
    assert s.hf_model_repo == ""


def test_settings_hf_optional_fields():
    """hf_revision and hf_cache_dir can be set with HF repo."""
    s = Settings(
        hf_model_repo="user/repo",
        hf_voices_subdir="voices",
        hf_revision="main",
        hf_cache_dir="/tmp/hf_cache",
    )
    assert s.hf_revision == "main"
    assert s.hf_cache_dir == "/tmp/hf_cache"
