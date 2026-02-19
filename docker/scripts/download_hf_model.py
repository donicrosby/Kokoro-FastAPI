#!/usr/bin/env python3
"""Pre-download Kokoro model from Hugging Face Hub at build or runtime.

Uses huggingface_hub only (no kokoro_onnx dependency). Populates the HF cache
and assembled voices so Kokoro.from_pretrained() finds them when the app runs.
"""

import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger


def main() -> int:
    repo = (os.environ.get("HF_MODEL_REPO") or "").strip()
    if not repo:
        logger.info("HF_MODEL_REPO not set; skipping Hugging Face model download")
        return 0

    model_filename = (os.environ.get("HF_MODEL_FILENAME") or "onnx/model.onnx").strip()
    voices_subdir = (os.environ.get("HF_VOICES_SUBDIR") or "").strip()
    voices_filename = (os.environ.get("HF_VOICES_FILENAME") or "").strip()
    revision = (os.environ.get("HF_MODEL_REVISION") or "").strip() or None
    cache_dir = (
        os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_CACHE_DIR") or ""
    ).strip() or None

    if bool(voices_subdir) == bool(voices_filename):
        logger.error(
            "Set exactly one of HF_VOICES_SUBDIR or HF_VOICES_FILENAME when HF_MODEL_REPO is set"
        )
        return 1

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        logger.error(f"huggingface_hub not available: {e}")
        return 1

    kwargs = {"repo_id": repo, "revision": revision, "cache_dir": cache_dir}

    logger.info(f"Downloading Hugging Face model: {repo} (cache_dir={cache_dir or 'default'})")

    # Download ONNX model (populates cache for from_pretrained)
    hf_hub_download(filename=model_filename, **kwargs)

    if voices_filename:
        hf_hub_download(filename=voices_filename, **kwargs)
    else:
        # Same layout as kokoro_onnx from_pretrained: assemble .bin files into voices.npz
        api = HfApi()
        prefix = voices_subdir.rstrip("/") + "/"
        files = api.list_repo_files(repo_id=repo, revision=revision)
        voice_files = [f for f in files if f.startswith(prefix) and f.endswith(".bin")]
        if not voice_files:
            logger.error(f"No .bin files under {prefix} in {repo}")
            return 1

        cache_base = Path(
            cache_dir
            or os.environ.get(
                "HF_HUB_CACHE",
                os.path.expanduser("~/.cache/huggingface/hub"),
            )
        )
        safe_repo = repo.replace("/", "--")
        rev = revision or "main"
        assembled_dir = cache_base / "kokoro_assembled_voices" / safe_repo / rev
        assembled_path = assembled_dir / "voices.npz"
        assembled_dir.mkdir(parents=True, exist_ok=True)

        if not assembled_path.exists():
            voices_dict = {}
            for rel_path in sorted(voice_files):
                path = hf_hub_download(filename=rel_path, **kwargs)
                stem = Path(rel_path).stem
                arr = np.fromfile(path, dtype=np.float32).reshape(-1, 1, 256)
                voices_dict[stem] = arr
            np.savez(assembled_path, **voices_dict)
            logger.info(f"Assembled {len(voices_dict)} voices into {assembled_path}")

    logger.info(f"Hugging Face model cached for {repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
