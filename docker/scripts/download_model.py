#!/usr/bin/env python3
"""Download and prepare Kokoro v1.0 ONNX model and voices."""

import os
from urllib.request import urlretrieve

from loguru import logger

# kokoro-onnx model-files-v1.0 release
BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
ONNX_MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"


def verify_onnx_files(model_path: str, voices_path: str) -> bool:
    """Verify that ONNX model and voices files exist and are valid.

    Args:
        model_path: Path to ONNX model file
        voices_path: Path to voices bundle file

    Returns:
        True if both files exist and have non-zero size
    """
    try:
        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            return False
        if not os.path.exists(voices_path) or os.path.getsize(voices_path) == 0:
            return False
        return True
    except Exception:
        return False


def download_model(output_dir: str) -> None:
    """Download ONNX model and voices from kokoro-onnx releases.

    Args:
        output_dir: Directory to save model files (e.g. api/src/models/v1_0)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, ONNX_MODEL_FILE)
        voices_path = os.path.join(output_dir, VOICES_FILE)

        if verify_onnx_files(model_path, voices_path):
            logger.info("ONNX model files already exist and are valid")
            return

        logger.info("Downloading Kokoro v1.0 ONNX model files")

        logger.info("Downloading model file...")
        urlretrieve(f"{BASE_URL}/{ONNX_MODEL_FILE}", model_path)

        logger.info("Downloading voices file...")
        urlretrieve(f"{BASE_URL}/{VOICES_FILE}", voices_path)

        if not verify_onnx_files(model_path, voices_path):
            raise RuntimeError("Failed to verify downloaded files")

        logger.info(f"✓ Model files prepared in {output_dir}")

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Kokoro v1.0 ONNX model and voices"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for model files"
    )

    args = parser.parse_args()
    download_model(args.output)


if __name__ == "__main__":
    main()
