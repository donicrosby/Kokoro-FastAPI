"""Language code mapping for kokoro-onnx and Misaki."""

# Single-letter or short codes (API) -> kokoro-onnx lang parameter
LANG_CODE_TO_ONNX: dict[str, str] = {
    "a": "en-us",
    "e": "en-us",
    "b": "en-gb",
    "z": "zh-cn",
    "j": "ja",
    "k": "ko",
    "en-us": "en-us",
    "en-gb": "en-gb",
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "ja": "ja",
    "ko": "ko",
}


def lang_code_to_onnx(lang_code: str) -> str:
    """Map internal lang code to kokoro-onnx lang string."""
    code = (lang_code or "a").lower().strip()
    return LANG_CODE_TO_ONNX.get(code, "en-us")
