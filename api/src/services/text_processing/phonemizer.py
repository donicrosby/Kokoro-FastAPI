"""Phonemization backends: Misaki (primary) and Espeak (fallback)."""

import re
from abc import ABC, abstractmethod
from typing import Dict, Optional


phonemizers: Dict[str, "PhonemizerBackend"] = {}


class PhonemizerBackend(ABC):
    """Abstract base class for phonemization backends."""

    @abstractmethod
    def phonemize(self, text: str) -> str:
        """Convert text to phonemes."""
        pass


class MisakiBackend(PhonemizerBackend):
    """Misaki G2P backend (recommended for Kokoro/kokoro-onnx)."""

    def __init__(self, language: str):
        """Create G2P for the given language code (a, b, e, z, j, k, en-us, etc.)."""
        self.language = language.lower().strip()
        self._g2p = self._create_g2p()

    def _create_g2p(self):
        if self.language in ("a", "e", "en-us"):
            from misaki import en, espeak
            try:
                fallback = espeak.EspeakFallback(british=False)
            except Exception:
                fallback = None
            return en.G2P(trf=False, british=False, fallback=fallback)
        if self.language == "b" or self.language == "en-gb":
            from misaki import en, espeak
            try:
                fallback = espeak.EspeakFallback(british=True)
            except Exception:
                fallback = None
            return en.G2P(trf=False, british=True, fallback=fallback)
        if self.language in ("z", "zh", "zh-cn"):
            from misaki import zh
            return zh.G2P()
        if self.language in ("j", "ja"):
            from misaki import ja
            return ja.G2P()
        if self.language in ("k", "ko"):
            from misaki import ko
            return ko.G2P()
        # Default to US English
        from misaki import en, espeak
        try:
            fallback = espeak.EspeakFallback(british=False)
        except Exception:
            fallback = None
        return en.G2P(trf=False, british=False, fallback=fallback)

    def phonemize(self, text: str) -> str:
        """Convert text to phonemes using Misaki."""
        phonemes, _ = self._g2p(text)
        return (phonemes or "").strip()


class EspeakBackend(PhonemizerBackend):
    """Espeak-based phonemizer (fallback when Misaki not used)."""

    def __init__(self, language: str):
        import phonemizer
        self.backend = phonemizer.backend.EspeakBackend(
            language=language, preserve_punctuation=True, with_stress=True
        )
        self.language = language

    def phonemize(self, text: str) -> str:
        ps = self.backend.phonemize([text])
        ps = ps[0] if ps else ""
        ps = ps.replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ").replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ")
        ps = ps.replace("ʲ", "j").replace("r", "ɹ").replace("x", "k").replace("ɬ", "l")
        ps = re.sub(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)", " ", ps)
        ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»"" ]|$)', "z", ps)
        if self.language == "en-us":
            ps = re.sub(r"(?<=nˈaɪn)ti(?!ː)", "di", ps)
        return ps.strip()


def create_phonemizer(language: str = "a") -> PhonemizerBackend:
    """Create phonemizer backend for the given language code.

    Language codes: a, e = en-us; b = en-gb; z = zh; j = ja; k = ko.
    """
    lang = (language or "a").lower().strip()
    try:
        return MisakiBackend(lang)
    except Exception:
        lang_map = {"a": "en-us", "b": "en-gb", "z": "z"}
        if lang not in lang_map:
            raise ValueError(f"Unsupported language code: {language}")
        return EspeakBackend(lang_map[lang])


def phonemize(text: str, language: str = "a") -> str:
    """Convert text to phonemes. Uses Misaki when available."""
    global phonemizers
    text = text.strip()
    if language not in phonemizers:
        phonemizers[language] = create_phonemizer(language)
    return phonemizers[language].phonemize(text).strip()
