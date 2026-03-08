"""Language configuration and utilities for multilingual TTS evaluation."""

from dataclasses import dataclass
from typing import Dict, Optional, Callable
from collections import defaultdict

try:
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
    from transformers import WhisperTokenizer
    NORMALIZERS_AVAILABLE = True
except ImportError:
    NORMALIZERS_AVAILABLE = False


@dataclass
class LanguageConfig:
    """Configuration for a language."""
    code: str  # e.g., "en", "ja", "es"
    name: str  # e.g., "English", "Japanese", "Spanish"
    whisper_code: str  # Whisper language code
    soniox_code: str  # Soniox language code
    normalizer: Optional[Callable] = None
    script_type: str = "latin"  # "latin", "cjk", "arabic", etc.


# Supported languages with their configurations
LANGUAGE_CONFIGS = {
    "en": LanguageConfig(
        code="en",
        name="English",
        whisper_code="en",
        soniox_code="en",
        script_type="latin"
    ),
    "ja": LanguageConfig(
        code="ja",
        name="Japanese",
        whisper_code="ja",
        soniox_code="ja",
        script_type="cjk"
    ),
    "es": LanguageConfig(
        code="es",
        name="Spanish",
        whisper_code="es",
        soniox_code="es",
        script_type="latin"
    ),
    "fr": LanguageConfig(
        code="fr",
        name="French",
        whisper_code="fr",
        soniox_code="fr",
        script_type="latin"
    ),
    "de": LanguageConfig(
        code="de",
        name="German",
        whisper_code="de",
        soniox_code="de",
        script_type="latin"
    ),
    "zh": LanguageConfig(
        code="zh",
        name="Chinese (Mandarin)",
        whisper_code="zh",
        soniox_code="zh",
        script_type="cjk"
    ),
    "ko": LanguageConfig(
        code="ko",
        name="Korean",
        whisper_code="ko",
        soniox_code="ko",
        script_type="hangul"
    ),
    "pt": LanguageConfig(
        code="pt",
        name="Portuguese",
        whisper_code="pt",
        soniox_code="pt",
        script_type="latin"
    ),
    "it": LanguageConfig(
        code="it",
        name="Italian",
        whisper_code="it",
        soniox_code="it",
        script_type="latin"
    ),
    "ru": LanguageConfig(
        code="ru",
        name="Russian",
        whisper_code="ru",
        soniox_code="ru",
        script_type="cyrillic"
    ),
    "ar": LanguageConfig(
        code="ar",
        name="Arabic",
        whisper_code="ar",
        soniox_code="ar",
        script_type="arabic"
    ),
    "hi": LanguageConfig(
        code="hi",
        name="Hindi",
        whisper_code="hi",
        soniox_code="hi",
        script_type="devanagari"
    ),
}


def get_language_config(lang_code: str) -> LanguageConfig:
    """Get language configuration by code."""
    if lang_code not in LANGUAGE_CONFIGS:
        raise ValueError(f"Unsupported language: {lang_code}. Supported: {list(LANGUAGE_CONFIGS.keys())}")
    return LANGUAGE_CONFIGS[lang_code]


def build_normalizers(whisper_model_id: str = "kotoba-tech/kotoba-whisper-v2.0") -> Dict[str, Callable]:
    """Build text normalizers for supported languages."""
    if not NORMALIZERS_AVAILABLE:
        # Return identity normalizers if transformers not available
        return defaultdict(lambda: lambda x: x.strip().lower())

    basic_normalizer = BasicTextNormalizer()
    normalizers = defaultdict(lambda: basic_normalizer)

    # English normalizer
    try:
        en_normalizer = EnglishTextNormalizer(
            WhisperTokenizer.from_pretrained(whisper_model_id).english_spelling_normalizer
        )
        normalizers["en"] = en_normalizer
    except Exception as e:
        print(f"Warning: Could not load English normalizer: {e}")

    # Japanese normalizer - remove spaces and normalize punctuation
    normalizers["ja"] = lambda x: basic_normalizer(x).replace(" ", "").replace("。.", "。")

    # Chinese normalizer - remove spaces
    normalizers["zh"] = lambda x: basic_normalizer(x).replace(" ", "")

    # Korean normalizer
    normalizers["ko"] = lambda x: basic_normalizer(x).replace(" ", "")

    return normalizers


def get_supported_languages() -> Dict[str, str]:
    """Get all supported languages."""
    return {code: config.name for code, config in LANGUAGE_CONFIGS.items()}
