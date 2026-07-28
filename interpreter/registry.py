"""Maps provider names in config.yaml to concrete classes.

Imports are deferred into the factory functions so that installing only the
cloud requirements does not fail on a missing `faster_whisper`, and vice
versa.
"""

from __future__ import annotations

from .config import ConfigError, ProvidersConfig
from .providers.base import STTProvider, TranslationProvider, TTSProvider

STT_CHOICES = ("openai", "faster-whisper", "fake")
TRANSLATION_CHOICES = ("openai", "deepl", "argos", "passthrough", "fake")
TTS_CHOICES = ("openai", "elevenlabs", "piper", "silent", "fake")


def _unknown(kind: str, name: str, choices: tuple[str, ...]) -> ConfigError:
    return ConfigError(
        f"providers.{kind}={name!r} is not recognised. Choose one of: {', '.join(choices)}"
    )


def build_stt(cfg: ProvidersConfig) -> STTProvider:
    name = cfg.stt.strip().lower()
    if name == "openai":
        from .providers.openai_provider import OpenAISTT

        return OpenAISTT(cfg.stt_model)
    if name in {"faster-whisper", "faster_whisper", "local"}:
        from .providers.local_provider import FasterWhisperSTT

        return FasterWhisperSTT(cfg)
    if name == "fake":
        from .providers.fake import FakeSTT

        return FakeSTT()
    raise _unknown("stt", name, STT_CHOICES)


def build_translation(cfg: ProvidersConfig) -> TranslationProvider:
    name = cfg.translation.strip().lower()
    if name == "openai":
        from .providers.openai_provider import OpenAITranslation

        return OpenAITranslation(cfg.translation_model)
    if name == "deepl":
        from .providers.cloud_extras import DeepLTranslation

        return DeepLTranslation()
    if name == "argos":
        from .providers.local_provider import ArgosTranslation

        return ArgosTranslation()
    if name == "passthrough":
        from .providers.local_provider import PassthroughTranslation

        return PassthroughTranslation()
    if name == "fake":
        from .providers.fake import FakeTranslation

        return FakeTranslation()
    raise _unknown("translation", name, TRANSLATION_CHOICES)


def build_tts(cfg: ProvidersConfig) -> TTSProvider:
    name = cfg.tts.strip().lower()
    if name == "openai":
        from .providers.openai_provider import OpenAITTS

        return OpenAITTS(cfg.tts_model)
    if name == "elevenlabs":
        from .providers.cloud_extras import ElevenLabsTTS

        return ElevenLabsTTS()
    if name == "piper":
        from .providers.local_provider import PiperTTS

        return PiperTTS()
    if name == "silent":
        from .providers.local_provider import SilentTTS

        return SilentTTS()
    if name == "fake":
        from .providers.fake import FakeTTS

        return FakeTTS()
    raise _unknown("tts", name, TTS_CHOICES)


def build_all(cfg: ProvidersConfig) -> tuple[STTProvider, TranslationProvider, TTSProvider]:
    return build_stt(cfg), build_translation(cfg), build_tts(cfg)
