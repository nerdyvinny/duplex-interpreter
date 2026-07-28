"""Optional cloud upgrades: DeepL translation and ElevenLabs voices.

Both are drop-in replacements for the OpenAI stages. DeepL is noticeably
better on European pairs and has a 500k-chars/month free tier; ElevenLabs
Flash v2.5 has the lowest synthesis latency of anything available.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from functools import lru_cache

from ..config import ConfigError, require_env
from .base import ProviderError, SpeechAudio, TranslationProvider, TTSProvider

log = logging.getLogger(__name__)

ELEVENLABS_SAMPLE_RATE = 24_000

# DeepL insists on a regional variant for these targets.
_DEEPL_TARGET_OVERRIDES = {
    "en": "EN-US",
    "pt": "PT-BR",
    "zh": "ZH-HANS",
}

# Public ElevenLabs voices, so a config written for OpenAI voice names still
# produces two distinguishable speakers instead of an error.
_ELEVENLABS_FALLBACK_VOICES = {
    "alloy": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "nova": "EXAVITQu4vr4xnSDxMaL",  # Bella
    "shimmer": "ThT5KcBeYPX3keUQqHPh",  # Dorothy
    "echo": "pNInz6obpgDQGcFmaJgB",  # Adam
    "onyx": "VR6AewLTigWG4xSOukaG",  # Arnold
    "sage": "TxGEqnHWrfWFTfGW9XjX",  # Josh
}


class DeepLTranslation(TranslationProvider):
    name = "deepl"

    def __init__(self, model: str | None = None) -> None:
        del model  # DeepL has no model selection
        try:
            import deepl  # noqa: F401
        except ImportError as exc:
            raise ConfigError(
                "providers.translation is 'deepl' but the package is missing. "
                "Run `pip install deepl` and set DEEPL_API_KEY in .env."
            ) from exc

    @staticmethod
    @lru_cache(maxsize=1)
    def _translator():
        import deepl

        return deepl.Translator(require_env("DEEPL_API_KEY", "deepl"))

    async def translate(
        self,
        text: str,
        *,
        source: str,
        target: str,
        context: list[str] | None = None,
    ) -> str:
        if not text.strip():
            return ""

        target_code = _DEEPL_TARGET_OVERRIDES.get(target, target.upper())
        # DeepL's context parameter improves consistency without being
        # translated itself — exactly what we want for conversation history.
        context_text = "\n".join(context[-3:]) if context else None

        def _call() -> str:
            try:
                result = self._translator().translate_text(
                    text,
                    source_lang=source.upper(),
                    target_lang=target_code,
                    context=context_text,
                    preserve_formatting=True,
                )
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(f"DeepL translation failed: {exc}") from exc
            return str(result.text).strip()

        return await asyncio.to_thread(_call)


class ElevenLabsTTS(TTSProvider):
    name = "elevenlabs"

    def __init__(self, model: str = "eleven_flash_v2_5") -> None:
        self.model = model
        try:
            import elevenlabs  # noqa: F401
        except ImportError as exc:
            raise ConfigError(
                "providers.tts is 'elevenlabs' but the package is missing. "
                "Run `pip install elevenlabs` and set ELEVENLABS_API_KEY in .env."
            ) from exc

    @staticmethod
    @lru_cache(maxsize=1)
    def _client():
        from elevenlabs.client import AsyncElevenLabs

        return AsyncElevenLabs(api_key=require_env("ELEVENLABS_API_KEY", "elevenlabs"))

    def _resolve_voice(self, voice: str) -> str:
        mapped = _ELEVENLABS_FALLBACK_VOICES.get(voice.lower())
        if mapped:
            log.debug("mapped voice %r to ElevenLabs voice %s", voice, mapped)
            return mapped
        return voice  # assume it is already a voice ID

    async def synthesize(self, text: str, *, language: str, voice: str) -> SpeechAudio:
        return SpeechAudio(
            chunks=self._stream(text, self._resolve_voice(voice), language),
            sample_rate=ELEVENLABS_SAMPLE_RATE,
        )

    async def _stream(self, text: str, voice_id: str, language: str) -> AsyncIterator[bytes]:
        remainder = b""
        try:
            stream = self._client().text_to_speech.stream(
                text=text,
                voice_id=voice_id,
                model_id=self.model,
                output_format="pcm_24000",
                language_code=language,
            )
            async for chunk in stream:
                if not chunk:
                    continue
                data = remainder + chunk
                usable = len(data) - (len(data) % 2)
                remainder = data[usable:]
                if usable:
                    yield data[:usable]
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"ElevenLabs synthesis failed: {exc}") from exc
