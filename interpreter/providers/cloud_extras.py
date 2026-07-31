# optional upgrades: deepl for translating and elevenlabs for the voice
#
# both just drop in place of the openai versions. deepl is noticeably
# better between european languages and has a free tier (500k characters a
# month), elevenlabs flash is the fastest tts I could find

import asyncio
import logging
from functools import lru_cache

from ..config import ConfigError, require_env
from .base import ProviderError, SpeechAudio, TranslationProvider, TTSProvider

log = logging.getLogger(__name__)

ELEVENLABS_SAMPLE_RATE = 24_000

# deepl won't accept plain "en" or "pt", it wants a region on these
_DEEPL_TARGET_OVERRIDES = {
    "en": "EN-US",
    "pt": "PT-BR",
    "zh": "ZH-HANS",
}

# elevenlabs uses long id strings instead of names. if somebody wrote an
# openai voice name in their config I map it to a real public voice so
# they still get two different sounding people instead of an error
_ELEVENLABS_FALLBACK_VOICES = {
    "alloy": "21m00Tcm4TlvDq8ikWAM",   # Rachel
    "nova": "EXAVITQu4vr4xnSDxMaL",    # Bella
    "shimmer": "ThT5KcBeYPX3keUQqHPh",  # Dorothy
    "echo": "pNInz6obpgDQGcFmaJgB",    # Adam
    "onyx": "VR6AewLTigWG4xSOukaG",    # Arnold
    "sage": "TxGEqnHWrfWFTfGW9XjX",    # Josh
}


class DeepLTranslation(TranslationProvider):
    name = "deepl"

    def __init__(self, model=None):
        # deepl has no model choice, the argument is just so it matches
        # the shape of the other providers
        try:
            import deepl
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

    async def translate(self, text, *, source, target, context=None):
        if not text.strip():
            return ""

        target_code = _DEEPL_TARGET_OVERRIDES.get(target, target.upper())

        # deepl's context parameter is nice, it reads the history to get
        # the tone right but doesn't translate it
        if context:
            context_text = "\n".join(context[-3:])
        else:
            context_text = None

        # the deepl library is not async so it has to go on a thread,
        # otherwise it blocks the audio
        def _call():
            try:
                result = self._translator().translate_text(
                    text,
                    source_lang=source.upper(),
                    target_lang=target_code,
                    context=context_text,
                    preserve_formatting=True,
                )
            except Exception as exc:
                raise ProviderError(f"DeepL translation failed: {exc}") from exc
            return str(result.text).strip()

        return await asyncio.to_thread(_call)


class ElevenLabsTTS(TTSProvider):
    name = "elevenlabs"

    def __init__(self, model="eleven_flash_v2_5"):
        self.model = model
        try:
            import elevenlabs
        except ImportError as exc:
            raise ConfigError(
                "providers.tts is 'elevenlabs' but the package is missing. "
                "Run `pip install elevenlabs` and set ELEVENLABS_API_KEY in .env."
            ) from exc

    @staticmethod
    @lru_cache(maxsize=1)
    def _client():
        from elevenlabs.client import AsyncElevenLabs
        return AsyncElevenLabs(
            api_key=require_env("ELEVENLABS_API_KEY", "elevenlabs")
        )

    def _resolve_voice(self, voice):
        mapped = _ELEVENLABS_FALLBACK_VOICES.get(voice.lower())
        if mapped:
            log.debug("mapped voice %r to ElevenLabs voice %s", voice, mapped)
            return mapped
        return voice  # assume they already put a real voice id in

    async def synthesize(self, text, *, language, voice):
        return SpeechAudio(
            chunks=self._stream(text, self._resolve_voice(voice), language),
            sample_rate=ELEVENLABS_SAMPLE_RATE,
        )

    async def _stream(self, text, voice_id, language):
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
                # same 2 byte alignment thing as the openai one
                data = remainder + chunk
                usable = len(data) - (len(data) % 2)
                remainder = data[usable:]
                if usable:
                    yield data[:usable]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(f"ElevenLabs synthesis failed: {exc}") from exc
