"""OpenAI providers: STT, translation and TTS from one API key.

This is the default path. `gpt-4o-mini-transcribe` for speech, `gpt-4o-mini`
for translation, `gpt-4o-mini-tts` for voice — roughly $0.02-0.04 per minute
of conversation.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from functools import lru_cache

import numpy as np

from ..config import ConfigError, require_env
from ..langid import normalize_language_name
from ..languages import name_of
from .base import (
    ProviderError,
    SpeechAudio,
    STTProvider,
    Transcript,
    TranslationProvider,
    TTSProvider,
    pcm_to_wav_bytes,
)

log = logging.getLogger(__name__)

# gpt-4o-mini-tts emits 24 kHz mono signed 16-bit little-endian for "pcm".
OPENAI_TTS_SAMPLE_RATE = 24_000


@lru_cache(maxsize=1)
def _client():
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ConfigError(
            "the `openai` package is not installed; run `pip install -r requirements.txt`"
        ) from exc

    api_key = require_env("OPENAI_API_KEY", "openai")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    # Conversation turns are small; failing fast and retrying beats a request
    # that hangs while the other person waits.
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=2)


class OpenAISTT(STTProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini-transcribe") -> None:
        self.model = model
        # Only the whisper-* models expose verbose_json, which is the only way
        # to get the audio-detected language back from the API.
        self._supports_verbose = model.startswith("whisper")

    def preflight(self, cfg) -> None:
        del cfg
        _client()

    async def transcribe(
        self,
        pcm: np.ndarray,
        *,
        language: str | None = None,
        candidates: tuple[str, ...] = (),
    ) -> Transcript:
        # Outside the try below: a missing key is a config problem to fix, not
        # a transient failure worth retrying.
        client = _client()
        audio = pcm_to_wav_bytes(pcm)
        request: dict = {
            "model": self.model,
            "file": ("utterance.wav", audio, "audio/wav"),
        }
        if language:
            request["language"] = language
        if self._supports_verbose:
            request["response_format"] = "verbose_json"
        if candidates and not language:
            # A hint, not a constraint — nudges the model toward the two
            # languages actually in play.
            request["prompt"] = "Conversation in " + " and ".join(
                name_of(c) for c in candidates
            )

        try:
            response = await client.audio.transcriptions.create(**request)
        except Exception as exc:  # noqa: BLE001 - surfaced as an event, not a crash
            raise ProviderError(f"OpenAI transcription failed: {exc}") from exc

        text = (getattr(response, "text", "") or "").strip()
        detected = normalize_language_name(getattr(response, "language", None)) or language
        return Transcript(text=text, language=detected)


class OpenAITranslation(TranslationProvider):
    name = "openai"

    _SYSTEM = (
        "You are a simultaneous interpreter in a live spoken conversation. "
        "Translate the user's message from {source} into {target}.\n"
        "Rules:\n"
        "- Output ONLY the translation. No preamble, no quotes, no notes, no "
        "romanization, no alternatives.\n"
        "- Preserve register and tone: keep casual speech casual and formal "
        "speech formal.\n"
        "- Keep proper nouns, names, numbers and units exactly as spoken.\n"
        "- Speech recognition makes mistakes; if a word is garbled, translate "
        "the most plausible intended meaning rather than the literal noise.\n"
        "- If the message is already in {target}, repeat it unchanged.\n"
        "- Never answer, explain, or comment on the message. You are a "
        "conduit, not a participant."
    )

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def preflight(self, cfg) -> None:
        del cfg
        _client()

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

        client = _client()
        messages: list[dict] = [
            {
                "role": "system",
                "content": self._SYSTEM.format(source=name_of(source), target=name_of(target)),
            }
        ]
        if context:
            # Recent turns let the model resolve pronouns and grammatical
            # gender that a bare sentence leaves ambiguous.
            messages.append(
                {
                    "role": "system",
                    "content": "Conversation so far, for context only:\n" + "\n".join(context),
                }
            )
        messages.append({"role": "user", "content": text})

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=400,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI translation failed: {exc}") from exc

        translated = (response.choices[0].message.content or "").strip()
        if not translated:
            raise ProviderError("OpenAI translation returned nothing")
        return translated.strip('"').strip()


class OpenAITTS(TTSProvider):
    name = "openai"

    _INSTRUCTIONS = (
        "Speak naturally and conversationally, at a brisk but clear pace, as "
        "though relaying what someone just said to you."
    )

    def __init__(self, model: str = "gpt-4o-mini-tts") -> None:
        self.model = model

    def preflight(self, cfg) -> None:
        del cfg
        _client()

    async def synthesize(self, text: str, *, language: str, voice: str) -> SpeechAudio:
        client = _client()
        return SpeechAudio(
            chunks=self._stream(client, text, voice), sample_rate=OPENAI_TTS_SAMPLE_RATE
        )

    async def _stream(self, client, text: str, voice: str) -> AsyncIterator[bytes]:
        request: dict = {
            "model": self.model,
            "voice": voice,
            "input": text,
            "response_format": "pcm",
        }
        # Only the gpt-4o-*-tts family accepts delivery instructions.
        if "gpt-4o" in self.model:
            request["instructions"] = self._INSTRUCTIONS

        remainder = b""
        try:
            async with client.audio.speech.with_streaming_response.create(
                **request
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=4096):
                    if not chunk:
                        continue
                    # int16 samples must not be split across chunk boundaries.
                    data = remainder + chunk
                    usable = len(data) - (len(data) % 2)
                    remainder = data[usable:]
                    if usable:
                        yield data[:usable]
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI speech synthesis failed: {exc}") from exc
