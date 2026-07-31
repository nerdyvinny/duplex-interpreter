# openai for all three stages, one api key covers everything
#
# this is the default. gpt-4o-mini-transcribe for the speech,
# gpt-4o-mini for the translating, gpt-4o-mini-tts for the voice.
# works out to something like 2 to 4 cents a minute of talking

import asyncio
import logging
import os
from functools import lru_cache

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

# what gpt-4o-mini-tts gives back when you ask for "pcm"
OPENAI_TTS_SAMPLE_RATE = 24_000


@lru_cache(maxsize=1)
def _client():
    # lru_cache so we only build one client no matter how many times this
    # gets called
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ConfigError(
            "the `openai` package is not installed; run `pip install -r requirements.txt`"
        ) from exc

    api_key = require_env("OPENAI_API_KEY", "openai")
    base_url = os.environ.get("OPENAI_BASE_URL") or None

    # short timeout on purpose. these are tiny requests and if one hangs
    # the other person is just standing there waiting, so failing fast and
    # retrying is better
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=2)


class OpenAISTT(STTProvider):
    name = "openai"

    def __init__(self, model="gpt-4o-mini-transcribe"):
        self.model = model
        # only the whisper-* models support verbose_json, and thats the
        # only way to get the detected language back out of the api
        self._supports_verbose = model.startswith("whisper")

    def preflight(self, cfg):
        _client()

    async def transcribe(self, pcm, *, language=None, candidates=()):
        # this is outside the try below on purpose. a missing api key is
        # not a network blip, its something you have to go fix
        client = _client()

        audio = pcm_to_wav_bytes(pcm)
        request = {
            "model": self.model,
            "file": ("utterance.wav", audio, "audio/wav"),
        }
        if language:
            request["language"] = language
        if self._supports_verbose:
            request["response_format"] = "verbose_json"
        if candidates and not language:
            # this is only a hint not a rule, it just nudges it towards the
            # two languages that are actually being spoken
            request["prompt"] = "Conversation in " + " and ".join(
                name_of(c) for c in candidates
            )

        try:
            response = await client.audio.transcriptions.create(**request)
        except Exception as exc:
            raise ProviderError(f"OpenAI transcription failed: {exc}") from exc

        text = (getattr(response, "text", "") or "").strip()
        detected = normalize_language_name(getattr(response, "language", None))
        if not detected:
            detected = language

        return Transcript(text=text, language=detected)


class OpenAITranslation(TranslationProvider):
    name = "openai"

    # I went through a lot of versions of this prompt. it kept ANSWERING
    # the sentence instead of translating it, especially questions, so
    # most of these rules are me arguing with it
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

    def __init__(self, model="gpt-4o-mini"):
        self.model = model

    def preflight(self, cfg):
        _client()

    async def translate(self, text, *, source, target, context=None):
        if not text.strip():
            return ""

        client = _client()
        messages = [
            {
                "role": "system",
                "content": self._SYSTEM.format(
                    source=name_of(source), target=name_of(target)
                ),
            }
        ]

        if context:
            # giving it the last few lines helps a lot with pronouns and
            # with languages that have gendered words
            messages.append(
                {
                    "role": "system",
                    "content": "Conversation so far, for context only:\n"
                    + "\n".join(context),
                }
            )

        messages.append({"role": "user", "content": text})

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,  # low, I want it consistent not creative
                max_tokens=400,
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI translation failed: {exc}") from exc

        translated = (response.choices[0].message.content or "").strip()
        if not translated:
            raise ProviderError("OpenAI translation returned nothing")

        # it sometimes wraps the answer in quotes even though I told it not to
        return translated.strip('"').strip()


class OpenAITTS(TTSProvider):
    name = "openai"

    _INSTRUCTIONS = (
        "Speak naturally and conversationally, at a brisk but clear pace, as "
        "though relaying what someone just said to you."
    )

    def __init__(self, model="gpt-4o-mini-tts"):
        self.model = model

    def preflight(self, cfg):
        _client()

    async def synthesize(self, text, *, language, voice):
        client = _client()
        return SpeechAudio(
            chunks=self._stream(client, text, voice),
            sample_rate=OPENAI_TTS_SAMPLE_RATE,
        )

    async def _stream(self, client, text, voice):
        request = {
            "model": self.model,
            "voice": voice,
            "input": text,
            "response_format": "pcm",
        }
        # only the gpt-4o ones take the instructions parameter, the older
        # tts-1 errors out if you send it
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

                    # each sample is 2 bytes and the network can split a
                    # chunk right down the middle of one. so hold onto the
                    # odd byte and stick it on the front of the next chunk
                    data = remainder + chunk
                    usable = len(data) - (len(data) % 2)
                    remainder = data[usable:]
                    if usable:
                        yield data[:usable]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(f"OpenAI speech synthesis failed: {exc}") from exc
