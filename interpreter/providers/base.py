"""Provider interfaces: STT, translation, TTS.

Every stage is swappable, so you can run cloud STT with local TTS, or DeepL
translation with OpenAI voices, by changing three strings in config.yaml.

All three are async because they are I/O bound. Local providers that are
actually CPU/GPU bound wrap their blocking call in `asyncio.to_thread` so the
event loop keeps serving the other speaker's audio.
"""

from __future__ import annotations

import io
import wave
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np

from ..config import PIPELINE_SAMPLE_RATE


class ProviderError(RuntimeError):
    """A provider failed in a way the pipeline should report, not crash on."""


@dataclass
class Transcript:
    text: str
    language: str | None = None
    # Only the local Whisper path can report this; the OpenAI API does not
    # expose it, so routing leans harder on the alternation heuristic there.
    language_confidence: float | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class SpeechAudio:
    """Streamed synthesized audio: raw mono int16 chunks plus their rate."""

    chunks: AsyncIterator[bytes]
    sample_rate: int


class Provider(ABC):
    """Shared lifecycle for all three stages."""

    name: str = "provider"

    def preflight(self, cfg) -> None:
        """Validate credentials and fetch models before the conversation starts.

        Runs before any audio device is opened, and is given the `AppConfig`
        so a provider can prepare exactly the language pair in play. Raise
        `ConfigError` for anything the user has to fix.

        Doing the work here matters as much as the checking: a provider that
        downloads its model lazily would otherwise stall in the middle of the
        first sentence somebody says.
        """

    async def aclose(self) -> None:
        return None


class STTProvider(Provider):
    name: str = "stt"

    @abstractmethod
    async def transcribe(
        self,
        pcm: np.ndarray,
        *,
        language: str | None = None,
        candidates: tuple[str, ...] = (),
    ) -> Transcript:
        """Transcribe mono int16 PCM at PIPELINE_SAMPLE_RATE.

        `language` pins the language when it is known (dual_mic mode).
        `candidates` narrows auto-detection to the two configured languages.
        """


class TranslationProvider(Provider):
    name: str = "translation"

    @abstractmethod
    async def translate(
        self,
        text: str,
        *,
        source: str,
        target: str,
        context: list[str] | None = None,
    ) -> str:
        """Translate `text`. `context` holds recent turns for pronoun/gender resolution."""


class TTSProvider(Provider):
    name: str = "tts"

    @abstractmethod
    async def synthesize(self, text: str, *, language: str, voice: str) -> SpeechAudio:
        """Stream synthesized audio. Should yield the first chunk ASAP."""


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int = PIPELINE_SAMPLE_RATE) -> bytes:
    """Wrap raw PCM in a WAV container in memory (what STT HTTP APIs want)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.astype(np.int16).tobytes())
    return buffer.getvalue()


def wav_bytes_to_pcm(raw: bytes) -> tuple[np.ndarray, int]:
    """Read a mono/stereo 16-bit WAV into mono int16 PCM plus its rate."""
    with wave.open(io.BytesIO(raw), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ProviderError(
                f"expected 16-bit WAV, got {handle.getsampwidth() * 8}-bit"
            )
        channels = handle.getnchannels()
        rate = handle.getframerate()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    if channels > 1:
        from ..audio.resample import downmix_to_mono

        pcm = downmix_to_mono(pcm, channels)
    return pcm, rate


async def single_chunk(data: bytes) -> AsyncIterator[bytes]:
    """Adapt a non-streaming TTS response to the streaming interface."""
    yield data
