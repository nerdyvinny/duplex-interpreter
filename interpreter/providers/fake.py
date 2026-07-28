"""Deterministic fake providers for the offline test suite.

These let the whole orchestrator be exercised — routing, ordering, the echo
guard, error handling — with no network, no API key and no audio hardware.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from ..config import PIPELINE_SAMPLE_RATE
from .base import (
    ProviderError,
    SpeechAudio,
    STTProvider,
    Transcript,
    TranslationProvider,
    TTSProvider,
)


class FakeSTT(STTProvider):
    """Returns queued transcripts in order, one per utterance."""

    name = "fake"

    def __init__(
        self,
        transcripts: list[Transcript] | None = None,
        *,
        delay: float = 0.0,
        delays: list[float] | None = None,
    ) -> None:
        self.transcripts = list(transcripts or [])
        self.delay = delay
        self.delays = list(delays or [])
        self.calls: list[dict] = []

    async def transcribe(
        self,
        pcm: np.ndarray,
        *,
        language: str | None = None,
        candidates: tuple[str, ...] = (),
    ) -> Transcript:
        index = len(self.calls)
        self.calls.append(
            {"samples": int(pcm.size), "language": language, "candidates": candidates}
        )
        wait = self.delays[index] if index < len(self.delays) else self.delay
        if wait:
            await asyncio.sleep(wait)
        if index < len(self.transcripts):
            return self.transcripts[index]
        return Transcript(text="", language=language)


class FakeTranslation(TranslationProvider):
    """Marks up the text so tests can assert direction without a real model."""

    name = "fake"

    def __init__(self, *, delay: float = 0.0, fail_on: set[str] | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on or set()
        self.calls: list[dict] = []

    async def translate(
        self,
        text: str,
        *,
        source: str,
        target: str,
        context: list[str] | None = None,
    ) -> str:
        self.calls.append(
            {"text": text, "source": source, "target": target, "context": list(context or [])}
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        if text in self.fail_on:
            raise ProviderError(f"fake translation refused {text!r}")
        return f"[{source}->{target}] {text}"


class FakeTTS(TTSProvider):
    """Emits a fixed amount of silence and records what it was asked to say."""

    name = "fake"

    def __init__(self, *, delay: float = 0.0, samples: int = 1600) -> None:
        self.delay = delay
        self.samples = samples
        self.calls: list[dict] = []

    async def synthesize(self, text: str, *, language: str, voice: str) -> SpeechAudio:
        self.calls.append({"text": text, "language": language, "voice": voice})
        delay, samples = self.delay, self.samples

        async def _chunks() -> AsyncIterator[bytes]:
            if delay:
                await asyncio.sleep(delay)
            yield np.zeros(samples, dtype=np.int16).tobytes()

        return SpeechAudio(chunks=_chunks(), sample_rate=PIPELINE_SAMPLE_RATE)
