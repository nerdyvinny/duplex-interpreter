# fake providers for the tests
#
# these let me run the whole thing (routing, ordering, the echo guard,
# error handling) with no network, no api key and no sound card. they give
# back the same answer every time so the tests aren't flaky

import asyncio

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
    # hands back whatever transcripts you queued up, in order
    name = "fake"

    def __init__(self, transcripts=None, *, delay=0.0, delays=None):
        self.transcripts = list(transcripts or [])
        self.delay = delay
        self.delays = list(delays or [])  # per call, for testing ordering
        self.calls = []

    async def transcribe(self, pcm, *, language=None, candidates=()):
        index = len(self.calls)
        self.calls.append(
            {
                "samples": int(pcm.size),
                "language": language,
                "candidates": candidates,
            }
        )

        if index < len(self.delays):
            wait = self.delays[index]
        else:
            wait = self.delay
        if wait:
            await asyncio.sleep(wait)

        if index < len(self.transcripts):
            return self.transcripts[index]
        return Transcript(text="", language=language)  # ran out


class FakeTranslation(TranslationProvider):
    # tags the text with the direction so the tests can check which way
    # it was translated without needing a real model
    name = "fake"

    def __init__(self, *, delay=0.0, fail_on=None):
        self.delay = delay
        self.fail_on = fail_on or set()
        self.calls = []

    async def translate(self, text, *, source, target, context=None):
        self.calls.append(
            {
                "text": text,
                "source": source,
                "target": target,
                "context": list(context or []),
            }
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        if text in self.fail_on:
            raise ProviderError(f"fake translation refused {text!r}")
        return f"[{source}->{target}] {text}"


class FakeTTS(TTSProvider):
    # gives back silence and writes down what it was asked to say
    name = "fake"

    def __init__(self, *, delay=0.0, samples=1600):
        self.delay = delay
        self.samples = samples
        self.calls = []

    async def synthesize(self, text, *, language, voice):
        self.calls.append({"text": text, "language": language, "voice": voice})

        # grab these now, the generator below runs later and self might
        # have moved on by then
        delay, samples = self.delay, self.samples

        async def _chunks():
            if delay:
                await asyncio.sleep(delay)
            yield np.zeros(samples, dtype=np.int16).tobytes()

        return SpeechAudio(chunks=_chunks(), sample_rate=PIPELINE_SAMPLE_RATE)
