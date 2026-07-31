# the interfaces every provider has to follow
#
# there are 3 stages (speech to text, translate, text to speech) and each
# one is swappable, so you can do cloud speech recognition with local
# voices or whatever by changing 3 strings in config.yaml.
#
# all of them are async because they're mostly waiting on the network. the
# local ones are actually waiting on the cpu/gpu instead, so those wrap
# their slow call in asyncio.to_thread, otherwise transcribing one person
# freezes the other person's microphone

import io
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..config import PIPELINE_SAMPLE_RATE


class ProviderError(RuntimeError):
    # a provider broke in a way we can report and keep going
    pass


@dataclass
class Transcript:
    text: str
    language: str = None
    # only the local whisper gives me this. the openai api doesn't return
    # it at all, so over there routing has to lean on the alternation guess
    language_confidence: float = None

    @property
    def is_empty(self):
        return not self.text.strip()


@dataclass
class SpeechAudio:
    # streamed audio: raw mono int16 chunks plus what rate they're at
    chunks: object
    sample_rate: int


class Provider(ABC):
    name = "provider"

    def preflight(self, cfg):
        # check credentials and download models BEFORE the conversation
        # starts. this runs before any audio device is opened.
        #
        # doing the actual work here matters as much as the checking does.
        # a provider that downloads its model lazily would freeze in the
        # middle of the first thing somebody says, which is the worst
        # possible moment
        pass

    async def aclose(self):
        pass


class STTProvider(Provider):
    name = "stt"

    @abstractmethod
    async def transcribe(self, pcm, *, language=None, candidates=()):
        # pcm is mono int16 at 16khz.
        # language pins it when we already know (dual_mic).
        # candidates narrows the auto detect down to our two languages
        pass


class TranslationProvider(Provider):
    name = "translation"

    @abstractmethod
    async def translate(self, text, *, source, target, context=None):
        # context is the last few lines, for pronouns and gender
        pass


class TTSProvider(Provider):
    name = "tts"

    @abstractmethod
    async def synthesize(self, text, *, language, voice):
        # should give back the first chunk as fast as possible
        pass


def pcm_to_wav_bytes(pcm, sample_rate=PIPELINE_SAMPLE_RATE):
    # the speech apis want a wav file, so build one in memory instead of
    # writing it to disk
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)  # 2 bytes = 16 bit
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.astype(np.int16).tobytes())
    return buffer.getvalue()


def wav_bytes_to_pcm(raw):
    # read a wav into mono int16, gives back (audio, samplerate)
    with wave.open(io.BytesIO(raw), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ProviderError(
                f"expected 16-bit WAV, got {handle.getsampwidth() * 8}-bit"
            )
        channels = handle.getnchannels()
        rate = handle.getframerate()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)

    if channels > 1:
        # imported here, if its at the top the audio package imports this
        # file and this file imports it back
        from ..audio.resample import downmix_to_mono
        pcm = downmix_to_mono(pcm, channels)

    return pcm, rate


async def single_chunk(data):
    # wraps a normal non streaming response so it fits the streaming shape
    yield data
