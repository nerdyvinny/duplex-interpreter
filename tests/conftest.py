# shared test helpers. nothing in here touches the network or a sound card

import sys
from pathlib import Path

import numpy as np
import pytest

# so "import interpreter" works no matter where pytest is run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter.config import PIPELINE_SAMPLE_RATE, AppConfig, ChannelConfig, LanguageSlot  # noqa: E402
from interpreter.config import DuplexConfig, Mode, ProvidersConfig, VadConfig  # noqa: E402


def tone(seconds, *, freq=220, amplitude=0.35):
    # fake "speech", loud enough that the energy vad calls it voiced
    t = np.arange(int(PIPELINE_SAMPLE_RATE * seconds)) / PIPELINE_SAMPLE_RATE
    # two frequencies instead of one. a pure sine is too clean and some of
    # the real vads refuse to believe its a person
    wave = np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * freq * 2.5 * t)
    return (wave / 1.4 * amplitude * 32767).astype(np.int16)


def silence(seconds):
    return np.zeros(int(PIPELINE_SAMPLE_RATE * seconds), dtype=np.int16)


@pytest.fixture
def vad_config():
    return VadConfig(
        backend="webrtc",
        silence_ms_to_end=300,
        preroll_ms=100,
        min_utterance_ms=200,
        max_utterance_ms=4000,
        start_frames=3,
    )


@pytest.fixture
def single_mic_config():
    return AppConfig(
        mode=Mode.SINGLE_MIC,
        language_a=LanguageSlot("en", "English", "alloy"),
        language_b=LanguageSlot("es", "Spanish", "nova"),
        channels=[ChannelConfig(id="A", language="auto")],
        providers=ProvidersConfig(stt="fake", translation="fake", tts="fake"),
        vad=VadConfig(backend="webrtc"),
        duplex=DuplexConfig(),
    )


@pytest.fixture
def dual_mic_config():
    return AppConfig(
        mode=Mode.DUAL_MIC,
        language_a=LanguageSlot("en", "English", "alloy"),
        language_b=LanguageSlot("es", "Spanish", "nova"),
        # different device numbers on purpose, the validator rejects it if
        # both channels share one microphone
        channels=[
            ChannelConfig(id="A", input_device=1, output_device=1, language="en"),
            ChannelConfig(id="B", input_device=2, output_device=2, language="es"),
        ],
        providers=ProvidersConfig(stt="fake", translation="fake", tts="fake"),
        vad=VadConfig(backend="webrtc"),
        duplex=DuplexConfig(shared_audio=False),
    )
