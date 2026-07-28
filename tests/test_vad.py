"""VAD segmentation over synthetic audio.

Uses the deterministic energy backend so these assert the *state machine*,
not the acoustic model.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import silence, tone
from interpreter.audio.vad import EnergyVad, SpeechSegmenter, State
from interpreter.config import PIPELINE_SAMPLE_RATE, VadConfig


def segmenter(**overrides) -> SpeechSegmenter:
    cfg = VadConfig(
        silence_ms_to_end=overrides.pop("silence_ms_to_end", 300),
        preroll_ms=overrides.pop("preroll_ms", 100),
        min_utterance_ms=overrides.pop("min_utterance_ms", 200),
        max_utterance_ms=overrides.pop("max_utterance_ms", 4000),
        start_frames=overrides.pop("start_frames", 3),
    )
    return SpeechSegmenter(cfg, backend=EnergyVad(threshold=0.02), **overrides)


def feed(seg: SpeechSegmenter, *parts: np.ndarray) -> list:
    utterances = []
    for part in parts:
        utterances.extend(seg.push(part))
    return utterances


def test_single_utterance_is_segmented():
    seg = segmenter()
    found = feed(seg, silence(0.3), tone(1.0), silence(0.6))

    assert len(found) == 1
    assert found[0].seq == 1
    assert found[0].channel_id == "A"
    # ~1s of speech plus pre-roll and the retained tail.
    assert 0.9 <= found[0].duration_seconds <= 1.6


def test_two_utterances_separated_by_a_pause():
    seg = segmenter()
    found = feed(
        seg,
        silence(0.2),
        tone(0.8),
        silence(0.7),  # longer than silence_ms_to_end -> closes the first
        tone(0.8),
        silence(0.7),
    )

    assert len(found) == 2
    assert [u.seq for u in found] == [1, 2]


def test_short_pause_does_not_split_an_utterance():
    """A speaker drawing breath mid-sentence must not produce two turns."""
    seg = segmenter(silence_ms_to_end=500)
    found = feed(seg, tone(0.6), silence(0.2), tone(0.6), silence(0.9))

    assert len(found) == 1
    assert found[0].duration_seconds >= 1.2


def test_blips_below_the_minimum_are_dropped():
    seg = segmenter(min_utterance_ms=400, start_frames=1)
    found = feed(seg, silence(0.2), tone(0.1), silence(0.6))

    assert found == []


def test_preroll_keeps_audio_from_before_the_trigger():
    """Without pre-roll every utterance would start mid-word."""
    with_preroll = segmenter(preroll_ms=300, start_frames=3)
    without = segmenter(preroll_ms=20, start_frames=3)

    long = feed(with_preroll, tone(0.8), silence(0.6))[0]
    short = feed(without, tone(0.8), silence(0.6))[0]

    assert long.pcm.size > short.pcm.size


def test_max_duration_force_flushes_a_monologue():
    seg = segmenter(max_utterance_ms=1000)
    found = feed(seg, tone(3.5))

    assert len(found) >= 3
    for utterance in found:
        assert utterance.duration_seconds <= 1.1


def test_flush_closes_an_open_utterance():
    seg = segmenter()
    assert feed(seg, tone(0.8)) == []
    assert seg.state is State.SPEAKING

    trailing = seg.flush()
    assert trailing is not None
    assert trailing.duration_seconds > 0.5
    assert seg.flush() is None  # nothing left


def test_reset_discards_the_in_progress_segment():
    """This is what the duplex gate calls when our own speaker starts."""
    seg = segmenter()
    feed(seg, tone(0.8))
    assert seg.in_speech

    seg.reset()
    assert not seg.in_speech
    assert seg.flush() is None


def test_pure_silence_produces_nothing():
    seg = segmenter()
    assert feed(seg, silence(3.0)) == []


def test_arbitrary_chunk_sizes_are_rebuffered():
    """Frames arrive at whatever size the device gives us."""
    seg = segmenter()
    audio = np.concatenate([silence(0.2), tone(1.0), silence(0.6)])

    found = []
    for offset in range(0, audio.size, 137):  # deliberately not a window multiple
        found.extend(seg.push(audio[offset : offset + 137]))

    assert len(found) == 1


def test_sequence_numbers_increment_per_channel():
    seg = segmenter(channel_id="B")
    found = feed(
        seg, tone(0.6), silence(0.6), tone(0.6), silence(0.6), tone(0.6), silence(0.6)
    )

    assert [u.seq for u in found] == [1, 2, 3]
    assert all(u.channel_id == "B" for u in found)


def test_silero_detects_real_speech_and_rejects_silence():
    """Regression guard for the v5 context window.

    Silero v5 conditions each 512-sample window on the 64 samples before it.
    Omitting that context makes the model return near-zero probability for
    everything, so the app silently never hears anyone. Skipped when the
    model can't be fetched.
    """
    from pathlib import Path

    from interpreter.audio.vad import SileroVad, VadUnavailable
    from interpreter.providers.base import wav_bytes_to_pcm

    sample = Path(__file__).resolve().parent.parent / "samples" / "hello_en.wav"
    if not sample.exists():
        pytest.skip("samples/hello_en.wav is not present")
    try:
        backend = SileroVad(threshold=0.5)
    except VadUnavailable as exc:
        pytest.skip(f"Silero model unavailable: {exc}")

    pcm, _ = wav_bytes_to_pcm(sample.read_bytes())
    window = backend.window_samples

    voiced = sum(
        backend.is_speech(pcm[i : i + window])
        for i in range(0, pcm.size - window, window)
    )
    assert voiced > 20, "Silero found essentially no speech in a spoken WAV"

    backend.reset()
    quiet = silence(2.0)
    voiced_in_silence = sum(
        backend.is_speech(quiet[i : i + window])
        for i in range(0, quiet.size - window, window)
    )
    assert voiced_in_silence == 0


def test_silero_and_webrtc_agree_on_the_sample():
    from pathlib import Path

    from interpreter.audio.vad import SileroVad, VadUnavailable, WebrtcVad
    from interpreter.providers.base import wav_bytes_to_pcm

    sample = Path(__file__).resolve().parent.parent / "samples" / "hello_en.wav"
    if not sample.exists():
        pytest.skip("samples/hello_en.wav is not present")
    try:
        silero = SileroVad(threshold=0.5)
    except VadUnavailable as exc:
        pytest.skip(f"Silero model unavailable: {exc}")

    pcm, _ = wav_bytes_to_pcm(sample.read_bytes())
    cfg = VadConfig(silence_ms_to_end=500, min_utterance_ms=250)

    counts = []
    for backend in (silero, WebrtcVad(2)):
        seg = SpeechSegmenter(cfg, backend=backend)
        found = feed(seg, pcm)
        trailing = seg.flush()
        counts.append(len(found) + (1 if trailing else 0))

    assert counts[0] == counts[1] > 0


@pytest.mark.parametrize("backend_name", ["webrtc"])
def test_real_backend_finds_speech_shaped_audio(backend_name):
    """Smoke test against the real WebRTC VAD, not the energy stand-in."""
    from interpreter.audio.vad import WebrtcVad

    cfg = VadConfig(
        backend=backend_name,
        silence_ms_to_end=300,
        min_utterance_ms=200,
        aggressiveness=0,
    )
    seg = SpeechSegmenter(cfg, backend=WebrtcVad(0))

    # A formant-ish buzz: WebRTC's VAD keys off spectral shape, not level.
    t = np.arange(int(PIPELINE_SAMPLE_RATE * 1.2)) / PIPELINE_SAMPLE_RATE
    voiced = sum(np.sin(2 * np.pi * f * t) for f in (140, 280, 700, 1400, 2100))
    speechlike = (voiced / 5 * 0.4 * 32767).astype(np.int16)

    found = feed(seg, silence(0.2), speechlike, silence(0.8))
    assert len(found) == 1
