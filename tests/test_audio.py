# tests for resampling, wav files, and the speaker ring buffer

import numpy as np
import pytest

from conftest import silence, tone
from interpreter.audio.playback import RecordingSpeaker
from interpreter.audio.resample import (
    StreamResampler,
    downmix_to_mono,
    resample_int16,
    rms,
)
from interpreter.config import PIPELINE_SAMPLE_RATE
from interpreter.providers.base import pcm_to_wav_bytes, wav_bytes_to_pcm


# ---- resampling ----


@pytest.mark.parametrize(("src", "dst"), [(16_000, 24_000), (24_000, 16_000), (48_000, 16_000), (22_050, 24_000)])
def test_resampling_preserves_duration(src, dst):
    pcm = np.tile(tone(1.0), 4)[:src]  # exactly one second at the source rate
    assert pcm.size == src
    out = resample_int16(pcm, src, dst)

    assert out.dtype == np.int16
    assert abs(out.size - dst) <= dst * 0.02


def test_matching_rates_are_a_no_op():
    pcm = tone(0.1)
    assert resample_int16(pcm, 16_000, 16_000) is pcm


def test_empty_input_is_handled():
    assert resample_int16(np.zeros(0, dtype=np.int16), 16_000, 24_000).size == 0


def test_resampling_keeps_the_signal_not_just_the_length():
    # up and back down again should look basically like it started
    original = tone(0.5, freq=440)
    round_tripped = resample_int16(resample_int16(original, 16_000, 48_000), 48_000, 16_000)

    length = min(original.size, round_tripped.size)
    correlation = np.corrcoef(
        original[:length].astype(float), round_tripped[:length].astype(float)
    )[0, 1]
    assert correlation > 0.95


def test_stream_resampler_matches_a_single_pass():
    # this is the clicking bug. chunk by chunk must match doing it in one go
    pcm = tone(0.5)
    streaming = StreamResampler(16_000, 24_000)

    pieces = [streaming.process(pcm[i : i + 1024]) for i in range(0, pcm.size, 1024)]
    pieces.append(streaming.process(np.zeros(0, dtype=np.int16), last=True))
    streamed = np.concatenate([p for p in pieces if p.size])
    one_shot = resample_int16(pcm, 16_000, 24_000)

    assert abs(streamed.size - one_shot.size) <= 64
    length = min(streamed.size, one_shot.size)
    assert np.corrcoef(streamed[:length].astype(float), one_shot[:length].astype(float))[0, 1] > 0.99


def test_stream_resampler_passthrough():
    streaming = StreamResampler(16_000, 16_000)
    assert streaming.passthrough

    pcm = tone(0.1)
    assert np.array_equal(streaming.process(pcm), pcm)


def test_downmix_averages_channels():
    left = np.array([100, 300, 500], dtype=np.int16)
    right = np.array([200, 100, 100], dtype=np.int16)
    interleaved = np.empty(6, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right

    assert np.array_equal(downmix_to_mono(interleaved, 2), np.array([150, 200, 300]))


def test_downmix_is_a_no_op_for_mono():
    pcm = tone(0.1)
    assert downmix_to_mono(pcm, 1) is pcm


def test_downmix_tolerates_a_truncated_final_frame():
    assert downmix_to_mono(np.array([10, 20, 30], dtype=np.int16), 2).size == 1


def test_rms_tracks_level():
    assert rms(silence(0.1)) == 0.0
    assert rms(tone(0.1, amplitude=0.5)) > rms(tone(0.1, amplitude=0.1))
    assert 0.0 < rms(tone(0.1, amplitude=0.5)) < 1.0
    assert rms(np.zeros(0, dtype=np.int16)) == 0.0


# ---- wav framing ----


def test_wav_round_trip():
    original = tone(0.4)
    pcm, rate = wav_bytes_to_pcm(pcm_to_wav_bytes(original))

    assert rate == PIPELINE_SAMPLE_RATE
    assert np.array_equal(pcm, original)


def test_a_stereo_wav_is_downmixed_on_read():
    mono = tone(0.2)
    stereo = np.repeat(mono, 2)

    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(stereo.tobytes())

    pcm, rate = wav_bytes_to_pcm(buffer.getvalue())
    assert rate == 44_100
    assert pcm.size == mono.size


# ---- playback ----


async def _chunks(*payloads: bytes):
    for payload in payloads:
        yield payload


async def test_recording_speaker_collects_audio():
    speaker = RecordingSpeaker()
    payload = tone(0.1).tobytes()

    completed = await speaker.play(
        _chunks(payload, payload), source_rate=PIPELINE_SAMPLE_RATE
    )

    assert completed
    assert speaker.pcm().size > 0


async def test_first_audio_callback_fires_once_on_the_first_chunk():
    # this callback is what starts the timer and shuts the echo gate
    speaker = RecordingSpeaker()
    fired = []
    payload = tone(0.05).tobytes()

    await speaker.play(
        _chunks(payload, payload, payload),
        source_rate=PIPELINE_SAMPLE_RATE,
        on_first_audio=lambda: fired.append(True),
    )

    assert fired == [True]


async def test_stop_cancels_an_in_flight_stream():
    # interrupting only works if this returns False instead of carrying on
    speaker = RecordingSpeaker()
    payload = tone(0.05).tobytes()

    async def interrupted():
        yield payload
        speaker.stop()
        yield payload

    assert await speaker.play(interrupted(), source_rate=PIPELINE_SAMPLE_RATE) is False


async def test_empty_chunks_are_skipped():
    speaker = RecordingSpeaker()
    completed = await speaker.play(
        _chunks(b"", tone(0.05).tobytes(), b""), source_rate=PIPELINE_SAMPLE_RATE
    )

    assert completed
    assert speaker.pcm().size > 0


async def test_playback_resamples_to_the_device_rate():
    speaker = RecordingSpeaker(sample_rate=48_000)
    payload = tone(1.0).tobytes()  # 1 second at 16 kHz

    await speaker.play(_chunks(payload), source_rate=PIPELINE_SAMPLE_RATE)

    assert abs(speaker.pcm().size - 48_000) < 48_000 * 0.05


# ---- array microphone ----


async def test_array_microphone_yields_fixed_size_frames():
    from interpreter.audio.capture import ArrayMicrophone
    from interpreter.config import FRAME_SAMPLES

    microphone = ArrayMicrophone(tone(0.5), trailing_silence_ms=0)
    frames = [frame async for frame in microphone.frames()]

    assert frames
    assert all(frame.size == FRAME_SAMPLES for frame in frames)
    assert not microphone.acoustic  # a file cannot be heard by a microphone


async def test_array_microphone_appends_trailing_silence():
    # without the padding the vad never notices the last sentence ended
    from interpreter.audio.capture import ArrayMicrophone

    microphone = ArrayMicrophone(tone(0.5), trailing_silence_ms=1000)
    total = 0
    async for frame in microphone.frames():
        total += frame.size

    assert total >= PIPELINE_SAMPLE_RATE * 1.4
