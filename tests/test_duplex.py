# tests for the three echo suppression layers

import time

import numpy as np

from conftest import silence, tone
from interpreter.config import DuplexConfig
from interpreter.duplex import DuplexGuard, normalize_for_compare


def guard(**overrides):
    cfg = DuplexConfig(**overrides)
    return DuplexGuard(cfg, channel_id="A")


def frames(pcm, size=320):
    for offset in range(0, pcm.size - size + 1, size):
        yield pcm[offset : offset + size]


# ---- layer 1: the output gate ----


def test_audio_passes_when_nothing_is_playing():
    g = guard()
    assert not g.gated
    assert all(not g.observe(f) for f in frames(tone(0.2)))


def test_audio_is_suppressed_while_our_own_speaker_is_playing():
    g = guard(bargein=False)
    g.playback_started()

    assert g.gated
    assert all(g.observe(f) for f in frames(tone(0.2)))
    assert g.suppressed_frames > 0


def test_the_gate_stays_shut_for_the_hangover_after_playback():
    # the room echo keeps going after the last sample
    g = guard(hangover_ms=200, bargein=False)
    g.playback_started()
    g.playback_finished()

    assert g.gated  # tail still decaying
    time.sleep(0.25)
    assert not g.gated


def test_the_gate_is_off_entirely_for_headsets():
    g = DuplexGuard(DuplexConfig(shared_audio=False), channel_id="A")
    g.playback_started()

    assert not g.gated
    assert not g.observe(tone(0.02))


# ---- layer 2: barge-in ----


def test_sustained_loud_speech_interrupts_playback():
    stopped = []
    g = guard(bargein=True, bargein_rms=0.05, bargein_ms=100)
    g.set_bargein_callback(lambda: stopped.append(True))
    g.playback_started()

    for frame in frames(tone(0.5, amplitude=0.5)):
        g.observe(frame)

    assert stopped == [True]
    assert g.bargeins == 1
    assert not g.gated  # reopened, so the interruption gets captured


def test_the_hangover_is_skipped_after_a_barge_in():
    # somebody interrupted and is talking RIGHT NOW, so don't shut the
    # gate on them. playback still "finishes" after a barge in because the
    # pipeline stops the stream and cleans up either way, and if I re-arm
    # the hangover there I lose the start of what they interrupted to say
    g = guard(bargein=True, bargein_rms=0.05, bargein_ms=100, hangover_ms=200)
    g.set_bargein_callback(lambda: None)
    g.playback_started()

    for frame in frames(tone(0.5, amplitude=0.5)):
        g.observe(frame)
    assert g.bargeins == 1

    g.playback_finished()

    assert not g.gated
    assert all(not g.observe(f) for f in frames(tone(0.2, amplitude=0.5)))


def test_the_hangover_still_applies_to_uninterrupted_playback():
    # only a barge in skips it, the room echo is real the rest of the time
    g = guard(bargein=True, bargein_rms=0.05, bargein_ms=100, hangover_ms=200)
    g.playback_started()
    g.playback_finished()

    assert g.gated


def test_a_barge_in_does_not_suppress_the_following_playback():
    # it should only skip once, not turn the gate off forever
    g = guard(bargein=True, bargein_rms=0.05, bargein_ms=100, hangover_ms=200)
    g.set_bargein_callback(lambda: None)

    g.playback_started()
    for frame in frames(tone(0.5, amplitude=0.5)):
        g.observe(frame)
    g.playback_finished()
    assert not g.gated

    g.playback_started()  # next translation
    assert g.gated
    g.playback_finished()
    assert g.gated  # this one was not interrupted, so the tail is guarded again


def test_a_brief_noise_does_not_interrupt():
    stopped = []
    g = guard(bargein=True, bargein_rms=0.05, bargein_ms=300)
    g.set_bargein_callback(lambda: stopped.append(True))
    g.playback_started()

    for frame in frames(np.concatenate([tone(0.06, amplitude=0.5), silence(0.3)])):
        g.observe(frame)

    assert stopped == []
    assert g.gated


def test_quiet_leakage_never_interrupts():
    # our own voice leaking into the mic is not somebody interrupting
    stopped = []
    g = guard(bargein=True, bargein_rms=0.15, bargein_ms=100)
    g.set_bargein_callback(lambda: stopped.append(True))
    g.playback_started()

    for frame in frames(tone(0.6, amplitude=0.04)):
        g.observe(frame)

    assert stopped == []


def test_a_failing_bargein_callback_does_not_propagate():
    def explode():
        raise RuntimeError("speaker is gone")

    g = guard(bargein=True, bargein_rms=0.05, bargein_ms=40)
    g.set_bargein_callback(explode)
    g.playback_started()

    for frame in frames(tone(0.3, amplitude=0.5)):
        g.observe(frame)  # must not raise

    assert g.bargeins >= 1


# ---- layer 3: the self-echo text guard ----


def test_exact_repeats_of_our_own_speech_are_dropped():
    g = guard()
    g.note_spoken("hola como estas")

    assert g.is_self_echo("hola como estas")
    assert g.echo_drops == 1


def test_punctuation_and_case_differences_still_match():
    g = guard()
    g.note_spoken("Hola, ¿cómo estás?")

    assert g.is_self_echo("hola como estas")


def test_a_truncated_echo_still_matches():
    # the gate normally catches the start so only the end leaks out
    g = guard()
    g.note_spoken("donde esta la estacion de tren mas cercana")

    assert g.is_self_echo("la estacion de tren mas cercana")


def test_genuinely_different_speech_passes():
    g = guard()
    g.note_spoken("hola como estas")

    assert not g.is_self_echo("where is the train station")
    assert not g.is_self_echo("necesito un taxi al aeropuerto")


def test_echoes_expire_after_the_window():
    g = guard(echo_guard_window_s=0.2)
    g.note_spoken("hola como estas")
    time.sleep(0.3)

    # said again much later, thats a real repeat not an echo
    assert not g.is_self_echo("hola como estas")


def test_the_echo_guard_is_off_for_headsets():
    g = DuplexGuard(DuplexConfig(shared_audio=False), channel_id="A")
    g.note_spoken("hola como estas")

    assert not g.is_self_echo("hola como estas")


def test_empty_text_is_never_an_echo():
    g = guard()
    g.note_spoken("hola")

    assert not g.is_self_echo("")
    assert not g.is_self_echo("   ")


def test_normalization_collapses_noise():
    assert normalize_for_compare("  Hello,   WORLD!! ") == "hello world"
    assert normalize_for_compare("¿Cómo estás?") == "cómo estás"


def test_stats_are_reported():
    g = guard(bargein=False)
    g.playback_started()
    for frame in frames(tone(0.1)):
        g.observe(frame)
    g.note_spoken("hola")
    g.is_self_echo("hola")

    stats = g.stats
    assert stats["suppressed_frames"] > 0
    assert stats["echo_drops"] == 1
