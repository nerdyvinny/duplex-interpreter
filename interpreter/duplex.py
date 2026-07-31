"""Stopping the app from translating its own voice.

When one microphone and one speaker share a room, the translation coming out
of the speaker goes straight back into the microphone. Untreated, the app
translates itself forever, and each loop drifts further from what anyone said.

Three layers, none of which need an acoustic echo canceller:

1. **Output gate** — while this channel's speaker is playing, its VAD input is
   discarded, plus a short hangover for the room's reverb tail.
2. **Barge-in** — sustained loud input while gated means a human is talking
   over the translation, so playback stops and the gate reopens. Without this
   the gate would make the app rude to interrupt.
3. **Self-echo guard** — anything transcribed that closely matches something
   we recently said is dropped. This is the backstop that catches whatever
   leaks past the gate (loud speakers, long reverb, a slow device buffer).

All three are disabled in dual_mic mode, where headsets solve the problem
physically and gating would only add latency.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from .audio.resample import rms
from .config import DuplexConfig, PIPELINE_SAMPLE_RATE

log = logging.getLogger(__name__)

_FRAME_MS = 1000.0 * 320 / PIPELINE_SAMPLE_RATE  # nominal frame duration


def normalize_for_compare(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    kept = [c.lower() if c.isalnum() or c.isspace() else " " for c in text]
    return " ".join("".join(kept).split())


@dataclass
class _SpokenLine:
    text: str
    at: float


class DuplexGuard:
    """Per-channel echo suppression. One instance per microphone."""

    def __init__(
        self,
        cfg: DuplexConfig,
        *,
        channel_id: str = "A",
        gate_enabled: bool = True,
        echo_guard_enabled: bool = True,
    ) -> None:
        self.cfg = cfg
        self.channel_id = channel_id
        # These are independent. The acoustic gate only makes sense when a
        # real microphone can hear a real speaker; the text-level echo guard
        # is cheap and stays on whenever one device serves both people.
        self.gate_enabled = gate_enabled and cfg.shared_audio
        self.echo_guard_enabled = echo_guard_enabled and cfg.shared_audio

        self._speaking_until = 0.0
        self._speaking = False
        self._barged_in = False
        self._loud_ms = 0.0
        self._spoken: deque[_SpokenLine] = deque(maxlen=12)
        self._bargein_callback = None

        self.suppressed_frames = 0
        self.echo_drops = 0
        self.bargeins = 0

    def set_bargein_callback(self, callback) -> None:
        """Called when a human talks over the translation; should stop playback."""
        self._bargein_callback = callback

    # --- playback state, driven by the pipeline ---

    def playback_started(self) -> None:
        self._speaking = True
        self._barged_in = False
        self._loud_ms = 0.0

    def playback_finished(self) -> None:
        self._speaking = False
        if self._barged_in:
            # A barge-in already opened the gate, and the person who
            # interrupted is mid-sentence right now. The hangover exists to
            # swallow the room's reverb tail after *our* speech ends; applying
            # it here would throw away the first `hangover_ms` of theirs — and
            # with it the start of the very utterance they interrupted to say.
            self._barged_in = False
            self._speaking_until = 0.0
            return
        self._speaking_until = time.monotonic() + self.cfg.hangover_ms / 1000.0

    def note_spoken(self, text: str) -> None:
        """Remember something we just said, for the self-echo guard."""
        normalized = normalize_for_compare(text)
        if normalized:
            self._spoken.append(_SpokenLine(normalized, time.monotonic()))

    # --- the gate ---

    @property
    def gated(self) -> bool:
        if not self.gate_enabled:
            return False
        return self._speaking or time.monotonic() < self._speaking_until

    def observe(self, frame: np.ndarray) -> bool:
        """Inspect one captured frame.

        Returns True if the frame should be discarded rather than fed to the
        VAD. Also drives barge-in detection.
        """
        if not self.gated:
            self._loud_ms = 0.0
            return False

        if self.cfg.bargein and self._speaking:
            level = rms(frame)
            if level >= self.cfg.bargein_rms:
                self._loud_ms += _FRAME_MS
                if self._loud_ms >= self.cfg.bargein_ms:
                    self._trigger_bargein()
                    return False
            else:
                self._loud_ms = 0.0

        self.suppressed_frames += 1
        return True

    def _trigger_bargein(self) -> None:
        self.bargeins += 1
        self._loud_ms = 0.0
        self._speaking = False
        self._speaking_until = 0.0
        self._barged_in = True
        log.info("channel %s: barge-in, stopping playback", self.channel_id)
        if self._bargein_callback is not None:
            try:
                self._bargein_callback()
            except Exception:  # noqa: BLE001 - never let this kill the loop
                log.exception("barge-in callback failed")

    # --- the self-echo backstop ---

    def is_self_echo(self, text: str) -> bool:
        """Did we just say this ourselves?"""
        if not self.echo_guard_enabled:
            return False
        candidate = normalize_for_compare(text)
        if not candidate:
            return False

        now = time.monotonic()
        window = self.cfg.echo_guard_window_s
        while self._spoken and now - self._spoken[0].at > window:
            self._spoken.popleft()

        for line in self._spoken:
            if _similar(candidate, line.text) >= self.cfg.echo_guard_similarity:
                self.echo_drops += 1
                log.info(
                    "channel %s: dropped %r as self-echo of %r",
                    self.channel_id,
                    text,
                    line.text,
                )
                return True
        return False

    @property
    def stats(self) -> dict[str, int]:
        return {
            "suppressed_frames": self.suppressed_frames,
            "echo_drops": self.echo_drops,
            "bargeins": self.bargeins,
        }


def _similar(a: str, b: str) -> float:
    """Similarity in 0..1, tolerant of one side being a prefix of the other.

    Echo often arrives truncated — the gate catches the start of our own
    speech and only the tail leaks through — so a containment check runs
    alongside the full ratio.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 8 and shorter in longer:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()
