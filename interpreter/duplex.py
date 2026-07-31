# stops the app from hearing itself and translating its own voice
#
# the problem: if the mic and the speaker are in the same room, the spanish
# coming out of the speaker goes right back into the mic, so it translates
# that back to english, then to spanish again... it never stops and it gets
# more wrong every time. took me a while to figure out why my first version
# went crazy after one sentence.
#
# I ended up doing 3 things:
#   1. while our speaker is playing, just throw away the mic audio
#   2. unless somebody is talking LOUD over it, then they probably want to
#      interrupt, so stop playing and listen
#   3. if we transcribe something that looks like what we just said, drop it.
#      this catches whatever sneaks past #1
#
# none of this runs in dual_mic mode because headphones already fix it

import logging
import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher

from .audio.resample import rms
from .config import PIPELINE_SAMPLE_RATE

log = logging.getLogger(__name__)

# how long one frame is in ms. 320 samples at 16000 hz = 20ms
_FRAME_MS = 1000.0 * 320 / PIPELINE_SAMPLE_RATE


def normalize_for_compare(text):
    # lowercase and rip out punctuation so "Hola!" and "hola" match
    kept = [c.lower() if c.isalnum() or c.isspace() else " " for c in text]
    return " ".join("".join(kept).split())


@dataclass
class _SpokenLine:
    text: str
    at: float


class DuplexGuard:
    # one of these per microphone

    def __init__(self, cfg, *, channel_id="A", gate_enabled=True,
                 echo_guard_enabled=True):
        self.cfg = cfg
        self.channel_id = channel_id
        # these two are seperate on purpose. the gate only makes sense if a
        # real mic can hear a real speaker, but the text check is basically
        # free so I leave it on whenever one device is doing both jobs
        self.gate_enabled = gate_enabled and cfg.shared_audio
        self.echo_guard_enabled = echo_guard_enabled and cfg.shared_audio

        self._speaking_until = 0.0
        self._speaking = False
        self._barged_in = False
        self._loud_ms = 0.0
        self._spoken = deque(maxlen=12)
        self._bargein_callback = None

        # counters, I print these at the end so I can tell if its working
        self.suppressed_frames = 0
        self.echo_drops = 0
        self.bargeins = 0

    def set_bargein_callback(self, callback):
        # gets called when somebody talks over the translation.
        # whatever you pass in should stop the playback
        self._bargein_callback = callback

    # ---- playback state, the pipeline calls these ----

    def playback_started(self):
        self._speaking = True
        self._barged_in = False
        self._loud_ms = 0.0

    def playback_finished(self):
        self._speaking = False

        if self._barged_in:
            # somebody interrupted us and they are talking RIGHT NOW.
            # the hangover below is for our own echo bouncing around the
            # room, if I apply it here I throw away the start of their
            # sentence, which is the whole thing they interrupted to say.
            # this was a really annoying bug to track down
            self._barged_in = False
            self._speaking_until = 0.0
            return

        self._speaking_until = time.monotonic() + self.cfg.hangover_ms / 1000.0

    def note_spoken(self, text):
        # remember what we said so we can recognize it if it comes back
        normalized = normalize_for_compare(text)
        if normalized:
            self._spoken.append(_SpokenLine(normalized, time.monotonic()))

    # ---- the gate ----

    @property
    def gated(self):
        if not self.gate_enabled:
            return False
        return self._speaking or time.monotonic() < self._speaking_until

    def observe(self, frame):
        # returns True if we should throw this frame away.
        # also does the barge in detection
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
                # went quiet again, so it was just a noise not a person
                self._loud_ms = 0.0

        self.suppressed_frames += 1
        return True

    def _trigger_bargein(self):
        self.bargeins += 1
        self._loud_ms = 0.0
        self._speaking = False
        self._speaking_until = 0.0
        self._barged_in = True
        log.info("channel %s: barge-in, stopping playback", self.channel_id)

        if self._bargein_callback is not None:
            try:
                self._bargein_callback()
            except Exception:
                # if stopping the speaker blows up thats bad but its not
                # worth killing the whole conversation over
                log.exception("barge-in callback failed")

    # ---- the text check ----

    def is_self_echo(self, text):
        if not self.echo_guard_enabled:
            return False

        candidate = normalize_for_compare(text)
        if not candidate:
            return False

        # forget anything we said more than a few seconds ago
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
    def stats(self):
        return {
            "suppressed_frames": self.suppressed_frames,
            "echo_drops": self.echo_drops,
            "bargeins": self.bargeins,
        }


def _similar(a, b):
    # 0 to 1, higher means more alike
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    # the echo usually comes back chopped up, because the gate catches the
    # first half and only the end leaks through. so if one is inside the
    # other thats good enough for me. the 8 is so short words like "si"
    # dont match everything
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 8 and shorter in longer:
        return 1.0

    return SequenceMatcher(None, a, b).ratio()
