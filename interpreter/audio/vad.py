# voice activity detection = figuring out when somebody is talking
#
# this is the part that replaces the push to talk button. it watches the
# audio and decides on its own where a sentence starts and stops.
#
# two options:
#   silero - a little 2mb neural net, way better when the room is noisy.
#            heads up: the "silero-vad" package on pypi installs torch and
#            torchaudio which is like 2.5 GB just to run a 2mb model, so I
#            load the onnx file directly instead
#   webrtc - google's old one. tiny, no download, but it thinks my keyboard
#            is a person talking
#
# the segmenter part is pure python, no audio devices and no network, so I
# can test all of it with fake audio

import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from ..config import PIPELINE_SAMPLE_RATE

_SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
_CACHE_DIR = Path(
    os.environ.get(
        "DUPLEX_INTERPRETER_CACHE", Path.home() / ".cache" / "duplex-interpreter"
    )
)
# silero v5 wants the last 64 samples of the previous window glued onto the
# front of the current one
_SILERO_CONTEXT_SAMPLES = 64

log = logging.getLogger(__name__)


class VadUnavailable(RuntimeError):
    # couldn't load this backend, caller should try the other one
    pass


@dataclass
class Utterance:
    # one chunk of speech, ready to send to the speech recognizer
    channel_id: str
    pcm: object          # mono int16 at 16khz
    seq: int
    captured_at: float = field(default_factory=time.monotonic)
    speech_seconds: float = 0.0

    @property
    def duration_seconds(self):
        return self.pcm.size / PIPELINE_SAMPLE_RATE


class VadBackend(ABC):
    # base class, tells you speech or not speech for one window

    window_samples = 0

    @property
    def window_ms(self):
        return 1000.0 * self.window_samples / PIPELINE_SAMPLE_RATE

    @abstractmethod
    def is_speech(self, window):
        pass

    def reset(self):
        # clear anything remembered between sentences.
        # webrtc and energy don't remember anything so they don't override
        pass


class SileroVad(VadBackend):
    window_samples = 512  # 32ms at 16khz, this is what it was trained on

    def __init__(self, threshold=0.5, model_path=None):
        try:
            import onnxruntime
        except ImportError as exc:
            raise VadUnavailable(
                "onnxruntime is not installed; run `pip install -r requirements.txt`"
            ) from exc

        self.threshold = threshold
        if model_path:
            path = Path(model_path)
        else:
            path = _ensure_silero_model()

        options = onnxruntime.SessionOptions()
        # one thread is plenty for a 2mb model, and it stops it fighting
        # the audio thread for cpu
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 3

        try:
            self._session = onnxruntime.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise VadUnavailable(
                f"could not load Silero VAD from {path}: {exc}"
            ) from exc

        self._input_names = {i.name for i in self._session.get_inputs()}
        # v5 has one combined "state" input, v4 had seperate h and c
        self._uses_fused_state = "state" in self._input_names
        self.reset()

    def reset(self):
        if self._uses_fused_state:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            # without this context thing it returns basically zero for
            # everything and never detects any speech at all. took me
            # forever to figure out, the model just silently does nothing
            self._context = np.zeros(_SILERO_CONTEXT_SAMPLES, dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def is_speech(self, window):
        samples = window.astype(np.float32) / 32768.0

        if self._uses_fused_state:
            audio = np.concatenate((self._context, samples)).reshape(1, -1)
            self._context = samples[-_SILERO_CONTEXT_SAMPLES:].copy()
        else:
            audio = samples.reshape(1, -1)

        inputs = {
            "input": audio,
            "sr": np.array(PIPELINE_SAMPLE_RATE, dtype=np.int64),
        }
        if self._uses_fused_state:
            inputs["state"] = self._state
        else:
            inputs["h"] = self._h
            inputs["c"] = self._c

        outputs = self._session.run(None, inputs)
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])

        # save the state for next time
        if self._uses_fused_state:
            self._state = outputs[1]
        else:
            self._h, self._c = outputs[1], outputs[2]

        return probability >= self.threshold


class WebrtcVad(VadBackend):
    # google's one. no model file, no state, instant
    window_samples = 320  # 20ms

    def __init__(self, aggressiveness=2):
        try:
            import webrtcvad
        except ImportError as exc:
            raise VadUnavailable(
                "webrtcvad is not installed; run `pip install -r requirements.txt`"
            ) from exc
        self._vad = webrtcvad.Vad(int(aggressiveness))

    def is_speech(self, window):
        return self._vad.is_speech(window.tobytes(), PIPELINE_SAMPLE_RATE)


class EnergyVad(VadBackend):
    # just checks if the audio is loud. its dumb but its predictable which
    # is exactly what I want in the tests
    window_samples = 320

    def __init__(self, threshold=0.02):
        self.threshold = threshold

    def is_speech(self, window):
        if window.size == 0:
            return False
        level = float(
            np.sqrt(np.mean(np.square(window.astype(np.float64) / 32768.0)))
        )
        return level >= self.threshold


def _ensure_silero_model():
    # find the onnx file, or download it (its only about 2mb)

    override = os.environ.get("SILERO_VAD_ONNX")
    if override:
        path = Path(override)
        if not path.exists():
            raise VadUnavailable(
                f"SILERO_VAD_ONNX points at a missing file: {path}"
            )
        return path

    cached = _CACHE_DIR / "silero_vad.onnx"
    # the size check is so a half finished download doesn't count
    if cached.exists() and cached.stat().st_size > 100_000:
        return cached

    # if they happen to have the real package installed, use its copy
    try:
        import silero_vad

        bundled = Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx"
        if bundled.exists():
            return bundled
    except ImportError:
        pass

    log.info("downloading Silero VAD model (~2.2 MB) to %s", cached)
    try:
        import httpx

        cached.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream(
            "GET", _SILERO_URL, timeout=30.0, follow_redirects=True
        ) as response:
            response.raise_for_status()
            # download to a .part file first then rename, so a failed
            # download doesn't leave a broken file that looks finished
            temporary = cached.with_suffix(".onnx.part")
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
            temporary.replace(cached)
    except Exception as exc:
        raise VadUnavailable(
            f"could not download the Silero VAD model ({exc}). "
            "Set vad.backend: webrtc in config.yaml to run without it."
        ) from exc

    return cached


def build_backend(cfg):
    # make the one they asked for, but fall back instead of crashing
    if cfg.backend == "webrtc":
        return WebrtcVad(cfg.aggressiveness)
    try:
        return SileroVad(cfg.threshold)
    except VadUnavailable as exc:
        log.warning("Silero VAD unavailable (%s); falling back to webrtc", exc)
        return WebrtcVad(cfg.aggressiveness)


class State(str, Enum):
    SILENT = "silent"
    SPEAKING = "speaking"


class SpeechSegmenter:
    # turns a stream of frames into seperate sentences
    #
    # how the state machine works:
    #  - I always keep the last preroll_ms of audio in a ring buffer, so
    #    when speech starts I can include the bit BEFORE it triggered.
    #    without this every sentence starts halfway through the first word
    #  - need start_frames voiced windows in a row to open a sentence,
    #    which throws out door slams and mouth clicks
    #  - silence_ms_to_end of quiet closes it
    #  - max_utterance_ms forces it closed so somebody rambling for two
    #    minutes still gets translated in pieces
    #  - if there wasn't at least min_utterance_ms of real speech, drop it

    def __init__(self, cfg, *, backend=None, channel_id="A"):
        self.cfg = cfg
        self.channel_id = channel_id
        self.backend = backend or build_backend(cfg)

        # convert all the millisecond settings into a number of windows.
        # max(1, ...) so nothing can end up as zero and break the loops
        window_ms = self.backend.window_ms
        self._window = self.backend.window_samples
        self._preroll_windows = max(1, int(round(cfg.preroll_ms / window_ms)))
        self._silence_windows_to_end = max(
            1, int(round(cfg.silence_ms_to_end / window_ms))
        )
        self._max_windows = max(1, int(round(cfg.max_utterance_ms / window_ms)))
        self._min_speech_windows = max(
            1, int(round(cfg.min_utterance_ms / window_ms))
        )
        # whisper likes a little silence on the end, but not the whole
        # timeout worth of it
        self._keep_trailing_windows = max(1, int(round(200 / window_ms)))

        self._state = State.SILENT
        self._pending = np.zeros(0, dtype=np.int16)
        self._preroll = deque(maxlen=self._preroll_windows)
        self._current = []
        self._voiced_run = 0
        self._silence_run = 0
        self._speech_windows = 0
        self._seq = 0

    @property
    def state(self):
        return self._state

    @property
    def in_speech(self):
        return self._state is State.SPEAKING

    def push(self, frame):
        # you can feed this any length, it buffers and cuts it into windows.
        # returns a list of any sentences that just finished
        if self._pending.size == 0:
            self._pending = frame.astype(np.int16)
        else:
            self._pending = np.concatenate(
                (self._pending, frame.astype(np.int16))
            )

        finished = []
        while self._pending.size >= self._window:
            window = self._pending[: self._window]
            self._pending = self._pending[self._window :]
            utterance = self._advance(window)
            if utterance is not None:
                finished.append(utterance)
        return finished

    def _advance(self, window):
        voiced = self.backend.is_speech(window)

        if self._state is State.SILENT:
            self._preroll.append(window)

            if voiced:
                self._voiced_run += 1
            else:
                self._voiced_run = 0  # streak broken, start over

            if self._voiced_run >= self.cfg.start_frames:
                # the preroll already has these voiced windows in it so I
                # don't need to add them again
                self._current = list(self._preroll)
                self._preroll.clear()
                self._speech_windows = self._voiced_run
                self._silence_run = 0
                self._voiced_run = 0
                self._state = State.SPEAKING
            return None

        # we're in the middle of a sentence
        self._current.append(window)

        if voiced:
            self._speech_windows += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self._silence_windows_to_end:
                return self._close(trim_trailing=True)

        if len(self._current) >= self._max_windows:
            return self._close(trim_trailing=False)
        return None

    def _close(self, *, trim_trailing):
        windows = self._current
        speech_windows = self._speech_windows
        self._reset_segment()

        # cut off most of the silence at the end, but leave a little
        if trim_trailing and self._silence_windows_to_end > self._keep_trailing_windows:
            drop = self._silence_windows_to_end - self._keep_trailing_windows
            if len(windows) > drop:
                windows = windows[:-drop]

        if speech_windows < self._min_speech_windows or not windows:
            log.debug(
                "channel %s dropped a %d-window blip", self.channel_id, len(windows)
            )
            return None

        self._seq += 1
        return Utterance(
            channel_id=self.channel_id,
            pcm=np.concatenate(windows),
            seq=self._seq,
            speech_seconds=speech_windows * self.backend.window_ms / 1000.0,
        )

    def _reset_segment(self):
        self._state = State.SILENT
        self._current = []
        self._preroll.clear()
        self._voiced_run = 0
        self._silence_run = 0
        self._speech_windows = 0

    def flush(self):
        # close whatever is open. used at shutdown and at the end of a file
        if self._state is not State.SPEAKING:
            return None
        return self._close(trim_trailing=False)

    def reset(self):
        # throw everything away. called when the echo gate closes
        self._reset_segment()
        self._pending = np.zeros(0, dtype=np.int16)
        self.backend.reset()
