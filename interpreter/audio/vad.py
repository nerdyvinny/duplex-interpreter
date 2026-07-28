"""Voice activity detection and utterance segmentation.

This is what replaces the push-to-talk button: it watches the stream and
decides, on its own, where one person's sentence starts and stops.

Two backends, one interface:
  * silero  - 2.2 MB ONNX model on onnxruntime. Far better in noise. Note the
              `silero-vad` PyPI package drags in torch + torchaudio (~2.5 GB)
              purely to run this; we load the ONNX directly instead.
  * webrtc  - `webrtcvad-wheels`. Tiny, instant, no download, more false
              triggers in a noisy room.

The segmenter itself is pure and synchronous: frames in, utterances out. No
audio devices, no network, no event loop — so it is fully unit-testable with
synthetic PCM.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from ..config import PIPELINE_SAMPLE_RATE, VadConfig

log = logging.getLogger(__name__)

_SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
_CACHE_DIR = Path(
    os.environ.get("DUPLEX_INTERPRETER_CACHE", Path.home() / ".cache" / "duplex-interpreter")
)
# Silero v5 conditions each 512-sample window on the 64 samples before it.
_SILERO_CONTEXT_SAMPLES = 64


class VadUnavailable(RuntimeError):
    """A backend could not be loaded; the caller should fall back."""


@dataclass
class Utterance:
    """One contiguous stretch of speech, ready for transcription."""

    channel_id: str
    pcm: np.ndarray  # mono int16 at PIPELINE_SAMPLE_RATE
    seq: int
    captured_at: float = field(default_factory=time.monotonic)
    speech_seconds: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.pcm.size / PIPELINE_SAMPLE_RATE


class VadBackend(ABC):
    """Frame-level speech/no-speech classifier."""

    window_samples: int

    @property
    def window_ms(self) -> float:
        return 1000.0 * self.window_samples / PIPELINE_SAMPLE_RATE

    @abstractmethod
    def is_speech(self, window: np.ndarray) -> bool:
        """Classify exactly `window_samples` of mono int16 audio."""

    def reset(self) -> None:
        """Clear any recurrent state between conversations."""


class SileroVad(VadBackend):
    """Silero VAD v4/v5 run directly on onnxruntime."""

    window_samples = 512  # 32 ms at 16 kHz — what the model was trained on

    def __init__(self, threshold: float = 0.5, model_path: str | Path | None = None) -> None:
        try:
            import onnxruntime
        except ImportError as exc:
            raise VadUnavailable(
                "onnxruntime is not installed; run `pip install -r requirements.txt`"
            ) from exc

        self.threshold = threshold
        path = Path(model_path) if model_path else _ensure_silero_model()

        options = onnxruntime.SessionOptions()
        # One thread is plenty for a 2 MB model and avoids fighting the
        # audio callback thread for cores.
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        try:
            self._session = onnxruntime.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001 - corrupt download, bad arch, ...
            raise VadUnavailable(f"could not load Silero VAD from {path}: {exc}") from exc

        self._input_names = {i.name for i in self._session.get_inputs()}
        # v5 carries one fused `state` tensor; v4 carried separate h/c.
        self._uses_fused_state = "state" in self._input_names
        self.reset()

    def reset(self) -> None:
        if self._uses_fused_state:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            # v5 expects the previous window's last 64 samples prepended to
            # each call. Without that context it returns near-zero
            # probabilities for everything and never detects speech at all.
            self._context = np.zeros(_SILERO_CONTEXT_SAMPLES, dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def is_speech(self, window: np.ndarray) -> bool:
        samples = window.astype(np.float32) / 32768.0
        if self._uses_fused_state:
            audio = np.concatenate((self._context, samples)).reshape(1, -1)
            self._context = samples[-_SILERO_CONTEXT_SAMPLES:].copy()
        else:
            audio = samples.reshape(1, -1)

        inputs = {"input": audio, "sr": np.array(PIPELINE_SAMPLE_RATE, dtype=np.int64)}
        if self._uses_fused_state:
            inputs["state"] = self._state
        else:
            inputs["h"] = self._h
            inputs["c"] = self._c

        outputs = self._session.run(None, inputs)
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        if self._uses_fused_state:
            self._state = outputs[1]
        else:
            self._h, self._c = outputs[1], outputs[2]
        return probability >= self.threshold


class WebrtcVad(VadBackend):
    """Google's WebRTC VAD. No download, no model, no recurrent state."""

    window_samples = 320  # 20 ms at 16 kHz

    def __init__(self, aggressiveness: int = 2) -> None:
        try:
            import webrtcvad
        except ImportError as exc:
            raise VadUnavailable(
                "webrtcvad is not installed; run `pip install -r requirements.txt`"
            ) from exc
        self._vad = webrtcvad.Vad(int(aggressiveness))

    def is_speech(self, window: np.ndarray) -> bool:
        return self._vad.is_speech(window.tobytes(), PIPELINE_SAMPLE_RATE)


class EnergyVad(VadBackend):
    """Plain RMS threshold. Deterministic, so the tests use it as a stand-in."""

    window_samples = 320

    def __init__(self, threshold: float = 0.02) -> None:
        self.threshold = threshold

    def is_speech(self, window: np.ndarray) -> bool:
        if window.size == 0:
            return False
        level = float(np.sqrt(np.mean(np.square(window.astype(np.float64) / 32768.0))))
        return level >= self.threshold


def _ensure_silero_model() -> Path:
    """Find the Silero ONNX locally, or fetch it once (~2.2 MB)."""
    override = os.environ.get("SILERO_VAD_ONNX")
    if override:
        path = Path(override)
        if not path.exists():
            raise VadUnavailable(f"SILERO_VAD_ONNX points at a missing file: {path}")
        return path

    cached = _CACHE_DIR / "silero_vad.onnx"
    if cached.exists() and cached.stat().st_size > 100_000:
        return cached

    # Reuse the package's bundled copy if the user happens to have installed it.
    try:
        import silero_vad  # type: ignore[import-not-found]

        bundled = Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx"
        if bundled.exists():
            return bundled
    except ImportError:
        pass

    log.info("downloading Silero VAD model (~2.2 MB) to %s", cached)
    try:
        import httpx

        cached.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", _SILERO_URL, timeout=30.0, follow_redirects=True) as response:
            response.raise_for_status()
            temporary = cached.with_suffix(".onnx.part")
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
            temporary.replace(cached)
    except Exception as exc:  # noqa: BLE001 - offline, proxy, DNS, ...
        raise VadUnavailable(
            f"could not download the Silero VAD model ({exc}). "
            "Set vad.backend: webrtc in config.yaml to run without it."
        ) from exc

    return cached


def build_backend(cfg: VadConfig) -> VadBackend:
    """Construct the configured backend, falling back rather than failing."""
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
    """Turns a frame stream into discrete utterances.

    The state machine, in order of what matters:

    * A **pre-roll ring** holds the last `preroll_ms` of audio at all times, so
      when speech is detected the utterance includes the moments *before* the
      trigger. Without this, every utterance starts mid-word.
    * `start_frames` consecutive voiced windows are needed to open a segment,
      which rejects door slams and mouth clicks.
    * `silence_ms_to_end` of quiet closes it. This is the dominant term in
      end-to-end latency and the main thing worth tuning.
    * `max_utterance_ms` force-flushes, so a monologue gets translated in
      pieces instead of after two minutes.
    * Segments with less than `min_utterance_ms` of actual speech are dropped.
    """

    def __init__(
        self,
        cfg: VadConfig,
        *,
        backend: VadBackend | None = None,
        channel_id: str = "A",
    ) -> None:
        self.cfg = cfg
        self.channel_id = channel_id
        self.backend = backend or build_backend(cfg)

        window_ms = self.backend.window_ms
        self._window = self.backend.window_samples
        self._preroll_windows = max(1, int(round(cfg.preroll_ms / window_ms)))
        self._silence_windows_to_end = max(1, int(round(cfg.silence_ms_to_end / window_ms)))
        self._max_windows = max(1, int(round(cfg.max_utterance_ms / window_ms)))
        self._min_speech_windows = max(1, int(round(cfg.min_utterance_ms / window_ms)))
        # Whisper does better with a little breathing room at the end, but not
        # the full silence timeout's worth.
        self._keep_trailing_windows = max(1, int(round(200 / window_ms)))

        self._state = State.SILENT
        self._pending = np.zeros(0, dtype=np.int16)
        self._preroll: deque[np.ndarray] = deque(maxlen=self._preroll_windows)
        self._current: list[np.ndarray] = []
        self._voiced_run = 0
        self._silence_run = 0
        self._speech_windows = 0
        self._seq = 0

    @property
    def state(self) -> State:
        return self._state

    @property
    def in_speech(self) -> bool:
        return self._state is State.SPEAKING

    def push(self, frame: np.ndarray) -> list[Utterance]:
        """Feed audio of any length. Returns any utterances that just closed."""
        self._pending = (
            frame.astype(np.int16)
            if self._pending.size == 0
            else np.concatenate((self._pending, frame.astype(np.int16)))
        )

        finished: list[Utterance] = []
        while self._pending.size >= self._window:
            window = self._pending[: self._window]
            self._pending = self._pending[self._window :]
            utterance = self._advance(window)
            if utterance is not None:
                finished.append(utterance)
        return finished

    def _advance(self, window: np.ndarray) -> Utterance | None:
        voiced = self.backend.is_speech(window)

        if self._state is State.SILENT:
            self._preroll.append(window)
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run >= self.cfg.start_frames:
                # The pre-roll already contains these voiced windows.
                self._current = list(self._preroll)
                self._preroll.clear()
                self._speech_windows = self._voiced_run
                self._silence_run = 0
                self._voiced_run = 0
                self._state = State.SPEAKING
            return None

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

    def _close(self, *, trim_trailing: bool) -> Utterance | None:
        windows = self._current
        speech_windows = self._speech_windows
        self._reset_segment()

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

    def _reset_segment(self) -> None:
        self._state = State.SILENT
        self._current = []
        self._preroll.clear()
        self._voiced_run = 0
        self._silence_run = 0
        self._speech_windows = 0

    def flush(self) -> Utterance | None:
        """Close any in-progress utterance. Used at shutdown and end of file."""
        if self._state is not State.SPEAKING:
            return None
        return self._close(trim_trailing=False)

    def reset(self) -> None:
        """Drop everything in flight — used when the duplex gate closes."""
        self._reset_segment()
        self._pending = np.zeros(0, dtype=np.int16)
        self.backend.reset()
