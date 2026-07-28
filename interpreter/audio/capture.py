"""Microphone capture: PortAudio callback thread -> asyncio frame queue.

The callback runs on a realtime audio thread and must never block, allocate
heavily, or raise. It does exactly three things: copy the buffer, convert it,
and hand it to the event loop via `call_soon_threadsafe`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np

from ..config import FRAME_SAMPLES, PIPELINE_SAMPLE_RATE
from . import devices
from .resample import StreamResampler, downmix_to_mono

log = logging.getLogger(__name__)

# If the consumer stalls this far behind, something is badly wrong; drop the
# oldest audio rather than growing memory without bound.
_MAX_QUEUED_FRAMES = 250  # 5 seconds at 20 ms/frame


class MicrophoneStream:
    """One microphone, delivering 20 ms mono int16 frames at 16 kHz.

    Opens at 16 kHz when the device allows it, otherwise opens at the device's
    native rate and resamples — WASAPI shared mode in particular often refuses
    arbitrary rates.
    """

    # A real microphone in a real room, so it can pick up a real speaker.
    acoustic = True

    def __init__(
        self,
        device: str | int | None = None,
        *,
        gain: float = 1.0,
        channel_id: str = "A",
    ) -> None:
        self.channel_id = channel_id
        self.gain = gain
        self.device_index = devices.resolve(device, kind="input")

        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream = None
        self._resampler: StreamResampler | None = None
        self._device_rate = PIPELINE_SAMPLE_RATE
        self._device_channels = 1
        self._pending = np.zeros(0, dtype=np.int16)
        self._dropped_frames = 0
        self._overflow_count = 0

    @property
    def description(self) -> str:
        return devices.describe(self.device_index, "input")

    def start(self) -> None:
        sd = devices._sounddevice()
        self._loop = asyncio.get_running_loop()

        self._device_rate, self._device_channels = self._negotiate_format(sd)
        self._resampler = StreamResampler(self._device_rate, PIPELINE_SAMPLE_RATE)

        # 20 ms at the device rate, so the callback fires at a steady cadence.
        blocksize = max(64, int(self._device_rate * 0.02))

        self._stream = sd.RawInputStream(
            samplerate=self._device_rate,
            blocksize=blocksize,
            device=self.device_index,
            channels=self._device_channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "channel %s capturing from %s at %d Hz (%d ch)",
            self.channel_id,
            self.description,
            self._device_rate,
            self._device_channels,
        )

    def _negotiate_format(self, sd) -> tuple[int, int]:
        """Pick a (samplerate, channels) pair the device will actually accept."""
        info = sd.query_devices(self.device_index, "input")
        native_rate = int(info["default_samplerate"])
        max_channels = max(1, int(info["max_input_channels"]))

        for rate in (PIPELINE_SAMPLE_RATE, native_rate, 48_000, 44_100):
            for channels in (1, max_channels):
                if channels > max_channels:
                    continue
                try:
                    sd.check_input_settings(
                        device=self.device_index,
                        samplerate=rate,
                        channels=channels,
                        dtype="int16",
                    )
                except Exception:  # noqa: BLE001 - probing, failure is expected
                    continue
                return rate, channels

        raise devices.AudioDeviceError(
            f"{self.description} rejected every sample rate we tried "
            f"(16000/{native_rate}/48000/44100 Hz). Pick a different input device."
        )

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        """PortAudio realtime thread. Keep this cheap and exception-free."""
        if status and status.input_overflow:
            self._overflow_count += 1

        try:
            pcm = np.frombuffer(bytes(indata), dtype=np.int16)
            if self._device_channels > 1:
                pcm = downmix_to_mono(pcm, self._device_channels)
            if self._resampler is not None and not self._resampler.passthrough:
                pcm = self._resampler.process(pcm)
            if self.gain != 1.0:
                pcm = (
                    (pcm.astype(np.float32) * self.gain)
                    .clip(-32768, 32767)
                    .astype(np.int16)
                )
            if self._loop is not None and pcm.size:
                self._loop.call_soon_threadsafe(self._enqueue, pcm)
        except Exception:  # noqa: BLE001 - a raise here would kill the audio thread
            log.exception("capture callback failed on channel %s", self.channel_id)

    def _enqueue(self, pcm: np.ndarray) -> None:
        """Runs on the event loop. Rebuffers into exact 20 ms frames."""
        self._pending = (
            pcm if self._pending.size == 0 else np.concatenate((self._pending, pcm))
        )
        while self._pending.size >= FRAME_SAMPLES:
            frame, self._pending = (
                self._pending[:FRAME_SAMPLES],
                self._pending[FRAME_SAMPLES:],
            )
            if self._queue.qsize() >= _MAX_QUEUED_FRAMES:
                self._queue.get_nowait()
                self._dropped_frames += 1
                if self._dropped_frames % 50 == 1:
                    log.warning(
                        "channel %s dropped %d frames — the pipeline is not keeping up",
                        self.channel_id,
                        self._dropped_frames,
                    )
            self._queue.put_nowait(frame)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield 20 ms mono int16 frames at 16 kHz until the stream is stopped."""
        while True:
            frame = await self._queue.get()
            if frame is None:  # sentinel from stop()
                return
            yield frame

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001 - teardown must not mask a real error
                log.debug("error closing input stream", exc_info=True)
            self._stream = None
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    @property
    def stats(self) -> dict[str, int]:
        return {"dropped_frames": self._dropped_frames, "overflows": self._overflow_count}


class ArrayMicrophone:
    """Feeds a fixed PCM array through the frame interface.

    Backs `--selftest` and the offline test suite, so the whole pipeline can be
    exercised with no microphone and no PortAudio.
    """

    # Nothing in a room can leak into a file, so the duplex gate stays off.
    acoustic = False

    def __init__(
        self,
        pcm: np.ndarray,
        *,
        channel_id: str = "A",
        realtime: bool = False,
        trailing_silence_ms: int = 1200,
    ) -> None:
        self.channel_id = channel_id
        self.gain = 1.0
        self.device_index = None
        self.realtime = realtime
        # A real conversation always ends with silence; without it the VAD
        # never sees the end of the final utterance.
        padding = np.zeros(
            PIPELINE_SAMPLE_RATE * trailing_silence_ms // 1000, dtype=np.int16
        )
        self._pcm = np.concatenate((pcm.astype(np.int16), padding))

    @property
    def description(self) -> str:
        return f"array ({self._pcm.size / PIPELINE_SAMPLE_RATE:.1f}s of PCM)"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    async def frames(self) -> AsyncIterator[np.ndarray]:
        for offset in range(0, self._pcm.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            if self.realtime:
                await asyncio.sleep(FRAME_SAMPLES / PIPELINE_SAMPLE_RATE)
            else:
                await asyncio.sleep(0)  # stay cooperative
            yield self._pcm[offset : offset + FRAME_SAMPLES]

    @property
    def stats(self) -> dict[str, int]:
        return {"dropped_frames": 0, "overflows": 0}
