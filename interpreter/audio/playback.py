"""Speaker output: a ring buffer the PortAudio callback drains.

TTS chunks are pushed in as they arrive from the network, so audio starts
playing on the first chunk instead of after synthesis completes. That is worth
a few hundred milliseconds per utterance.

`stop()` clears the buffer immediately, which is what makes barge-in possible.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from collections.abc import AsyncIterator

import numpy as np

from . import devices
from .resample import StreamResampler

log = logging.getLogger(__name__)


class SpeakerStream:
    """One output device, fed by an in-memory ring buffer."""

    # Makes actual sound, so a microphone in the same room can hear it.
    acoustic = True

    def __init__(self, device: str | int | None = None, *, channel_id: str = "A") -> None:
        self.channel_id = channel_id
        self.device_index = devices.resolve(device, kind="output")

        self._lock = threading.Lock()
        self._buffer: deque[np.ndarray] = deque()
        self._buffered_samples = 0
        self._generation = 0  # bumped by stop() to cancel in-flight feeds
        self._stream = None
        self._device_rate = 24_000
        self._device_channels = 1
        self._underruns = 0
        self._idle = threading.Event()
        self._idle.set()

    @property
    def description(self) -> str:
        return devices.describe(self.device_index, "output")

    @property
    def sample_rate(self) -> int:
        return self._device_rate

    def start(self) -> None:
        sd = devices._sounddevice()
        self._device_rate, self._device_channels = self._negotiate_format(sd)
        self._stream = sd.RawOutputStream(
            samplerate=self._device_rate,
            blocksize=0,  # let PortAudio choose; lowest latency it can manage
            device=self.device_index,
            channels=self._device_channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "channel %s playing to %s at %d Hz (%d ch)",
            self.channel_id,
            self.description,
            self._device_rate,
            self._device_channels,
        )

    def _negotiate_format(self, sd) -> tuple[int, int]:
        info = sd.query_devices(self.device_index, "output")
        native_rate = int(info["default_samplerate"])
        max_channels = max(1, int(info["max_output_channels"]))
        # 24 kHz first: it is what gpt-4o-mini-tts emits, so the common case
        # needs no resampling at all.
        for rate in (24_000, native_rate, 48_000, 44_100):
            for channels in (1, min(2, max_channels)):
                if channels > max_channels:
                    continue
                try:
                    sd.check_output_settings(
                        device=self.device_index,
                        samplerate=rate,
                        channels=channels,
                        dtype="int16",
                    )
                except Exception:  # noqa: BLE001 - probing
                    continue
                return rate, channels
        raise devices.AudioDeviceError(
            f"{self.description} rejected every sample rate we tried. "
            "Pick a different output device."
        )

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        """PortAudio realtime thread. Never blocks; pads with silence."""
        needed = frames * self._device_channels
        chunk = np.zeros(needed, dtype=np.int16)
        filled = 0

        with self._lock:
            while filled < needed and self._buffer:
                head = self._buffer[0]
                take = min(head.size, needed - filled)
                chunk[filled : filled + take] = head[:take]
                filled += take
                if take == head.size:
                    self._buffer.popleft()
                else:
                    self._buffer[0] = head[take:]
                self._buffered_samples -= take
            if not self._buffer:
                self._idle.set()

        if filled == 0 and status:
            self._underruns += 1
        outdata[:] = chunk.tobytes()

    def _enqueue(self, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        if self._device_channels > 1:
            pcm = np.repeat(pcm, self._device_channels)
        with self._lock:
            self._buffer.append(pcm)
            self._buffered_samples += pcm.size
            self._idle.clear()

    async def play(
        self,
        chunks: AsyncIterator[bytes],
        *,
        source_rate: int,
        on_first_audio=None,
    ) -> bool:
        """Stream TTS bytes to the speaker.

        Returns False if `stop()` interrupted this playback, True if it ran to
        completion.
        """
        generation = self._generation
        resampler = StreamResampler(source_rate, self._device_rate)
        first = True

        async for raw in chunks:
            if self._generation != generation:
                return False  # barged in on
            if not raw:
                continue
            pcm = np.frombuffer(raw, dtype=np.int16)
            self._enqueue(resampler.process(pcm))
            if first:
                first = False
                if on_first_audio is not None:
                    on_first_audio()

        tail = resampler.process(np.zeros(0, dtype=np.int16), last=True)
        if tail.size:
            self._enqueue(tail)

        await self.drain(generation)
        return self._generation == generation

    async def drain(self, generation: int | None = None) -> None:
        """Wait until the buffer empties (or playback is cancelled)."""
        while True:
            if generation is not None and self._generation != generation:
                return
            with self._lock:
                remaining = self._buffered_samples
            if remaining <= 0:
                # Give the device a beat to actually emit the last block.
                await asyncio.sleep(0.02)
                return
            await asyncio.sleep(min(0.05, max(0.005, remaining / self._device_rate / 2)))

    def stop(self) -> None:
        """Drop everything queued. Any in-flight `play()` returns False."""
        with self._lock:
            self._generation += 1
            self._buffer.clear()
            self._buffered_samples = 0
            self._idle.set()

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._buffered_samples > 0

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001 - teardown
                log.debug("error closing output stream", exc_info=True)
            self._stream = None

    @property
    def stats(self) -> dict[str, int]:
        return {"underruns": self._underruns}


class RecordingSpeaker:
    """Collects PCM instead of playing it.

    Used by `--selftest` (writes a WAV you can listen to) and by the offline
    tests, so playback ordering can be asserted without audio hardware.
    """

    # Silent by construction, so it can never be picked up by a microphone.
    acoustic = False

    def __init__(self, *, channel_id: str = "A", sample_rate: int = 24_000) -> None:
        self.channel_id = channel_id
        self._device_rate = sample_rate
        self._generation = 0
        self.chunks: list[np.ndarray] = []
        self.played_order: list[str] = []

    @property
    def description(self) -> str:
        return "recording buffer"

    @property
    def sample_rate(self) -> int:
        return self._device_rate

    def start(self) -> None:
        return None

    async def play(
        self,
        chunks: AsyncIterator[bytes],
        *,
        source_rate: int,
        on_first_audio=None,
    ) -> bool:
        generation = self._generation
        resampler = StreamResampler(source_rate, self._device_rate)
        first = True
        async for raw in chunks:
            if self._generation != generation:
                return False
            if not raw:
                continue
            self.chunks.append(resampler.process(np.frombuffer(raw, dtype=np.int16)))
            if first:
                first = False
                if on_first_audio is not None:
                    on_first_audio()

        # The resampler holds back a filter's worth of samples. On a short
        # utterance that can be the entire thing, so always flush the tail.
        tail = resampler.process(np.zeros(0, dtype=np.int16), last=True)
        if tail.size:
            self.chunks.append(tail)
        return True

    async def drain(self, generation: int | None = None) -> None:
        return None

    def stop(self) -> None:
        self._generation += 1

    @property
    def is_playing(self) -> bool:
        return False

    def close(self) -> None:
        return None

    @property
    def stats(self) -> dict[str, int]:
        return {"underruns": 0}

    def pcm(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self.chunks)
