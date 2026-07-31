# speaker output
#
# there is a buffer that portaudio pulls from on its own thread. I push the
# TTS audio in as it arrives from the network instead of waiting for the
# whole thing, so it starts talking on the first chunk. saves a few hundred
# ms per sentence which is a lot when you're waiting for a translation.
#
# stop() empties the buffer instantly, thats what makes interrupting work

import asyncio
import logging
import threading
from collections import deque

import numpy as np

from . import devices
from .resample import StreamResampler

log = logging.getLogger(__name__)


class SpeakerStream:
    acoustic = True  # makes real sound so a mic in the room will hear it

    def __init__(self, device=None, *, channel_id="A"):
        self.channel_id = channel_id
        self.device_index = devices.resolve(device, kind="output")

        # a real threading lock, not asyncio, because the portaudio thread
        # touches this buffer too
        self._lock = threading.Lock()
        self._buffer = deque()
        self._buffered_samples = 0
        self._generation = 0  # stop() bumps this to cancel whatever is playing
        self._stream = None
        self._device_rate = 24_000
        self._device_channels = 1
        self._underruns = 0
        self._idle = threading.Event()
        self._idle.set()
        # how many play() calls are currently feeding. outside of that an
        # empty buffer just means nobody is talking, not a problem
        self._feeding = 0

    @property
    def description(self):
        return devices.describe(self.device_index, "output")

    @property
    def sample_rate(self):
        return self._device_rate

    def start(self):
        sd = devices._sounddevice()
        self._device_rate, self._device_channels = self._negotiate_format(sd)
        self._stream = sd.RawOutputStream(
            samplerate=self._device_rate,
            blocksize=0,  # 0 lets portaudio pick, it goes as low as it can
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

    def _negotiate_format(self, sd):
        info = sd.query_devices(self.device_index, "output")
        native_rate = int(info["default_samplerate"])
        max_channels = max(1, int(info["max_output_channels"]))

        # 24000 first because thats what openai's tts gives me, so the
        # normal case needs no resampling at all
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
                except Exception:
                    continue
                return rate, channels

        raise devices.AudioDeviceError(
            f"{self.description} rejected every sample rate we tried. "
            "Pick a different output device."
        )

    def _callback(self, outdata, frames, time_info, status):
        # AUDIO THREAD. never block in here. if there isn't enough audio
        # just leave the rest as zeros, which is silence
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
                    self._buffer.popleft()  # used the whole thing
                else:
                    self._buffer[0] = head[take:]  # save the rest
                self._buffered_samples -= take

            if not self._buffer:
                self._idle.set()

        # only counts as an underrun if a play() was actually mid stream,
        # meaning more audio WAS coming and it didn't get here in time.
        # an empty buffer any other time is just nobody talking, which is
        # most of a conversation. my first version counted those and
        # reported like 4000 underruns on a totally fine run
        if filled < needed and self._feeding:
            self._underruns += 1

        outdata[:] = chunk.tobytes()

    def _enqueue(self, pcm):
        if pcm.size == 0:
            return
        if self._device_channels > 1:
            # np.repeat turns [a, b] into [a, a, b, b] which is exactly the
            # LRLR layout stereo wants
            pcm = np.repeat(pcm, self._device_channels)
        with self._lock:
            self._buffer.append(pcm)
            self._buffered_samples += pcm.size
            self._idle.clear()

    async def play(self, chunks, *, source_rate, on_first_audio=None):
        # returns False if stop() cut us off, True if it finished normally
        generation = self._generation
        resampler = StreamResampler(source_rate, self._device_rate)
        first = True

        try:
            async for raw in chunks:
                if self._generation != generation:
                    return False  # somebody barged in on us
                if not raw:
                    continue

                pcm = np.frombuffer(raw, dtype=np.int16)
                self._enqueue(resampler.process(pcm))

                if first:
                    first = False
                    # only NOW does the device actually expect audio. if I
                    # counted from the top of play() then waiting for the
                    # first chunk to download would look like an underrun
                    self._feeding += 1
                    if on_first_audio is not None:
                        on_first_audio()
        finally:
            if not first:
                self._feeding -= 1

        # the resampler holds a few samples back, get them out
        tail = resampler.process(np.zeros(0, dtype=np.int16), last=True)
        if tail.size:
            self._enqueue(tail)

        await self.drain(generation)
        return self._generation == generation

    async def drain(self, generation=None):
        # wait until the buffer is empty
        while True:
            if generation is not None and self._generation != generation:
                return  # cancelled, don't bother

            with self._lock:
                remaining = self._buffered_samples

            if remaining <= 0:
                # give the device a moment to actually push out the last bit
                await asyncio.sleep(0.02)
                return

            # sleep for roughly half of whats left, but keep it sane
            await asyncio.sleep(
                min(0.05, max(0.005, remaining / self._device_rate / 2))
            )

    def stop(self):
        # dump everything. any play() that is running returns False
        with self._lock:
            self._generation += 1
            self._buffer.clear()
            self._buffered_samples = 0
            self._idle.set()

    @property
    def is_playing(self):
        with self._lock:
            return self._buffered_samples > 0

    def close(self):
        self.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.debug("error closing output stream", exc_info=True)
            self._stream = None

    @property
    def stats(self):
        return {"underruns": self._underruns}


class RecordingSpeaker:
    # saves the audio instead of playing it. --selftest uses this to write
    # a wav you can listen to, and the tests use it to check the order
    # things played in without needing a sound card

    acoustic = False  # silent, so no microphone can ever pick it up

    def __init__(self, *, channel_id="A", sample_rate=24_000):
        self.channel_id = channel_id
        self._device_rate = sample_rate
        self._generation = 0
        self.chunks = []
        self.played_order = []

    @property
    def description(self):
        return "recording buffer"

    @property
    def sample_rate(self):
        return self._device_rate

    def start(self):
        pass

    async def play(self, chunks, *, source_rate, on_first_audio=None):
        generation = self._generation
        resampler = StreamResampler(source_rate, self._device_rate)
        first = True

        async for raw in chunks:
            if self._generation != generation:
                return False
            if not raw:
                continue

            self.chunks.append(
                resampler.process(np.frombuffer(raw, dtype=np.int16))
            )
            if first:
                first = False
                if on_first_audio is not None:
                    on_first_audio()

        # always flush. on a short sentence the leftover in the resampler
        # can literally be the entire thing
        tail = resampler.process(np.zeros(0, dtype=np.int16), last=True)
        if tail.size:
            self.chunks.append(tail)
        return True

    async def drain(self, generation=None):
        pass

    def stop(self):
        self._generation += 1

    @property
    def is_playing(self):
        return False

    def close(self):
        pass

    @property
    def stats(self):
        return {"underruns": 0}

    def pcm(self):
        if not self.chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self.chunks)
