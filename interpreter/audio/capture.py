# microphone input
#
# portaudio calls _callback on its own thread whenever it has audio. that
# thread is realtime so you are not allowed to do anything slow in there or
# block it, or the audio glitches. so the callback does the bare minimum
# and hands the data over to the asyncio loop

import asyncio
import logging

import numpy as np

from ..config import FRAME_SAMPLES, PIPELINE_SAMPLE_RATE
from . import devices
from .resample import StreamResampler, downmix_to_mono

log = logging.getLogger(__name__)

# if we get this far behind something is very wrong, so start throwing away
# the oldest audio instead of eating all the ram. 250 frames = 5 seconds
_MAX_QUEUED_FRAMES = 250


class MicrophoneStream:
    # gives you 20ms mono int16 frames at 16khz.
    #
    # it opens at 16khz if the device allows, otherwise it opens at whatever
    # the device wants and resamples. windows WASAPI especially just refuses
    # random sample rates

    acoustic = True  # a real mic in a real room, it can hear the speaker

    def __init__(self, device=None, *, gain=1.0, channel_id="A"):
        self.channel_id = channel_id
        self.gain = gain
        self.device_index = devices.resolve(device, kind="input")

        self._queue = asyncio.Queue()
        self._loop = None
        self._stream = None
        self._resampler = None
        self._device_rate = PIPELINE_SAMPLE_RATE
        self._device_channels = 1
        self._pending = np.zeros(0, dtype=np.int16)
        self._dropped_frames = 0
        self._overflow_count = 0

    @property
    def description(self):
        return devices.describe(self.device_index, "input")

    def start(self):
        sd = devices._sounddevice()
        self._loop = asyncio.get_running_loop()

        self._device_rate, self._device_channels = self._negotiate_format(sd)
        self._resampler = StreamResampler(self._device_rate, PIPELINE_SAMPLE_RATE)

        # ask for 20ms worth so the callback fires at a steady rate
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

    def _negotiate_format(self, sd):
        # try a bunch of sample rates until one works
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
                except Exception:
                    continue  # nope, try the next one
                return rate, channels

        raise devices.AudioDeviceError(
            f"{self.description} rejected every sample rate we tried "
            f"(16000/{native_rate}/48000/44100 Hz). Pick a different input device."
        )

    def _callback(self, indata, frames, time_info, status):
        # THIS RUNS ON THE AUDIO THREAD. keep it fast and never raise
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
                # hand it to the event loop, this is the threadsafe way
                self._loop.call_soon_threadsafe(self._enqueue, pcm)
        except Exception:
            # if this raises portaudio kills the whole audio thread and
            # the app just goes silent with no error, so catch everything
            log.exception("capture callback failed on channel %s", self.channel_id)

    def _enqueue(self, pcm):
        # runs on the event loop. cuts everything into exactly 20ms frames
        if self._pending.size == 0:
            self._pending = pcm
        else:
            self._pending = np.concatenate((self._pending, pcm))

        while self._pending.size >= FRAME_SAMPLES:
            frame = self._pending[:FRAME_SAMPLES]
            self._pending = self._pending[FRAME_SAMPLES:]

            if self._queue.qsize() >= _MAX_QUEUED_FRAMES:
                self._queue.get_nowait()  # drop the oldest one
                self._dropped_frames += 1
                # only log occasionally, otherwise it spams
                if self._dropped_frames % 50 == 1:
                    log.warning(
                        "channel %s dropped %d frames - the pipeline is not keeping up",
                        self.channel_id,
                        self._dropped_frames,
                    )

            self._queue.put_nowait(frame)

    async def frames(self):
        while True:
            frame = await self._queue.get()
            if frame is None:
                return  # stop() puts a None in to end the loop
            yield frame

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.debug("error closing input stream", exc_info=True)
            self._stream = None

        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    @property
    def stats(self):
        return {
            "dropped_frames": self._dropped_frames,
            "overflows": self._overflow_count,
        }


class ArrayMicrophone:
    # pretends to be a microphone but just plays back an array.
    # this is what --selftest uses, and all the tests, so I can run the
    # whole thing with no microphone and no portaudio at all

    acoustic = False  # its a file, nothing in the room can leak into it

    def __init__(self, pcm, *, channel_id="A", realtime=False,
                 trailing_silence_ms=1200):
        self.channel_id = channel_id
        self.gain = 1.0
        self.device_index = None
        self.realtime = realtime

        # a real conversation always ends with silence. without this the
        # vad never notices the last sentence ended and it gets lost
        padding = np.zeros(
            PIPELINE_SAMPLE_RATE * trailing_silence_ms // 1000, dtype=np.int16
        )
        self._pcm = np.concatenate((pcm.astype(np.int16), padding))

    @property
    def description(self):
        return f"array ({self._pcm.size / PIPELINE_SAMPLE_RATE:.1f}s of PCM)"

    def start(self):
        pass

    def stop(self):
        pass

    async def frames(self):
        for offset in range(0, self._pcm.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            if self.realtime:
                # actually wait, so the timings match real life
                await asyncio.sleep(FRAME_SAMPLES / PIPELINE_SAMPLE_RATE)
            else:
                await asyncio.sleep(0)  # just let other tasks run
            yield self._pcm[offset : offset + FRAME_SAMPLES]

    @property
    def stats(self):
        return {"dropped_frames": 0, "overflows": 0}
