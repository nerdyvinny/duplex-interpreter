# builds all the pieces and hooks them up to each other
#
# the wiring is the confusing part so heres what happens:
#
# single_mic: one channel. the mic and the speaker are in the same room so
#   the output goes back to itself and the echo stuff is turned on.
#
# dual_mic: two channels, CROSSED OVER. what mic A hears gets played in B's
#   earpiece and the other way around. so nobody hears their own words
#   translated back at them, which is less confusing, and its also why I
#   can turn the whole echo thing off in this mode

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from . import registry
from .audio.capture import MicrophoneStream
from .audio.playback import SpeakerStream
from .audio.vad import SpeechSegmenter, build_backend
from .config import Mode
from .duplex import DuplexGuard
from .events import EventBus
from .pipeline import ChannelPipeline, Conversation, OutputSink
from .routing import LanguageRouter

log = logging.getLogger(__name__)

# how long to wait for in flight translations after the mic stops.
# its a lot but a local model on cpu can take a few seconds per sentence
# and cutting off the last thing somebody said is worse than waiting
DRAIN_TIMEOUT_S = 60.0


@dataclass
class Channel:
    # all the stuff that belongs to one microphone
    config: object
    microphone: object
    speaker: object
    guard: object
    router: object
    segmenter: object
    pipeline: object = None


class Orchestrator:
    def __init__(self, cfg, *, bus=None, providers=None,
                 microphone_factory=None, speaker_factory=None,
                 vad_factory=None):
        self.cfg = cfg
        self.bus = bus or EventBus()
        self.conversation = Conversation()

        if providers is None:
            providers = registry.build_all(cfg.providers)
        self.stt, self.translation, self.tts = providers

        # the factory arguments are so the tests can swap in fake mics and
        # speakers without needing real hardware
        self._microphone_factory = microphone_factory or self._default_microphone
        self._speaker_factory = speaker_factory or self._default_speaker
        # one vad per channel! silero remembers state between frames so two
        # mics sharing one would confuse each other
        self._vad_factory = vad_factory or (lambda _channel: build_backend(cfg.vad))

        self.channels = []
        self._tasks = []
        self._stop = asyncio.Event()
        self._started = False

    # ---- default hardware ----

    def _default_microphone(self, channel):
        return MicrophoneStream(
            channel.input_device, gain=channel.input_gain, channel_id=channel.id
        )

    def _default_speaker(self, channel):
        return SpeakerStream(channel.output_device, channel_id=channel.id)

    # ---- setup ----

    def build(self):
        if self.channels:
            return  # already did this

        # do this before opening any audio device. it checks the api keys
        # and downloads models, so a missing key is one clear error at the
        # start instead of a freeze in the middle of the first sentence
        for provider in (self.stt, self.translation, self.tts):
            provider.preflight(self.cfg)

        shared_room = self.cfg.mode is Mode.SINGLE_MIC

        for channel_config in self.cfg.channels:
            microphone = self._microphone_factory(channel_config)
            speaker = self._speaker_factory(channel_config)

            # the gate costs latency and can eat words, so only turn it on
            # if a real mic could actually hear a real speaker. a wav file
            # and a recording buffer obviously cant
            can_hear_itself = (
                shared_room
                and getattr(microphone, "acoustic", True)
                and getattr(speaker, "acoustic", True)
            )

            guard = DuplexGuard(
                self.cfg.duplex,
                channel_id=channel_config.id,
                gate_enabled=can_hear_itself,
                echo_guard_enabled=shared_room,
            )

            if channel_config.language == "auto":
                pinned = None
            else:
                pinned = channel_config.language
            router = LanguageRouter(
                self.cfg.language_codes,
                pinned=pinned,
                channel_id=channel_config.id,
            )

            segmenter = SpeechSegmenter(
                self.cfg.vad,
                backend=self._vad_factory(channel_config),
                channel_id=channel_config.id,
            )

            self.channels.append(
                Channel(
                    config=channel_config,
                    microphone=microphone,
                    speaker=speaker,
                    guard=guard,
                    router=router,
                    segmenter=segmenter,
                )
            )

        # second loop because the sink for channel A might be channel B's
        # speaker, so they all have to exist before I can wire them
        for index, channel in enumerate(self.channels):
            sink = self._sink_for(index)
            channel.pipeline = ChannelPipeline(
                cfg=self.cfg,
                channel_id=channel.config.id,
                microphone=channel.microphone,
                segmenter=channel.segmenter,
                guard=channel.guard,
                router=channel.router,
                sink=sink,
                stt=self.stt,
                translation=self.translation,
                tts=self.tts,
                bus=self.bus,
                conversation=self.conversation,
            )
            # a mic can only hear its OWN speaker, so thats the one barge in
            # stops. not always the same as the sink it writes to
            channel.guard.set_bargein_callback(channel.speaker.stop)

    def _sink_for(self, index):
        if self.cfg.mode is Mode.SINGLE_MIC:
            own = self.channels[index]
            return OutputSink(speaker=own.speaker, guard=own.guard)
        # dual_mic, swap them. 1 - index turns 0 into 1 and 1 into 0
        listener = self.channels[1 - index]
        return OutputSink(speaker=listener.speaker, guard=listener.guard)

    # ---- running ----

    async def start(self):
        self.build()
        for channel in self.channels:
            channel.speaker.start()
            channel.microphone.start()
        self._started = True

        self._tasks = [
            asyncio.create_task(
                channel.pipeline.run(), name=f"pipeline-{channel.config.id}"
            )
            for channel in self.channels
        ]
        log.info(
            "running in %s mode: %s <-> %s",
            self.cfg.mode.value,
            self.cfg.language_a.name,
            self.cfg.language_b.name,
        )

    async def run_until_stopped(self):
        # runs until stop() or until ALL the pipelines end by themselves.
        # it has to be all of them, not the first one. in --selftest one
        # channel gets the wav file and the other one gets nothing, so if I
        # returned on the first finish it would cut off the talking channel
        if not self._started:
            await self.start()
        if not self._tasks:
            return

        pipelines = asyncio.gather(*self._tasks)
        stop_waiter = asyncio.create_task(self._stop.wait(), name="stop-signal")
        try:
            done, _ = await asyncio.wait(
                {pipelines, stop_waiter}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            stop_waiter.cancel()

        if pipelines in done:
            pipelines.result()  # re raise if a pipeline crashed
        else:
            # user hit ctrl-c. stop recording but leave the in flight
            # translations for shutdown() to finish
            pipelines.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pipelines

    def stop(self):
        self._stop.set()

    async def shutdown(self, *, drain=True):
        # stop recording, let the last translations finish, close everything
        self._stop.set()

        for channel in self.channels:
            try:
                channel.microphone.stop()
            except Exception:
                log.debug("error stopping microphone", exc_info=True)

        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

        if drain:
            for channel in self.channels:
                if channel.pipeline is None:
                    continue
                try:
                    await asyncio.wait_for(
                        channel.pipeline.drain(), timeout=DRAIN_TIMEOUT_S
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    log.warning(
                        "channel %s still had work in flight after %.0fs; "
                        "the last utterance may be incomplete",
                        channel.config.id,
                        DRAIN_TIMEOUT_S,
                    )

        for channel in self.channels:
            try:
                channel.speaker.close()
            except Exception:
                log.debug("error closing speaker", exc_info=True)

        for provider in (self.stt, self.translation, self.tts):
            try:
                await provider.aclose()
            except Exception:
                log.debug("error closing provider", exc_info=True)

        self._started = False

    # lets you do "async with Orchestrator(cfg) as o:"
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc_info):
        await self.shutdown()

    # ---- printing stuff ----

    def stats(self):
        report = {}
        for channel in self.channels:
            report[channel.config.id] = {
                **channel.microphone.stats,
                **channel.speaker.stats,
                **channel.guard.stats,
            }
        return report

    def describe(self):
        # the summary I print when it starts up
        lines = [f"mode: {self.cfg.mode.value}"]

        for index, channel in enumerate(self.channels):
            sink = self._sink_for(index)
            if channel.config.language == "auto":
                language = "auto-detect"
            else:
                language = self.cfg.slot_for(channel.config.language).name
            lines.append(
                f"  channel {channel.config.id}: {channel.microphone.description} "
                f"({language}) -> {sink.speaker.description}"
            )

        if any(c.guard.gate_enabled for c in self.channels):
            detail = "on (the microphone can hear the speaker)"
        elif any(c.guard.echo_guard_enabled for c in self.channels):
            detail = "text-only (no acoustic path between the devices)"
        else:
            detail = "off (each person has their own headset)"
        lines.append(f"  echo suppression: {detail}")

        return lines
