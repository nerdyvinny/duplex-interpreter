"""Wires channels together and runs the conversation.

The cross-wiring is the part worth reading closely:

* **single_mic** — one channel. Its microphone and its speaker are the same
  physical room, so the sink points back at itself and the duplex guard is
  active.
* **dual_mic** — two channels, cross-wired. What microphone A hears is
  translated and played on **B's** earpiece, and vice versa. Nobody hears a
  translation of their own words, which is both less confusing and why the
  echo machinery can be switched off entirely.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

from . import registry
from .audio.capture import MicrophoneStream
from .audio.playback import SpeakerStream
from .audio.vad import SpeechSegmenter, VadBackend, build_backend
from .config import AppConfig, ChannelConfig, Mode
from .duplex import DuplexGuard
from .events import EventBus
from .pipeline import ChannelPipeline, Conversation, OutputSink
from .providers.base import STTProvider, TranslationProvider, TTSProvider
from .routing import LanguageRouter

log = logging.getLogger(__name__)

MicrophoneFactory = Callable[[ChannelConfig], object]
SpeakerFactory = Callable[[ChannelConfig], object]
VadFactory = Callable[[ChannelConfig], VadBackend]


@dataclass
class Channel:
    """Everything belonging to one microphone."""

    config: ChannelConfig
    microphone: object
    speaker: object
    guard: DuplexGuard
    router: LanguageRouter
    segmenter: SpeechSegmenter
    pipeline: ChannelPipeline | None = None


class Orchestrator:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        bus: EventBus | None = None,
        providers: tuple[STTProvider, TranslationProvider, TTSProvider] | None = None,
        microphone_factory: MicrophoneFactory | None = None,
        speaker_factory: SpeakerFactory | None = None,
        vad_factory: VadFactory | None = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus or EventBus()
        self.conversation = Conversation()

        self.stt, self.translation, self.tts = providers or registry.build_all(cfg.providers)

        self._microphone_factory = microphone_factory or self._default_microphone
        self._speaker_factory = speaker_factory or self._default_speaker
        # One backend per channel: Silero carries recurrent state, so two
        # microphones must not share an instance.
        self._vad_factory = vad_factory or (lambda _channel: build_backend(cfg.vad))

        self.channels: list[Channel] = []
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self._started = False

    # --- default hardware factories ---

    def _default_microphone(self, channel: ChannelConfig):
        return MicrophoneStream(
            channel.input_device, gain=channel.input_gain, channel_id=channel.id
        )

    def _default_speaker(self, channel: ChannelConfig):
        return SpeakerStream(channel.output_device, channel_id=channel.id)

    # --- construction ---

    def build(self) -> None:
        if self.channels:
            return

        # Before any audio device is opened: one clear message about a missing
        # key beats every utterance failing once the conversation is running.
        for provider in (self.stt, self.translation, self.tts):
            provider.preflight()

        shared_room = self.cfg.mode is Mode.SINGLE_MIC

        for channel_config in self.cfg.channels:
            microphone = self._microphone_factory(channel_config)
            speaker = self._speaker_factory(channel_config)
            # The acoustic gate costs latency and can swallow speech, so it is
            # only worth having when a real microphone can actually hear a
            # real speaker. File-fed audio and recording buffers can't.
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
            router = LanguageRouter(
                self.cfg.language_codes,
                pinned=None if channel_config.language == "auto" else channel_config.language,
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
            # A microphone can only ever hear its *own* speaker, so that is
            # what barge-in interrupts — not necessarily the sink it writes to.
            channel.guard.set_bargein_callback(channel.speaker.stop)

    def _sink_for(self, index: int) -> OutputSink:
        if self.cfg.mode is Mode.SINGLE_MIC:
            own = self.channels[index]
            return OutputSink(speaker=own.speaker, guard=own.guard)
        # dual_mic: play A's translations on B's earpiece.
        listener = self.channels[1 - index]
        return OutputSink(speaker=listener.speaker, guard=listener.guard)

    # --- lifecycle ---

    async def start(self) -> None:
        self.build()
        for channel in self.channels:
            channel.speaker.start()
            channel.microphone.start()
        self._started = True
        self._tasks = [
            asyncio.create_task(channel.pipeline.run(), name=f"pipeline-{channel.config.id}")
            for channel in self.channels
        ]
        log.info(
            "running in %s mode: %s <-> %s",
            self.cfg.mode.value,
            self.cfg.language_a.name,
            self.cfg.language_b.name,
        )

    async def run_until_stopped(self) -> None:
        """Run until `stop()`, or until *every* pipeline finishes on its own.

        Waiting on all of them matters when the channels have unequal amounts
        of audio — as they do in `--selftest`, where one channel is fed a file
        and the other is silent. Returning on the first completion would cut
        the talking channel off mid-sentence.
        """
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
            pipelines.result()  # re-raise a pipeline crash rather than exiting silently
        else:
            # Stopped by request: wind capture down, but leave in-flight
            # translations for shutdown() to drain.
            pipelines.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pipelines

    def stop(self) -> None:
        self._stop.set()

    async def shutdown(self, *, drain: bool = True) -> None:
        """Stop capture, let in-flight translations finish, close devices."""
        self._stop.set()

        for channel in self.channels:
            try:
                channel.microphone.stop()
            except Exception:  # noqa: BLE001 - teardown
                log.debug("error stopping microphone", exc_info=True)

        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

        if drain:
            for channel in self.channels:
                if channel.pipeline is not None:
                    try:
                        await asyncio.wait_for(channel.pipeline.drain(), timeout=15.0)
                    except (TimeoutError, asyncio.TimeoutError):
                        log.warning("channel %s did not drain in time", channel.config.id)

        for channel in self.channels:
            try:
                channel.speaker.close()
            except Exception:  # noqa: BLE001 - teardown
                log.debug("error closing speaker", exc_info=True)

        for provider in (self.stt, self.translation, self.tts):
            try:
                await provider.aclose()
            except Exception:  # noqa: BLE001 - teardown
                log.debug("error closing provider", exc_info=True)

        self._started = False

    async def __aenter__(self) -> Orchestrator:
        await self.start()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.shutdown()

    # --- reporting ---

    def stats(self) -> dict[str, dict[str, int]]:
        report: dict[str, dict[str, int]] = {}
        for channel in self.channels:
            report[channel.config.id] = {
                **channel.microphone.stats,
                **channel.speaker.stats,
                **channel.guard.stats,
            }
        return report

    def describe(self) -> list[str]:
        """Human-readable wiring summary, printed at startup."""
        lines = [f"mode: {self.cfg.mode.value}"]
        for index, channel in enumerate(self.channels):
            sink = self._sink_for(index)
            language = (
                "auto-detect"
                if channel.config.language == "auto"
                else self.cfg.slot_for(channel.config.language).name
            )
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
