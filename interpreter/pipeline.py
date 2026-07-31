"""One channel's journey from microphone to translated speech.

Concurrency model, and why it is shaped this way:

* Each channel runs its own `ChannelPipeline`. Two channels never wait on each
  other, which is what makes "both people talk at once" actually simultaneous
  rather than merely interleaved.
* Within a channel, up to `max_in_flight` utterances are transcribed and
  translated concurrently, so a slow network call on one sentence doesn't
  stall the next. The concurrency limit is released *before* playback, so
  utterance N+1 can be in translation while N is still being spoken.
* Playback is strictly ordered by sequence number. Without that, a short
  fast sentence would overtake a long slow one and the conversation would
  come out scrambled. Every exit path releases its slot, including drops and
  errors, so one failure can't deadlock everything behind it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .audio.vad import SpeechSegmenter, Utterance
from .config import AppConfig, ConfigError
from .duplex import DuplexGuard
from .events import EventBus, PipelineEvent, Stage
from .providers.base import ProviderError, STTProvider, TranslationProvider, TTSProvider
from .routing import LanguageRouter

log = logging.getLogger(__name__)


class PlaybackOrder:
    """Serializes playback by sequence number without blocking earlier stages.

    Releases arrive out of order — a dropped or failed utterance releases as
    soon as it is filtered, long before the slower sentences ahead of it have
    finished translating. So the cursor may only step forward over a *run* of
    already-released sequences; jumping it straight to the highest one seen
    would open the gate for every pending utterance simultaneously and let
    them play in whatever order translation happened to finish.
    """

    def __init__(self) -> None:
        self._next = 1
        self._released: set[int] = set()
        self._condition = asyncio.Condition()

    async def wait_turn(self, seq: int) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._next == seq)

    async def release(self, seq: int) -> None:
        async with self._condition:
            self._released.add(seq)
            while self._next in self._released:
                self._released.discard(self._next)
                self._next += 1
            self._condition.notify_all()


@dataclass
class Conversation:
    """Shared transcript history, used as translation context.

    Both directions share one log: resolving "he said he'd be late" needs the
    other person's turns, not just this microphone's.
    """

    turns: list[str] = field(default_factory=list)
    limit: int = 40

    def add(self, language: str, text: str) -> None:
        self.turns.append(f"{language}: {text}")
        if len(self.turns) > self.limit:
            del self.turns[: len(self.turns) - self.limit]

    def recent(self, count: int) -> list[str]:
        return self.turns[-count:] if count > 0 else []


@dataclass
class OutputSink:
    """Where a channel's translations are played, and who can hear them."""

    speaker: object  # SpeakerStream or RecordingSpeaker
    guard: DuplexGuard  # the guard for the microphone that shares this room


class ChannelPipeline:
    """One microphone, one segmenter, one output."""

    def __init__(
        self,
        *,
        cfg: AppConfig,
        channel_id: str,
        microphone,
        segmenter: SpeechSegmenter,
        guard: DuplexGuard,
        router: LanguageRouter,
        sink: OutputSink,
        stt: STTProvider,
        translation: TranslationProvider,
        tts: TTSProvider,
        bus: EventBus,
        conversation: Conversation,
        max_in_flight: int = 3,
    ) -> None:
        self.cfg = cfg
        self.channel_id = channel_id
        self.microphone = microphone
        self.segmenter = segmenter
        self.guard = guard
        self.router = router
        self.sink = sink
        self.stt = stt
        self.translation = translation
        self.tts = tts
        self.bus = bus
        self.conversation = conversation

        self._semaphore = asyncio.Semaphore(max_in_flight)
        self._order = PlaybackOrder()
        self._tasks: set[asyncio.Task] = set()
        self._stopping = False

    async def run(self) -> None:
        """Consume frames until the microphone stops."""
        try:
            async for frame in self.microphone.frames():
                if self._stopping:
                    break
                if self.guard.observe(frame):
                    # Our own speaker is talking; drop this audio and forget
                    # any half-open segment so the echo can't start one.
                    self.segmenter.reset()
                    continue
                for utterance in self.segmenter.push(frame):
                    self._spawn(utterance)
        except asyncio.CancelledError:
            raise
        finally:
            # Only flush when the result will actually be spawned: `flush()`
            # consumes a sequence number, and a number nothing ever releases
            # would stall every utterance queued behind it.
            if not self._stopping:
                trailing = self.segmenter.flush()
                if trailing is not None:
                    self._spawn(trailing)

    def _spawn(self, utterance: Utterance) -> None:
        self.bus.emit(
            PipelineEvent(
                seq=utterance.seq,
                channel_id=self.channel_id,
                stage=Stage.CAPTURED,
                audio_seconds=utterance.duration_seconds,
            )
        )
        task = asyncio.create_task(
            self._process(utterance), name=f"{self.channel_id}-{utterance.seq}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _process(self, utterance: Utterance) -> None:
        seq = utterance.seq
        timings: dict[str, float] = {}
        released = False

        try:
            async with self._semaphore:
                transcript = await self._timed(
                    timings,
                    "stt",
                    self._transcribe(utterance),
                )
                if transcript is None or transcript.is_empty:
                    self._drop(seq, timings, "no speech recognized")
                    return

                if self.guard.is_self_echo(transcript.text):
                    self._drop(seq, timings, "self-echo of our own playback")
                    return

                decision = self.router.route(transcript)
                if decision.reject:
                    self._drop(seq, timings, decision.reason)
                    return

                self.bus.emit(
                    PipelineEvent(
                        seq=seq,
                        channel_id=self.channel_id,
                        stage=Stage.TRANSCRIBED,
                        source_lang=decision.source,
                        target_lang=decision.target,
                        source_text=transcript.text,
                        timings=dict(timings),
                        audio_seconds=utterance.duration_seconds,
                        detail=decision.reason,
                    )
                )

                context = self.conversation.recent(self.cfg.providers.context_turns)
                self.conversation.add(decision.source, transcript.text)

                translated = await self._timed(
                    timings,
                    "translate",
                    self._translate(transcript.text, decision, context),
                )
                if not translated:
                    self._drop(seq, timings, "translation was empty")
                    return

            self.bus.emit(
                PipelineEvent(
                    seq=seq,
                    channel_id=self.channel_id,
                    stage=Stage.TRANSLATED,
                    source_lang=decision.source,
                    target_lang=decision.target,
                    source_text=transcript.text,
                    target_text=translated,
                    timings=dict(timings),
                )
            )

            await self._order.wait_turn(seq)
            try:
                await self._speak(seq, decision, transcript.text, translated, timings)
            finally:
                released = True
                await self._order.release(seq)

        except asyncio.CancelledError:
            raise
        except ConfigError:
            # Nothing the conversation can do about a missing key. Let it
            # reach the orchestrator and stop the run with one clear message.
            raise
        except ProviderError as exc:
            self._error(seq, timings, str(exc))
        except Exception as exc:  # noqa: BLE001 - one bad turn must not end the call
            log.exception("channel %s utterance %d failed", self.channel_id, seq)
            self._error(seq, timings, f"{type(exc).__name__}: {exc}")
        finally:
            if not released:
                await self._order.release(seq)

    async def _speak(
        self,
        seq: int,
        decision,
        source_text: str,
        translated: str,
        timings: dict[str, float],
    ) -> None:
        voice = self.cfg.slot_for(decision.target).voice
        started = time.monotonic()
        audio = await self.tts.synthesize(translated, language=decision.target, voice=voice)

        guard = self.sink.guard
        guard.note_spoken(translated)
        guard.playback_started()

        def _first_audio() -> None:
            timings["tts"] = (time.monotonic() - started) * 1000.0
            self.bus.emit(
                PipelineEvent(
                    seq=seq,
                    channel_id=self.channel_id,
                    stage=Stage.SPEAKING,
                    source_lang=decision.source,
                    target_lang=decision.target,
                    source_text=source_text,
                    target_text=translated,
                    timings=dict(timings),
                )
            )

        try:
            completed = await self.sink.speaker.play(
                audio.chunks,
                source_rate=audio.sample_rate,
                on_first_audio=_first_audio,
            )
        finally:
            guard.playback_finished()

        timings.setdefault("tts", (time.monotonic() - started) * 1000.0)
        self.bus.emit(
            PipelineEvent(
                seq=seq,
                channel_id=self.channel_id,
                stage=Stage.DONE,
                source_lang=decision.source,
                target_lang=decision.target,
                source_text=source_text,
                target_text=translated,
                timings=dict(timings),
                detail=None if completed else "interrupted",
            )
        )

    # --- stage helpers, each retrying once on a transient provider failure ---

    async def _transcribe(self, utterance: Utterance):
        language = self.router.pinned
        return await _retry_once(
            lambda: self.stt.transcribe(
                utterance.pcm,
                language=language,
                candidates=self.cfg.language_codes,
            ),
            what="transcription",
        )

    async def _translate(self, text: str, decision, context: list[str]) -> str:
        if decision.source == decision.target:
            return text
        return await _retry_once(
            lambda: self.translation.translate(
                text,
                source=decision.source,
                target=decision.target,
                context=context,
            ),
            what="translation",
        )

    @staticmethod
    async def _timed(timings: dict[str, float], key: str, awaitable):
        started = time.monotonic()
        try:
            return await awaitable
        finally:
            timings[key] = (time.monotonic() - started) * 1000.0

    def _drop(self, seq: int, timings: dict[str, float], reason: str) -> None:
        self.bus.emit(
            PipelineEvent(
                seq=seq,
                channel_id=self.channel_id,
                stage=Stage.DROPPED,
                timings=dict(timings),
                detail=reason,
            )
        )

    def _error(self, seq: int, timings: dict[str, float], message: str) -> None:
        self.bus.emit(
            PipelineEvent(
                seq=seq,
                channel_id=self.channel_id,
                stage=Stage.ERROR,
                timings=dict(timings),
                detail=message,
            )
        )

    async def drain(self) -> None:
        """Let anything still in flight finish."""
        self._stopping = True
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)


async def _retry_once(factory, *, what: str, backoff: float = 0.4):
    """Retry a provider call once. Network blips shouldn't cost a sentence.

    `ConfigError` is deliberately not caught: a missing API key won't fix
    itself, and retrying only doubles the noise before the same failure.
    """
    try:
        return await factory()
    except ProviderError as exc:
        log.warning("%s failed (%s); retrying once", what, exc)
        await asyncio.sleep(backoff)
        return await factory()
