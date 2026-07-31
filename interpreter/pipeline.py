# this is where one microphone's audio turns into translated speech
#
# some notes to myself about how this works, because I rewrote it like
# four times before it stopped sounding scrambled:
#
# - each mic gets its own ChannelPipeline. they don't wait on each other,
#   thats what makes both people able to talk at the same time
# - inside one channel I let up to 3 sentences be transcribed/translated at
#   once, so a slow api call on sentence 1 doesn't block sentence 2
# - BUT playback has to be in order. if a short sentence finishes fast it
#   will jump ahead of a long one and the conversation comes out backwards.
#   PlaybackOrder below is what fixes that

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .config import ConfigError
from .events import PipelineEvent, Stage
from .providers.base import ProviderError

log = logging.getLogger(__name__)


class PlaybackOrder:
    # makes sentences play in the order they were said, 1 then 2 then 3
    #
    # the tricky part: a sentence that gets dropped (nothing recognized,
    # or it was our own echo) is "done" way before the slow ones in front
    # of it. my first version just did _next = max(_next, seq + 1) and that
    # let one early drop unlock EVERYTHING waiting, so they all played at
    # once in whatever order they finished. so now I keep a set of the ones
    # that finished and only move the counter forward one step at a time.

    def __init__(self):
        self._next = 1
        self._released = set()
        self._condition = asyncio.Condition()

    async def wait_turn(self, seq):
        async with self._condition:
            await self._condition.wait_for(lambda: self._next == seq)

    async def release(self, seq):
        async with self._condition:
            self._released.add(seq)
            # walk forward over anything already finished
            while self._next in self._released:
                self._released.discard(self._next)
                self._next += 1
            self._condition.notify_all()


@dataclass
class Conversation:
    # everything both people have said so far. I give the last few lines to
    # the translator as context, otherwise stuff like "he said he'd be late"
    # has no idea who "he" is. both directions share one list on purpose

    turns: list = field(default_factory=list)
    limit: int = 40

    def add(self, language, text):
        self.turns.append(f"{language}: {text}")
        if len(self.turns) > self.limit:
            del self.turns[: len(self.turns) - self.limit]

    def recent(self, count):
        if count > 0:
            return self.turns[-count:]
        return []


@dataclass
class OutputSink:
    # where this channel's translations get played, and whose guard
    # belongs to the mic sitting next to that speaker
    speaker: object
    guard: object


class ChannelPipeline:
    # one mic, one segmenter, one output

    def __init__(self, *, cfg, channel_id, microphone, segmenter, guard,
                 router, sink, stt, translation, tts, bus, conversation,
                 max_in_flight=3):
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
        self._tasks = set()
        self._stopping = False

    async def run(self):
        # keep eating frames until the mic stops
        try:
            async for frame in self.microphone.frames():
                if self._stopping:
                    break

                if self.guard.observe(frame):
                    # our own speaker is going, dump this audio. also wipe
                    # any half started sentence so the echo cant open one
                    self.segmenter.reset()
                    continue

                for utterance in self.segmenter.push(frame):
                    self._spawn(utterance)
        except asyncio.CancelledError:
            raise
        finally:
            # only flush if we're going to use the result. flush() burns a
            # sequence number and if nobody ever releases that number then
            # everything behind it waits forever
            if not self._stopping:
                trailing = self.segmenter.flush()
                if trailing is not None:
                    self._spawn(trailing)

    def _spawn(self, utterance):
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
        # have to keep a reference or python garbage collects the task
        # while its still running. found that out the hard way
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _process(self, utterance):
        seq = utterance.seq
        timings = {}
        released = False

        try:
            async with self._semaphore:
                transcript = await self._timed(
                    timings, "stt", self._transcribe(utterance)
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

            # let go of the semaphore before playing, so the next sentence
            # can be translating while this one is still talking
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
            # missing api key or something. no point continuing, let this
            # go all the way up so the user gets one clear message
            raise
        except ProviderError as exc:
            self._error(seq, timings, str(exc))
        except Exception as exc:
            # one bad sentence should not end the call
            log.exception("channel %s utterance %d failed", self.channel_id, seq)
            self._error(seq, timings, f"{type(exc).__name__}: {exc}")
        finally:
            # EVERY path has to release or everything behind it hangs
            if not released:
                await self._order.release(seq)

    async def _speak(self, seq, decision, source_text, translated, timings):
        voice = self.cfg.slot_for(decision.target).voice
        started = time.monotonic()
        audio = await self.tts.synthesize(
            translated, language=decision.target, voice=voice
        )

        guard = self.sink.guard
        guard.note_spoken(translated)
        guard.playback_started()

        def _first_audio():
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

        # if the callback never fired (no audio came back) set it here
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

    # ---- the actual stages ----

    async def _transcribe(self, utterance):
        language = self.router.pinned
        return await _retry_once(
            lambda: self.stt.transcribe(
                utterance.pcm,
                language=language,
                candidates=self.cfg.language_codes,
            ),
            what="transcription",
        )

    async def _translate(self, text, decision, context):
        if decision.source == decision.target:
            return text  # nothing to do
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
    async def _timed(timings, key, awaitable):
        # times how long a stage took and saves it in ms
        started = time.monotonic()
        try:
            return await awaitable
        finally:
            timings[key] = (time.monotonic() - started) * 1000.0

    def _drop(self, seq, timings, reason):
        self.bus.emit(
            PipelineEvent(
                seq=seq,
                channel_id=self.channel_id,
                stage=Stage.DROPPED,
                timings=dict(timings),
                detail=reason,
            )
        )

    def _error(self, seq, timings, message):
        self.bus.emit(
            PipelineEvent(
                seq=seq,
                channel_id=self.channel_id,
                stage=Stage.ERROR,
                timings=dict(timings),
                detail=message,
            )
        )

    async def drain(self):
        # wait for whatever is still running to finish
        self._stopping = True
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)


async def _retry_once(factory, *, what, backoff=0.4):
    # wifi hiccups shouldn't cost you a whole sentence so try twice.
    # not catching ConfigError though, a missing api key is never going to
    # fix itself and retrying just prints the same error twice
    try:
        return await factory()
    except ProviderError as exc:
        log.warning("%s failed (%s); retrying once", what, exc)
        await asyncio.sleep(backoff)
        return await factory()
