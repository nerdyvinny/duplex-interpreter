"""End-to-end orchestrator behaviour with fake providers and fake audio.

No network, no API key, no microphone — but every real module in the path:
segmenter, duplex guard, router, pipeline, orchestrator.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import silence, tone
from interpreter.audio.capture import ArrayMicrophone
from interpreter.audio.playback import RecordingSpeaker
from interpreter.audio.vad import EnergyVad
from interpreter.events import EventBus, Stage
from interpreter.orchestrator import Orchestrator
from interpreter.providers.base import Transcript
from interpreter.providers.fake import FakeSTT, FakeTranslation, FakeTTS


class Harness:
    """Builds an orchestrator whose audio and providers are fully scripted."""

    def __init__(self, cfg, *, audio: dict[str, np.ndarray], stt: FakeSTT):
        self.cfg = cfg
        self.bus = EventBus()
        self.stt = stt
        self.translation = FakeTranslation()
        self.tts = FakeTTS()
        self.speakers: dict[str, RecordingSpeaker] = {}
        self._audio = audio

        self.orchestrator = Orchestrator(
            cfg,
            bus=self.bus,
            providers=(self.stt, self.translation, self.tts),
            microphone_factory=self._microphone,
            speaker_factory=self._speaker,
            vad_factory=lambda _channel: EnergyVad(threshold=0.02),
        )

    def _microphone(self, channel):
        return ArrayMicrophone(
            self._audio.get(channel.id, np.zeros(0, dtype=np.int16)),
            channel_id=channel.id,
        )

    def _speaker(self, channel):
        speaker = RecordingSpeaker(channel_id=channel.id)
        self.speakers[channel.id] = speaker
        return speaker

    async def run(self, timeout: float = 20.0):
        import asyncio

        await self.orchestrator.start()
        try:
            await asyncio.wait_for(self.orchestrator.run_until_stopped(), timeout=timeout)
        finally:
            await self.orchestrator.shutdown()
        return self.bus.history

    def events(self, stage: Stage):
        return [e for e in self.bus.history if e.stage is stage]


def one_utterance() -> np.ndarray:
    return np.concatenate([silence(0.2), tone(0.8), silence(0.8)])


def two_utterances() -> np.ndarray:
    return np.concatenate(
        [silence(0.2), tone(0.8), silence(0.9), tone(0.8), silence(0.9)]
    )


async def test_single_mic_translates_english_to_spanish(single_mic_config):
    harness = Harness(
        single_mic_config,
        audio={"A": one_utterance()},
        stt=FakeSTT([Transcript(text="hello how are you")]),
    )
    await harness.run()

    done = harness.events(Stage.DONE)
    assert len(done) == 1
    assert done[0].source_lang == "en"
    assert done[0].target_lang == "es"
    assert done[0].target_text == "[en->es] hello how are you"

    # The translation was actually handed to the speaker.
    assert harness.speakers["A"].pcm().size > 0


async def test_single_mic_routes_spanish_back_to_english(single_mic_config):
    harness = Harness(
        single_mic_config,
        audio={"A": one_utterance()},
        stt=FakeSTT([Transcript(text="hola como estas muy bien gracias")]),
    )
    await harness.run()

    done = harness.events(Stage.DONE)
    assert len(done) == 1
    assert done[0].source_lang == "es"
    assert done[0].target_lang == "en"


async def test_both_directions_in_one_conversation(single_mic_config):
    harness = Harness(
        single_mic_config,
        audio={"A": two_utterances()},
        stt=FakeSTT(
            [
                Transcript(text="hello what is your name"),
                Transcript(text="hola me llamo Ana gracias"),
            ]
        ),
    )
    await harness.run()

    done = harness.events(Stage.DONE)
    assert len(done) == 2
    assert [(e.source_lang, e.target_lang) for e in done] == [("en", "es"), ("es", "en")]


async def test_dual_mic_cross_wires_output_to_the_other_headset(dual_mic_config):
    """A speaks into mic A; the Spanish comes out of B's earpiece, not A's."""
    harness = Harness(
        dual_mic_config,
        audio={"A": one_utterance()},
        stt=FakeSTT([Transcript(text="hello there")]),
    )
    await harness.run()

    assert harness.speakers["B"].pcm().size > 0
    assert harness.speakers["A"].pcm().size == 0

    done = harness.events(Stage.DONE)
    assert len(done) == 1
    assert done[0].source_lang == "en"
    assert done[0].target_lang == "es"


async def test_dual_mic_pins_language_and_skips_detection(dual_mic_config):
    """Pinned channels must not run language ID at all — that's the point."""
    harness = Harness(
        dual_mic_config,
        audio={"A": one_utterance()},
        # Ambiguous text that text-LID could plausibly call either way.
        stt=FakeSTT([Transcript(text="no")]),
    )
    await harness.run()

    done = harness.events(Stage.DONE)
    assert len(done) == 1
    assert done[0].source_lang == "en"  # from the pin, not from the word "no"
    assert harness.stt.calls[0]["language"] == "en"  # pinned hint passed to STT


async def test_both_channels_run_concurrently(dual_mic_config):
    """The whole promise: two people talking at once both get translated."""
    harness = Harness(
        dual_mic_config,
        audio={"A": one_utterance(), "B": one_utterance()},
        stt=FakeSTT(
            [Transcript(text="hello"), Transcript(text="hola")],
            delay=0.05,
        ),
    )
    await harness.run()

    done = harness.events(Stage.DONE)
    assert len(done) == 2
    assert {e.channel_id for e in done} == {"A", "B"}
    # Each side's translation landed in the *other* person's earpiece.
    assert harness.speakers["A"].pcm().size > 0
    assert harness.speakers["B"].pcm().size > 0


async def test_playback_stays_in_order_when_stages_finish_out_of_order(single_mic_config):
    """A fast short sentence must not overtake a slow earlier one."""
    harness = Harness(
        single_mic_config,
        audio={"A": two_utterances()},
        stt=FakeSTT(
            [
                Transcript(text="the first thing I said"),
                Transcript(text="the second thing I said"),
            ],
            delays=[0.35, 0.0],  # utterance 2 finishes STT long before 1
        ),
    )
    await harness.run()

    spoken = [call["text"] for call in harness.tts.calls]
    assert spoken == [
        "[en->es] the first thing I said",
        "[en->es] the second thing I said",
    ]


async def test_empty_transcript_is_dropped(single_mic_config):
    harness = Harness(
        single_mic_config,
        audio={"A": one_utterance()},
        stt=FakeSTT([Transcript(text="   ")]),
    )
    await harness.run()

    assert harness.events(Stage.DONE) == []
    dropped = harness.events(Stage.DROPPED)
    assert len(dropped) == 1
    assert "no speech" in dropped[0].detail


async def test_a_hallucinated_third_language_is_dropped(single_mic_config):
    """From a live session: room noise came back from Whisper as Cyrillic.

    It must be dropped, not routed by alternation into a "translation" of
    noise into itself.
    """
    harness = Harness(
        single_mic_config,
        audio={"A": two_utterances()},
        stt=FakeSTT(
            [Transcript(text="Диана"), Transcript(text="where is the station")]
        ),
    )
    await harness.run()

    dropped = harness.events(Stage.DROPPED)
    assert len(dropped) == 1
    assert "neither" in dropped[0].detail

    # The real utterance behind it still goes through.
    done = harness.events(Stage.DONE)
    assert len(done) == 1
    assert done[0].source_text == "where is the station"
    assert harness.tts.calls == [
        {"text": "[en->es] where is the station", "language": "es", "voice": "nova"}
    ]


async def test_self_echo_is_dropped(single_mic_config):
    """The microphone hearing our own translation must not start a loop."""
    harness = Harness(
        single_mic_config,
        audio={"A": two_utterances()},
        stt=FakeSTT(
            [
                Transcript(text="hello there"),
                # What the speaker said, fed back in by the room.
                Transcript(text="[en->es] hello there"),
            ]
        ),
    )
    await harness.run()

    dropped = harness.events(Stage.DROPPED)
    assert any("self-echo" in (e.detail or "") for e in dropped)
    assert len(harness.events(Stage.DONE)) == 1


async def test_dual_mic_does_not_apply_the_echo_guard(dual_mic_config):
    """With headsets, repeating yourself is legitimate, not an echo."""
    harness = Harness(
        dual_mic_config,
        audio={"A": two_utterances()},
        stt=FakeSTT(
            [Transcript(text="say that again"), Transcript(text="say that again")]
        ),
    )
    await harness.run()

    assert len(harness.events(Stage.DONE)) == 2
    assert harness.events(Stage.DROPPED) == []


async def test_a_failing_translation_does_not_block_later_utterances(single_mic_config):
    """One bad turn must not deadlock the ordered playback queue behind it."""
    harness = Harness(
        single_mic_config,
        audio={"A": two_utterances()},
        stt=FakeSTT(
            [Transcript(text="this one explodes"), Transcript(text="this one is fine")]
        ),
    )
    harness.translation.fail_on = {"this one explodes"}
    await harness.run()

    errors = harness.events(Stage.ERROR)
    done = harness.events(Stage.DONE)
    assert len(errors) == 1
    assert len(done) == 1
    assert done[0].source_text == "this one is fine"


async def test_translation_receives_conversation_context(single_mic_config):
    harness = Harness(
        single_mic_config,
        audio={"A": two_utterances()},
        stt=FakeSTT(
            [Transcript(text="where is the station"), Transcript(text="esta muy cerca de aqui")]
        ),
    )
    await harness.run()

    assert harness.translation.calls[0]["context"] == []
    # The second turn sees the first, so pronouns resolve.
    assert "where is the station" in " ".join(harness.translation.calls[1]["context"])


async def test_timings_are_recorded_for_each_stage(single_mic_config):
    harness = Harness(
        single_mic_config,
        audio={"A": one_utterance()},
        stt=FakeSTT([Transcript(text="hello")], delay=0.02),
    )
    harness.translation.delay = 0.02
    await harness.run()

    done = harness.events(Stage.DONE)[0]
    assert set(done.timings) == {"stt", "translate", "tts"}
    assert done.timings["stt"] >= 15.0
    assert done.total_ms > 0


async def test_silence_produces_no_turns(single_mic_config):
    harness = Harness(
        single_mic_config,
        audio={"A": silence(2.0)},
        stt=FakeSTT([Transcript(text="should never be reached")]),
    )
    await harness.run()

    assert harness.bus.history == []
    assert harness.stt.calls == []


async def test_stt_is_told_the_candidate_languages(single_mic_config):
    harness = Harness(
        single_mic_config,
        audio={"A": one_utterance()},
        stt=FakeSTT([Transcript(text="hello")]),
    )
    await harness.run()

    assert harness.stt.calls[0]["candidates"] == ("en", "es")
    assert harness.stt.calls[0]["language"] is None  # auto-detect in single_mic


async def test_tts_uses_the_target_language_voice(single_mic_config):
    harness = Harness(
        single_mic_config,
        audio={"A": one_utterance()},
        stt=FakeSTT([Transcript(text="hello there friend")]),
    )
    await harness.run()

    assert harness.tts.calls[0]["language"] == "es"
    assert harness.tts.calls[0]["voice"] == "nova"  # language B's voice
