"""Language identification and routing.

Single-mic mode lives or dies on this: if the app guesses wrong about who
spoke, it translates into the language the speaker was already using.
"""

from __future__ import annotations

import pytest

from interpreter import langid
from interpreter.providers.base import Transcript
from interpreter.routing import LanguageRouter


# --------------------------------------------------------------------------
# text language ID
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello how are you doing today", "en"),
        ("what time does the train leave", "en"),
        ("hola como estas amigo", "es"),
        ("donde esta la estacion de tren por favor", "es"),
        ("muchas gracias por su ayuda", "es"),
        ("I would like a coffee please", "en"),
    ],
)
def test_english_spanish_are_separated(text, expected):
    guess = langid.identify(text, ("en", "es"))
    assert guess.language == expected
    assert guess.confidence > 0.3


def test_accented_characters_are_strong_evidence():
    guess = langid.identify("mañana", ("en", "es"))
    assert guess.language == "es"


@pytest.mark.parametrize(
    ("text", "candidates", "expected"),
    [
        ("привет как дела", ("en", "ru"), "ru"),
        ("hello there", ("en", "ru"), "en"),
        ("こんにちは元気ですか", ("en", "ja"), "ja"),
        ("مرحبا كيف حالك", ("en", "ar"), "ar"),
        ("안녕하세요", ("en", "ko"), "ko"),
        ("γεια σου", ("en", "el"), "el"),
    ],
)
def test_different_scripts_are_decided_on_sight(text, candidates, expected):
    guess = langid.identify(text, candidates)
    assert guess.language == expected
    assert guess.confidence > 0.9


def test_ambiguous_short_words_report_low_confidence():
    """"no" is a real word in both. The router must be told it's a guess."""
    guess = langid.identify("no", ("en", "es"))
    assert guess.confidence < 0.45


def test_empty_text_yields_nothing():
    assert langid.identify("", ("en", "es")).language is None
    assert langid.identify("   ", ("en", "es")).language is None


def test_two_candidates_are_required():
    with pytest.raises(ValueError):
        langid.identify("hello", ("en",))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("english", "en"),
        ("English", "en"),
        ("spanish", "es"),
        ("en", "en"),
        ("en-US", "en"),
        ("chinese", "zh"),
        (None, None),
        ("", None),
    ],
)
def test_stt_language_names_normalize_to_codes(raw, expected):
    assert langid.normalize_language_name(raw) == expected


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


def test_pinned_channel_never_guesses():
    router = LanguageRouter(("en", "es"), pinned="en")
    decision = router.route(Transcript(text="no", language="es", language_confidence=0.99))

    assert decision.source == "en"
    assert decision.target == "es"
    assert decision.confidence == 1.0


def test_clear_text_decides_the_direction():
    router = LanguageRouter(("en", "es"))
    decision = router.route(Transcript(text="donde esta el bano por favor"))

    assert decision.source == "es"
    assert decision.target == "en"


def test_agreeing_signals_raise_confidence():
    router = LanguageRouter(("en", "es"))
    agreeing = router.route(
        Transcript(text="hola como estas", language="es", language_confidence=0.9)
    )
    text_only = LanguageRouter(("en", "es")).route(Transcript(text="hola como estas"))

    assert agreeing.source == "es"
    assert agreeing.confidence > text_only.confidence


def test_text_wins_when_it_disagrees_with_audio():
    """Whisper mislabels short clips constantly; the transcript is better."""
    router = LanguageRouter(("en", "es"))
    decision = router.route(
        Transcript(
            text="where is the train station please",
            language="es",
            language_confidence=0.8,
        )
    )

    assert decision.source == "en"
    assert "trusting text" in decision.reason


def test_confident_audio_wins_when_the_text_is_uninformative():
    router = LanguageRouter(("en", "es"))
    decision = router.route(
        Transcript(text="mmm", language="es", language_confidence=0.95)
    )

    assert decision.source == "es"


def test_alternation_takes_over_when_nothing_is_usable():
    router = LanguageRouter(("en", "es"))

    first = router.route(Transcript(text="I would like the menu please"))
    assert first.source == "en"

    # Unintelligible: no text signal, no audio signal.
    second = router.route(Transcript(text="mmhm"))
    assert second.source == "es"  # assume the other person is speaking
    assert second.target == "en"
    assert second.confidence < 0.6


def test_alternation_defaults_to_language_a_at_the_start():
    router = LanguageRouter(("en", "es"))
    decision = router.route(Transcript(text="mmhm"))

    assert decision.source == "en"
    assert "defaulting" in decision.reason


def test_source_and_target_are_always_different():
    router = LanguageRouter(("en", "es"))
    for text in ["hello", "hola", "mmm", "gracias amigo", "thank you very much"]:
        decision = router.route(Transcript(text=text))
        assert decision.source != decision.target
        assert {decision.source, decision.target} == {"en", "es"}


def test_out_of_pair_audio_detection_is_ignored():
    """Whisper sometimes returns a third language; it isn't an option here."""
    router = LanguageRouter(("en", "es"))
    decision = router.route(
        Transcript(text="hola buenos dias", language="pt", language_confidence=0.9)
    )

    assert decision.source == "es"


def test_identical_candidates_are_rejected():
    with pytest.raises(ValueError):
        LanguageRouter(("en", "en"))


# --------------------------------------------------------------------------
# rejecting hallucinations
#
# All of these come from a real live session: the VAD opened on room noise
# and Whisper turned it into words.
# --------------------------------------------------------------------------


def test_text_in_a_third_script_is_flagged_foreign():
    guess = langid.identify("Диана", ("en", "es"))

    assert guess.foreign is True
    assert guess.language is None


def test_a_third_script_is_dropped_rather_than_guessed():
    """The exact failure from live testing: noise came back as Cyrillic."""
    router = LanguageRouter(("en", "es"))
    decision = router.route(Transcript(text="Диана"))

    assert decision.reject is True
    assert "neither" in decision.reason or "not en or es" in decision.reason


def test_a_rejection_does_not_disturb_the_alternation_state():
    """A hallucination is not a turn, so it must not flip whose go it is."""
    router = LanguageRouter(("en", "es"))

    router.route(Transcript(text="I would like the menu please"))  # en
    assert router.route(Transcript(text="Диана")).reject is True

    # Next unclear utterance should still alternate off the English turn.
    following = router.route(Transcript(text="mmhm"))
    assert following.reject is False
    assert following.source == "es"


def test_an_out_of_pair_recognizer_language_with_no_text_signal_is_dropped():
    router = LanguageRouter(("en", "es"))
    decision = router.route(Transcript(text="mm", language="cy"))

    assert decision.reject is True
    assert "cy" in decision.reason


def test_an_out_of_pair_language_is_kept_when_the_text_is_clear():
    """Whisper mislabels constantly; a readable transcript overrules it."""
    router = LanguageRouter(("en", "es"))
    decision = router.route(Transcript(text="donde esta la estacion", language="pt"))

    assert decision.reject is False
    assert decision.source == "es"


def test_ordinary_ambiguous_speech_is_still_routed_not_dropped():
    """Rejection is for foreign text, not for merely unclear text."""
    router = LanguageRouter(("en", "es"))

    for text in ("no", "ok", "mmhm", "yeah"):
        decision = router.route(Transcript(text=text))
        assert decision.reject is False, f"{text!r} should route, not drop"


def test_a_matching_third_script_is_not_foreign():
    """Cyrillic is only foreign if neither speaker uses it."""
    guess = langid.identify("привет как дела", ("en", "ru"))

    assert guess.foreign is False
    assert guess.language == "ru"
