"""Language table: ISO-639-1 codes, display names, and default voices.

Kept deliberately small and dependency-free so the CLI wizard, the config
validator and the provider layer can all share one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

# OpenAI's gpt-4o-mini-tts voices are all multilingual, so the per-language
# default here is about giving the two sides *distinguishable* voices rather
# than about accent accuracy.
_DEFAULT_VOICES = ("alloy", "nova", "echo", "shimmer", "onyx", "sage")


@dataclass(frozen=True)
class Language:
    code: str  # ISO-639-1
    name: str  # English display name
    native: str  # endonym, shown in the CLI wizard
    voice: str  # default TTS voice


# Not exhaustive — any code Whisper supports will work if you add it here.
LANGUAGES: dict[str, Language] = {
    lang.code: lang
    for lang in (
        Language("en", "English", "English", "alloy"),
        Language("es", "Spanish", "Espanol", "nova"),
        Language("fr", "French", "Francais", "shimmer"),
        Language("de", "German", "Deutsch", "onyx"),
        Language("it", "Italian", "Italiano", "echo"),
        Language("pt", "Portuguese", "Portugues", "sage"),
        Language("nl", "Dutch", "Nederlands", "alloy"),
        Language("pl", "Polish", "Polski", "nova"),
        Language("ru", "Russian", "Russkiy", "onyx"),
        Language("uk", "Ukrainian", "Ukrainska", "shimmer"),
        Language("tr", "Turkish", "Turkce", "echo"),
        Language("ar", "Arabic", "al-Arabiyya", "onyx"),
        Language("he", "Hebrew", "Ivrit", "sage"),
        Language("hi", "Hindi", "Hindi", "nova"),
        Language("ur", "Urdu", "Urdu", "echo"),
        Language("zh", "Chinese", "Zhongwen", "shimmer"),
        Language("ja", "Japanese", "Nihongo", "nova"),
        Language("ko", "Korean", "Hangugeo", "alloy"),
        Language("vi", "Vietnamese", "Tieng Viet", "echo"),
        Language("th", "Thai", "Phasa Thai", "sage"),
        Language("id", "Indonesian", "Bahasa Indonesia", "alloy"),
        Language("sv", "Swedish", "Svenska", "shimmer"),
        Language("no", "Norwegian", "Norsk", "onyx"),
        Language("da", "Danish", "Dansk", "echo"),
        Language("fi", "Finnish", "Suomi", "sage"),
        Language("cs", "Czech", "Cestina", "nova"),
        Language("el", "Greek", "Ellinika", "alloy"),
        Language("ro", "Romanian", "Romana", "shimmer"),
        Language("hu", "Hungarian", "Magyar", "onyx"),
        Language("tl", "Tagalog", "Tagalog", "nova"),
    )
}


def get(code: str) -> Language:
    """Look up a language, tolerating 'en-US' / 'EN' style input."""
    normalized = code.strip().lower().replace("_", "-").split("-")[0]
    if normalized not in LANGUAGES:
        raise KeyError(
            f"Unknown language code {code!r}. Known codes: "
            + ", ".join(sorted(LANGUAGES))
        )
    return LANGUAGES[normalized]


def is_known(code: str) -> bool:
    try:
        get(code)
    except KeyError:
        return False
    return True


def name_of(code: str) -> str:
    """Display name, falling back to the raw code for anything unlisted."""
    try:
        return get(code).name
    except KeyError:
        return code


def default_voice(code: str, taken: set[str] | None = None) -> str:
    """Default voice for a language, avoiding voices already in use.

    Two speakers sharing one voice is confusing in shared-speaker mode, so the
    wizard passes the already-assigned voices in `taken`.
    """
    taken = taken or set()
    try:
        preferred = get(code).voice
    except KeyError:
        preferred = _DEFAULT_VOICES[0]
    if preferred not in taken:
        return preferred
    for candidate in _DEFAULT_VOICES:
        if candidate not in taken:
            return candidate
    return preferred
