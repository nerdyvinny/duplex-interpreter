# all the languages the app knows about.
# I put them in one place so the setup wizard and the config checker
# can both use the same list instead of me typing it twice

from dataclasses import dataclass

# the openai voices. they all speak every language so it doesn't really
# matter which one goes with which language, I just wanted the two people
# to sound different from each other
_DEFAULT_VOICES = ("alloy", "nova", "echo", "shimmer", "onyx", "sage")


@dataclass(frozen=True)
class Language:
    code: str      # like "en"
    name: str      # like "English"
    native: str    # what they call it themselves
    voice: str     # which voice to use


# this isn't every language, just the ones I bothered to add.
# whisper knows way more, you can just add a line here if you need one
LANGUAGES = {
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


def get(code):
    # people write "en-US" or "EN" so chop off the extra stuff first
    normalized = code.strip().lower().replace("_", "-").split("-")[0]
    if normalized not in LANGUAGES:
        raise KeyError(
            f"Unknown language code {code!r}. Known codes: "
            + ", ".join(sorted(LANGUAGES))
        )
    return LANGUAGES[normalized]


def is_known(code):
    try:
        get(code)
    except KeyError:
        return False
    return True


def name_of(code):
    try:
        return get(code).name
    except KeyError:
        return code  # just show the code if I never added that language


def default_voice(code, taken=None):
    # "taken" is the voices already used. if both people get the same voice
    # you can't tell who is talking, which I found out the annoying way
    if taken is None:
        taken = set()
    try:
        preferred = get(code).voice
    except KeyError:
        preferred = _DEFAULT_VOICES[0]

    if preferred not in taken:
        return preferred

    # already used, grab any free one
    for candidate in _DEFAULT_VOICES:
        if candidate not in taken:
            return candidate
    return preferred  # ran out, whatever
