# guesses which of the two languages a piece of text is
#
# whisper can tell you what language the AUDIO was but its bad at it when
# the sentence is short, and conversations are full of short sentences like
# "no", "ok", "si". once I have the text though, picking between 2 languages
# I already know is a much easier problem, plus its instant and offline.
#
# two steps:
#   1. look at the alphabet. if its english vs russian this is basically
#      always right and I can stop here
#   2. count common words and accented letters. this is for when both
#      languages use the same alphabet like english vs spanish

import re
import unicodedata
from dataclasses import dataclass

# alphabets that give the language away immediately.
# these are unicode ranges, I got them off the unicode charts
_SCRIPT_RANGES = [
    ("cyrillic", (0x0400, 0x04FF)),
    ("greek", (0x0370, 0x03FF)),
    ("hebrew", (0x0590, 0x05FF)),
    ("arabic", (0x0600, 0x06FF)),
    ("devanagari", (0x0900, 0x097F)),
    ("thai", (0x0E00, 0x0E7F)),
    ("hangul", (0xAC00, 0xD7AF)),
    ("kana", (0x3040, 0x30FF)),
    ("han", (0x4E00, 0x9FFF)),
]

_LANG_SCRIPTS = {
    "ru": {"cyrillic"},
    "uk": {"cyrillic"},
    "el": {"greek"},
    "he": {"hebrew"},
    "ar": {"arabic"},
    "ur": {"arabic"},
    "hi": {"devanagari"},
    "th": {"thai"},
    "ko": {"hangul"},
    "ja": {"kana", "han"},  # japanese uses both
    "zh": {"han"},
}

# the most common little words in each language. I kept these short on
# purpose, these are the words that actually show up when somebody says
# one sentence out loud
_STOPWORDS = {
    "en": {
        "the", "and", "is", "are", "you", "i", "to", "of", "it", "that", "this",
        "what", "how", "not", "do", "does", "did", "have", "has", "was", "were",
        "yes", "no", "please", "thanks", "thank", "hello", "hi", "we", "they",
        "my", "your", "can", "will", "would", "there", "here", "with", "for",
    },
    "es": {
        "el", "la", "los", "las", "de", "que", "y", "es", "en", "un", "una",
        "por", "con", "no", "si", "para", "como", "esta", "este", "muy", "pero",
        "hola", "gracias", "buenos", "buenas", "donde", "cuando", "porque",
        "yo", "tu", "usted", "nosotros", "bien", "mas", "tambien", "ser",
    },
    "fr": {
        "le", "la", "les", "de", "des", "et", "est", "un", "une", "que", "qui",
        "pour", "dans", "pas", "vous", "je", "tu", "il", "elle", "nous", "ce",
        "bonjour", "merci", "oui", "non", "avec", "sur", "mais", "tres", "bien",
        "comment", "pourquoi", "ou", "quand", "faire", "etre", "avoir",
    },
    "de": {
        "der", "die", "das", "und", "ist", "ein", "eine", "nicht", "ich", "du",
        "sie", "wir", "zu", "mit", "auf", "fur", "von", "es", "auch", "aber",
        "hallo", "danke", "ja", "nein", "bitte", "was", "wie", "wo", "wann",
        "warum", "haben", "sein", "werden", "sehr", "gut", "noch", "schon",
    },
    "it": {
        "il", "la", "le", "di", "che", "e", "un", "una", "per", "con", "non",
        "sono", "questo", "questa", "come", "dove", "quando", "perche", "ciao",
        "grazie", "si", "no", "molto", "bene", "anche", "essere", "avere",
    },
    "pt": {
        "o", "a", "os", "as", "de", "que", "e", "um", "uma", "para", "com",
        "nao", "sim", "isso", "este", "esta", "como", "onde", "quando",
        "porque", "ola", "obrigado", "obrigada", "muito", "bem", "tambem",
    },
    "nl": {
        "de", "het", "een", "en", "is", "van", "ik", "je", "niet", "dat", "op",
        "met", "voor", "hallo", "dank", "ja", "nee", "hoe", "wat", "waar",
        "wanneer", "waarom", "zijn", "hebben", "heel", "goed", "ook",
    },
    "pl": {
        "i", "w", "na", "nie", "to", "jest", "sie", "z", "do", "ze", "co",
        "jak", "gdzie", "kiedy", "dlaczego", "tak", "dziekuje", "czesc",
        "bardzo", "dobrze", "byc", "miec", "ale", "juz", "tylko",
    },
    "tr": {
        "bir", "ve", "bu", "ne", "icin", "ile", "var", "yok", "evet", "hayir",
        "merhaba", "tesekkur", "nasil", "nerede", "ne zaman", "neden", "cok",
        "iyi", "ama", "da", "de", "olarak", "benim", "senin",
    },
    "id": {
        "yang", "di", "dan", "ini", "itu", "dengan", "untuk", "tidak", "ada",
        "saya", "kamu", "apa", "dimana", "kapan", "kenapa", "ya", "terima",
        "kasih", "halo", "sangat", "baik", "juga", "bisa",
    },
    "sv": {
        "och", "att", "det", "en", "ett", "som", "ar", "jag", "du", "inte",
        "med", "for", "hej", "tack", "ja", "nej", "hur", "vad", "var", "nar",
        "varfor", "mycket", "bra", "ocksa", "har", "vara",
    },
    "vi": {
        "va", "la", "cua", "co", "khong", "toi", "ban", "nay", "duoc", "cho",
        "xin", "chao", "cam", "on", "vang", "gi", "dau", "khi", "nao", "rat",
        "tot", "cung", "the",
    },
    "tl": {
        "ang", "ng", "sa", "na", "ay", "at", "mga", "ko", "mo", "hindi", "oo",
        "salamat", "kumusta", "ano", "saan", "kailan", "bakit", "po", "ito",
        "iyan", "may", "para", "din", "rin",
    },
}

# letters you basically only see in one language of a pair
_MARKER_CHARS = {
    "es": "ñ¿¡áéíóúü",
    "fr": "àâçéèêëîïôûùüÿœ",
    "de": "äöüß",
    "pt": "ãõáâêéíóôúç",
    "it": "àèéìòù",
    "pl": "ąćęłńóśźż",
    "tr": "çğıöşü",
    "nl": "ëïĳ",
    "sv": "åäö",
    "vi": "ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệ",
}

# grabs words, no numbers or underscores
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class LangGuess:
    language: str
    confidence: float  # 0 to 1
    reason: str
    # foreign=True means its definitely NEITHER language, which is different
    # from language=None which just means I couldn't tell. the caller should
    # throw the foreign ones away, they're almost always whisper making
    # stuff up about background noise
    foreign: bool = False


def _strip_accents(text):
    # café -> cafe
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _script_profile(text):
    # count how many letters of each alphabet are in here
    counts = {}
    for char in text:
        code = ord(char)
        for script, (low, high) in _SCRIPT_RANGES:
            if low <= code <= high:
                counts[script] = counts.get(script, 0) + 1
                break
        else:
            # the for/else runs when the loop didn't break, so this is a
            # letter that wasn't in any of my ranges
            if char.isalpha() and code < 0x0250:
                counts["latin"] = counts.get("latin", 0) + 1
    return counts


def identify(text, candidates):
    if len(candidates) != 2:
        raise ValueError(f"expected exactly 2 candidates, got {candidates}")

    stripped = text.strip()
    if not stripped:
        return LangGuess(None, 0.0, "empty text")

    first, second = candidates

    # ---- step 1, the alphabet ----
    scripts = _script_profile(stripped)
    if scripts:
        dominant = max(scripts, key=lambda s: scripts[s])
        total = sum(scripts.values())
        share = scripts[dominant] / total

        first_scripts = _LANG_SCRIPTS.get(first, {"latin"})
        second_scripts = _LANG_SCRIPTS.get(second, {"latin"})

        # only useful if the two languages use different alphabets
        if first_scripts != second_scripts and share >= 0.6:
            if dominant in first_scripts and dominant not in second_scripts:
                return LangGuess(first, 0.98, f"{dominant} script")
            if dominant in second_scripts and dominant not in first_scripts:
                return LangGuess(second, 0.98, f"{dominant} script")

        # its written in an alphabet NEITHER person uses. whisper does this
        # when its hallucinating on noise, I had my fan come back as russian
        # once. guessing a direction for this is worse than admitting its
        # not something we were asked to handle
        if share >= 0.6 and dominant not in first_scripts | second_scripts:
            return LangGuess(
                None,
                0.0,
                f"{dominant} script is neither {first} nor {second}",
                foreign=True,
            )

    # ---- step 2, common words + accents ----
    words = [w.lower() for w in _WORD_RE.findall(stripped)]
    if not words:
        return LangGuess(None, 0.0, "no alphabetic words")

    folded = [_strip_accents(w) for w in words]
    lowered = stripped.lower()

    scores = {}
    for lang in candidates:
        stopwords = _STOPWORDS.get(lang, set())
        hits = sum(1 for w in folded if w in stopwords)
        score = hits / len(folded)

        markers = _MARKER_CHARS.get(lang, "")
        if markers:
            marker_hits = sum(1 for c in lowered if c in markers)
            # accents are strong evidence but I cap it, otherwise one ñ
            # beats a whole sentence of english words
            score += min(0.5, marker_hits * 0.25)

        scores[lang] = score

    if scores[first] >= scores[second]:
        best, worst = first, second
    else:
        best, worst = second, first

    margin = scores[best] - scores[worst]
    if scores[best] <= 0.0:
        return LangGuess(None, 0.0, "no distinguishing words")

    # turn the margin into a confidence number. longer sentences get a
    # small bonus because theres more evidence
    confidence = min(0.95, margin * 2.5 + min(len(folded), 8) / 40.0)
    return LangGuess(
        best, confidence, f"lexical margin {margin:.2f} over {len(folded)} words"
    )


def normalize_language_name(value):
    # the speech apis are inconsistent about this. whisper-1 gives you
    # "english", the newer models give "en", and sometimes nothing at all
    if not value:
        return None

    cleaned = value.strip().lower().replace("_", "-")
    if not cleaned:
        return None

    # already a code like "en" or "en-US"
    if len(cleaned) <= 3 or "-" in cleaned:
        return cleaned.split("-")[0]

    # imported here to avoid a circular import, languages doesn't need me
    # but config imports both
    from . import languages

    for code, lang in languages.LANGUAGES.items():
        if cleaned == lang.name.lower():
            return code

    # names that don't match my table
    extra = {
        "mandarin": "zh",
        "chinese": "zh",
        "castilian": "es",
        "flemish": "nl",
        "farsi": "fa",
        "brazilian portuguese": "pt",
    }
    return extra.get(cleaned)
