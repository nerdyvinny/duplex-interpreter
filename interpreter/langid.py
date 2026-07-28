"""Text-based language identification between exactly two candidates.

Whisper's *audio* language ID is unreliable on the short utterances a
conversation is made of ("no", "ok", "si"). Once we have the transcript,
deciding between two known candidates is a much easier problem, and doing it
on text is both instant and offline.

Two stages:
  1. Script detection — decisive when the candidates use different alphabets
     (en/ru, en/ar, en/zh, ...). This is essentially always correct.
  2. Stopword and character scoring — for same-script pairs (en/es, fr/de).
     Returns a confidence the caller can threshold, falling back to the
     alternation heuristic when it is too low to trust.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Scripts that identify a language essentially on sight.
_SCRIPT_RANGES: list[tuple[str, tuple[int, int]]] = [
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

_LANG_SCRIPTS: dict[str, set[str]] = {
    "ru": {"cyrillic"},
    "uk": {"cyrillic"},
    "el": {"greek"},
    "he": {"hebrew"},
    "ar": {"arabic"},
    "ur": {"arabic"},
    "hi": {"devanagari"},
    "th": {"thai"},
    "ko": {"hangul"},
    "ja": {"kana", "han"},
    "zh": {"han"},
}

# High-frequency function words. Deliberately short lists: these are the words
# that actually show up in one-sentence conversational turns.
_STOPWORDS: dict[str, set[str]] = {
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

# Characters that only really appear in one of a pair of same-script languages.
_MARKER_CHARS: dict[str, str] = {
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

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class LangGuess:
    language: str | None
    confidence: float  # 0..1
    reason: str


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _script_profile(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        code = ord(char)
        for script, (low, high) in _SCRIPT_RANGES:
            if low <= code <= high:
                counts[script] = counts.get(script, 0) + 1
                break
        else:
            if char.isalpha() and code < 0x0250:
                counts["latin"] = counts.get("latin", 0) + 1
    return counts


def identify(text: str, candidates: tuple[str, ...]) -> LangGuess:
    """Pick which of `candidates` `text` is written in."""
    if len(candidates) != 2:
        raise ValueError(f"expected exactly 2 candidates, got {candidates}")
    stripped = text.strip()
    if not stripped:
        return LangGuess(None, 0.0, "empty text")

    first, second = candidates

    # --- stage 1: script ---
    scripts = _script_profile(stripped)
    if scripts:
        dominant = max(scripts, key=lambda s: scripts[s])
        total = sum(scripts.values())
        share = scripts[dominant] / total
        first_scripts = _LANG_SCRIPTS.get(first, {"latin"})
        second_scripts = _LANG_SCRIPTS.get(second, {"latin"})
        if first_scripts != second_scripts and share >= 0.6:
            if dominant in first_scripts and dominant not in second_scripts:
                return LangGuess(first, 0.98, f"{dominant} script")
            if dominant in second_scripts and dominant not in first_scripts:
                return LangGuess(second, 0.98, f"{dominant} script")

    # --- stage 2: stopwords + marker characters ---
    words = [w.lower() for w in _WORD_RE.findall(stripped)]
    if not words:
        return LangGuess(None, 0.0, "no alphabetic words")
    folded = [_strip_accents(w) for w in words]
    lowered = stripped.lower()

    scores: dict[str, float] = {}
    for lang in candidates:
        stopwords = _STOPWORDS.get(lang, set())
        hits = sum(1 for w in folded if w in stopwords)
        score = hits / len(folded)
        markers = _MARKER_CHARS.get(lang, "")
        if markers:
            marker_hits = sum(1 for c in lowered if c in markers)
            # Accented characters are strong evidence but shouldn't swamp a
            # long sentence's worth of stopword evidence.
            score += min(0.5, marker_hits * 0.25)
        scores[lang] = score

    best, worst = (first, second) if scores[first] >= scores[second] else (second, first)
    margin = scores[best] - scores[worst]
    if scores[best] <= 0.0:
        return LangGuess(None, 0.0, "no distinguishing words")

    # Map the margin onto a confidence. A single stopword in a three-word
    # sentence should not read as certainty.
    confidence = min(0.95, margin * 2.5 + min(len(folded), 8) / 40.0)
    return LangGuess(best, confidence, f"lexical margin {margin:.2f} over {len(folded)} words")


def normalize_language_name(value: str | None) -> str | None:
    """Map whatever an STT API returned onto an ISO-639-1 code.

    `whisper-1` returns full English names ("english"); newer models return
    codes, or nothing at all.
    """
    if not value:
        return None
    cleaned = value.strip().lower().replace("_", "-")
    if not cleaned:
        return None
    if len(cleaned) <= 3 or "-" in cleaned:
        return cleaned.split("-")[0]

    from . import languages

    for code, lang in languages.LANGUAGES.items():
        if cleaned == lang.name.lower():
            return code
    extra = {
        "mandarin": "zh",
        "chinese": "zh",
        "castilian": "es",
        "flemish": "nl",
        "farsi": "fa",
        "brazilian portuguese": "pt",
    }
    return extra.get(cleaned)
