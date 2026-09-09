from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


class WordEquivalence:
    """
    Two-way lookup of word/phrase equivalence groups, per language.

    Equivalence groups declare variants (single words or multi-word phrases)
    that should be treated as matches during word-error validation,
    eg "all right" <-> "alright".

    This supplements (and is distinct from) the Double Metaphone homophone
    check in TextNormalizer.sounds_the_same_en, which is single-word-only
    and cannot relate a phrase to a single word.

    Entries must be lowercase, single-spaced, and normalized in the same
    spirit as TextNormalizer.normalize_common() output (no punctuation).
    """

    _EQUIVALENCE_GROUPS: dict[str, list[tuple[str, ...]]] = {

        "en": [

            # Note slightly liberal application of equivalencies here like "got to" <--> "gotta"

            # Split/merged forms whose letters differ, so
            # TextNormalizer.normalize_spacing_en cannot align them
            ("all right", "alright"),
            ("all together", "altogether"),
            ("all ready", "already"),
            ("where ever", "wherever"),

            # Irregular contractions: normalization strips apostrophes,
            # so "don't" -> "dont" but "do not" -> "do not" (no alignment)

            ("do not", "dont"),
            ("will not", "wont"),
            ("can not", "cannot", "cant"),
            ("it is", "its"),
            ("you are", "youre"),
            ("they are", "theyre"),
            ("i am", "im"),

            # Informal reductions (common in dialogue-heavy source text)
            ("got to", "gotta"),
            ("out of", "outta"),
            ("let me", "lemme"),
            ("give me", "gimme"),

            # Spelling variants Double Metaphone does not unify
            ("judgment", "judgement"),
            ("acknowledgment", "acknowledgement"),
        ],
    }

    # Derived two-way lookup: variant (word or phrase) -> its group.
    # Cached mappings are read-only so callers cannot corrupt process-global data.
    _LOOKUP: dict[str, Mapping[str, frozenset[str]]] = {}

    # Common full names and ISO 639-2 codes -> primary 2-letter code.
    # Keeps equivalence (and whitelist parity) from being silently disabled by a
    # region/script/name-qualified project language code.
    _LANGUAGE_ALIASES: dict[str, str] = {
        "english": "en",
        "eng": "en",
        "spanish": "es",
        "spa": "es",
    }

    @staticmethod
    def normalize_language_code(language_code: str) -> str:
        """Returns the base, lowercase language code (eg ``en-US`` -> ``en``).

        Accepts BCP-47 region/script variants (``en-US``, ``en_GB``), ISO 639-2
        three-letter codes (``eng``), and common full names (``English``), so a
        region- or name-qualified project code does not silently disable the
        equivalence feature.
        """
        code = language_code.strip().lower()
        if not code:
            return ""
        base = code.replace("_", "-").split("-", 1)[0]
        return WordEquivalence._LANGUAGE_ALIASES.get(base, base)

    @classmethod
    def supports_language(cls, language_code: str) -> bool:
        return cls.normalize_language_code(language_code) in cls._EQUIVALENCE_GROUPS

    @classmethod
    def _get_lookup(cls, language_code: str) -> Mapping[str, frozenset[str]]:
        """
        Lazily builds and caches the read-only variant -> group index for a language.
        """
        code = cls.normalize_language_code(language_code)
        lookup = cls._LOOKUP.get(code)
        if lookup is None:
            mutable_lookup: dict[str, frozenset[str]] = {}
            for group in cls._EQUIVALENCE_GROUPS.get(code, []):
                frozen = frozenset(group)
                for member in group:
                    mutable_lookup[member] = frozen
            lookup = MappingProxyType(mutable_lookup)
            cls._LOOKUP[code] = lookup
        return lookup

    @classmethod
    def is_equivalent(cls, source_phrase: str, transcript_phrase: str, language_code: str) -> bool:
        """
        Returns True if the two words/phrases belong to the same equivalence
        group for the given language.

        Phrases are compared case-insensitively with whitespace collapsed,
        since callers may pass raw word-list joins.
        """
        if not cls.supports_language(language_code):
            return False

        a = " ".join(source_phrase.casefold().split())
        b = " ".join(transcript_phrase.casefold().split())
        if not a or not b:
            return False

        lookup = cls._get_lookup(language_code)
        return a in lookup and b in lookup and lookup[a] is lookup[b]

    @classmethod
    def get_lookup(cls, language_code: str) -> Mapping[str, frozenset[str]]:
        """
        Returns the read-only variant -> group index for a language (empty
        mapping if unsupported). Exposed for performance-sensitive callers
        that want to do their own membership checks; prefer is_equivalent()
        otherwise.

        Keys are lowercase phrases. Input to word-error validation is
        already casefolded by TextNormalizer, so normalized text can be
        used as keys directly.
        """
        if not cls.supports_language(language_code):
            return MappingProxyType({})
        return cls._get_lookup(language_code)

    @classmethod
    def max_phrase_length(cls, language_code: str) -> int:
        """
        Longest phrase (in words) appearing in the equivalence data for
        the given language. Callers use this to bound window searches;
        returns 0 when the language has no equivalence data.
        """
        if not cls.supports_language(language_code):
            return 0
        code = cls.normalize_language_code(language_code)
        return max(
            len(member.split())
            for group in cls._EQUIVALENCE_GROUPS[code]
            for member in group
        )

    @classmethod
    def equivalent_phrase(
        cls,
        source_words: list[str],
        source_start: int,
        transcript_words: list[str],
        transcript_start: int,
        language_code: str,
    ) -> tuple[int, int] | None:
        """
        Convenience phrase-aware query for token lists.

        If a source window starting at `source_start` and a transcript
        window starting at `transcript_start` form an equivalence pair,
        returns (source_len, transcript_len) for the smallest such pair
        (scanning source length, then transcript length). Else None.
        """
        if not cls.supports_language(language_code):
            return None

        max_len = cls.max_phrase_length(language_code)
        for s_len in range(1, min(max_len, len(source_words) - source_start) + 1):
            for t_len in range(1, min(max_len, len(transcript_words) - transcript_start) + 1):
                s_phrase = " ".join(source_words[source_start:source_start + s_len])
                t_phrase = " ".join(transcript_words[transcript_start:transcript_start + t_len])
                if cls.is_equivalent(s_phrase, t_phrase, language_code):
                    return (s_len, t_len)
        return None
