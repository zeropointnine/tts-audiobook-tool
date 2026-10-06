from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from tts_audiobook_tool.text_ops import language_util


class WordEquivalence:
    """
    Two-way lookup of declared word/phrase relationships, per language.

    Pairs declare matching variants (single words or multi-word phrases),
    eg "all right" <-> "alright". Pairs can share a variant
    without making their other endpoints equivalent: "hed" can match both
    "he had" and "he would" without allowing "he had" <-> "he would".
    Relationships are direct, never transitively inferred.

    This supplements (and is distinct from) the Double Metaphone homophone
    check in TextNormalizer.sounds_the_same_en, which is single-word-only
    and cannot relate a phrase to a single word.

    Entries must be lowercase, single-spaced, and normalized in the same
    spirit as TextNormalizer.normalize_common() output (no punctuation).
    """

    # Each declared pair is automatically indexed in both directions.
    _EQUIVALENCE_PAIRS: dict[str, list[tuple[str, str]]] = {

        "en": [

            # Note slightly liberal application of equivalencies here generally,
            # especially wrt contractions, which Whisper frequently elides anyway.

            # Split/merged forms whose letters differ, so
            # TextNormalizer.normalize_spacing_en cannot align them
            ("all right", "alright"),
            ("all together", "altogether"),
            ("all ready", "already"),
            ("can not", "cannot"),
            ("where ever", "wherever"),

            # Contractions (note that normalization strips apostrophes)
            # (Omitting pairs that have greater 'sound difference') (editorial call)

            ("can not", "cant"),
            ("cannot", "cant"),
            ("did not", "didnt"),
            ("do not", "dont"),
            ("i am", "im"),
            ("it is", "its"),
            ("might have", "mightve"),
            ("must have", "mustve"),
            ("should have", "shouldve"),
            ("that would", "thatd"),
            ("there are", "therere"),
            ("there is", "theres"),
            ("they are", "theyre"),
            ("what are", "whatre"),
            ("what have", "whatve"),
            ("where are", "wherere"),
            ("where have", "whereve"),
            ("whod have", "whodve"),
            ("why is", "whys"),
            ("why are", "whyre"),
            ("will not", "wont"),
            ("would have", "wouldve"),
            ("you are", "youre"),

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

    # Derived two-way lookup: variant -> its directly related variants.
    # Cached mappings and neighbor sets are read-only. Overlapping declarations
    # accumulate edges, without overwriting earlier edges or adding transitive ones.
    _LOOKUP: dict[str, Mapping[str, frozenset[str]]] = {}

    @staticmethod
    def normalize_language_code(language_code: str) -> str:
        """Base lowercase language code; see `language_util.normalize_language_code`."""
        return language_util.normalize_language_code(language_code)

    @classmethod
    def supports_language(cls, language_code: str) -> bool:
        code = cls.normalize_language_code(language_code)
        return code in cls._EQUIVALENCE_PAIRS

    @staticmethod
    def _validate_pair(pair: tuple[str, str], language_code: str) -> None:
        """Reject declarations that cannot match normalized comparison text."""
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(
                f"Equivalence pair for {language_code!r} needs exactly two distinct variants: {pair!r}"
            )
        for phrase in pair:
            if (
                not phrase
                or phrase != phrase.casefold()
                or phrase != " ".join(phrase.split())
                or not all(word.isalnum() for word in phrase.split())
            ):
                raise ValueError(
                    f"Invalid equivalence phrase for {language_code!r}: {phrase!r}; "
                    "use nonempty, casefolded, single-spaced alphanumeric words"
                )

    @classmethod
    def _get_lookup(cls, language_code: str) -> Mapping[str, frozenset[str]]:
        """Lazily validate and cache a read-only direct-neighbor index."""
        code = cls.normalize_language_code(language_code)
        lookup = cls._LOOKUP.get(code)
        if lookup is None:
            mutable_lookup: dict[str, set[str]] = {}
            for pair in cls._EQUIVALENCE_PAIRS.get(code, []):
                cls._validate_pair(pair, code)
                a, b = pair
                mutable_lookup.setdefault(a, set()).add(b)
                mutable_lookup.setdefault(b, set()).add(a)
            # Publish only after every entry validates, so a failure cannot leave
            # a partially populated cache behind.
            lookup = MappingProxyType({
                member: frozenset(neighbors)
                for member, neighbors in mutable_lookup.items()
            })
            cls._LOOKUP[code] = lookup
        return lookup

    @classmethod
    def is_equivalent(cls, source_phrase: str, transcript_phrase: str, language_code: str) -> bool:
        """
        Returns True if the two words/phrases have a declared direct
        relationship for the given language. Identical indexed variants also
        return True, preserving the existing query API; alignment itself uses
        direct matching for identical text.

        Phrases are compared case-insensitively with whitespace collapsed,
        since callers may pass raw word-list joins.
        """
        if not cls.supports_language(language_code):
            return False

        a = " ".join(source_phrase.casefold().split())
        b = " ".join(transcript_phrase.casefold().split())
        if not a or not b:
            return False

        neighbors = cls._get_lookup(language_code).get(a)
        return neighbors is not None and (a == b or b in neighbors)

    @classmethod
    def get_lookup(cls, language_code: str) -> Mapping[str, frozenset[str]]:
        """
        Returns the read-only variant -> direct neighbors index for a language
        (empty mapping if unsupported). Exposed for performance-sensitive callers
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
        return max(
            (len(member.split()) for member in cls.get_lookup(language_code)),
            default=0,
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
        window starting at `transcript_start` form a declared equivalence pair,
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
