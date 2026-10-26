"""
Saved attribution results: who speaks each dialog utterance, per attributed
line range, stored as `[project]/speaker_assign/attributions.json` (see
SpeakerAssignFileUtil). Speaker labels are roster ids or reserved answers (see
SpeakerLabelUtil). Line ranges are half-open over book-wide line indices.

On disk:

    {
        "format": "speaker_attributions.v1",
        "ranges": [                                 # oldest first
            {
                "start": 251, "end": 481,
                "text_hash": "3ae0f21b9de65a3c",    # of the range's text when attributed
                "created": "2026-10-22T12:57:03-04:00",
                "llm": {"url": "...", "model": "...", "extra_params": {}},
                "utterances": [
                    {"lines": [252], "speaker": "BARON", "uncertain": false, "text": "“PEOPLE DIE,”"}
                ]
            }
        ]
    }

Instances are immutable; `AttributionsFile.with_range` returns a new file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil


ATTRIBUTIONS_FILE_FORMAT = "speaker_attributions.v1"
# Stored utterance text is an excerpt, for review only
UTTERANCE_TEXT_EXCERPT_CHARS = 120


@dataclass(frozen=True)
class UtteranceAttribution:
    """The attributed speaker of one utterance (its book-wide lines)."""
    lines: tuple[int, ...]
    speaker: str
    uncertain: bool = False
    text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "lines", tuple(self.lines))
        if not self.lines or not all(isinstance(line, int) and line >= 0 for line in self.lines):
            raise ValueError(f"Utterance attribution needs non-negative line indices: {self.lines!r}")
        if not isinstance(self.speaker, str) or not self.speaker:
            raise ValueError("Utterance attribution needs a speaker label")

    @property
    def first_line(self) -> int:
        return self.lines[0]

    def to_dict(self) -> dict[str, Any]:
        return {"lines": list(self.lines), "speaker": self.speaker, "uncertain": self.uncertain, "text": self.text}

    @staticmethod
    def from_dict(value: Any) -> UtteranceAttribution:
        if not isinstance(value, dict) or not isinstance(value.get("lines"), list):
            raise ValueError(f"Expected an utterance attribution object. Value: {value!r}")
        text = value.get("text")
        return UtteranceAttribution(
            lines=tuple(value["lines"]),
            speaker=value.get("speaker"),  # type: ignore[arg-type]  (validated by __post_init__)
            uncertain=value.get("uncertain") is True,
            text=text if isinstance(text, str) else "",
        )


@dataclass(frozen=True)
class RangeAttribution:
    """The attributions of one attributed line range [start, end)."""
    start: int
    end: int
    text_hash: str
    utterances: tuple[UtteranceAttribution, ...]
    created: str = ""
    llm: dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "utterances", tuple(self.utterances))
        object.__setattr__(self, "llm", dict(self.llm))
        if not (isinstance(self.start, int) and isinstance(self.end, int) and 0 <= self.start < self.end):
            raise ValueError(f"Invalid line range: {self.start!r}-{self.end!r}")
        if any(not self.start <= line < self.end for u in self.utterances for line in u.lines):
            raise ValueError(f"Utterance lines outside the range {self.start}-{self.end}")

    @property
    def key(self) -> tuple[int, int]:
        return self.start, self.end

    def is_valid_for(self, book: Book) -> bool:
        """Whether the book's text in this range is unchanged since it was attributed."""
        return (self.end <= len(book.flat_phrase_groups)
                and UtteranceUtil.range_text_hash(book, self.start, self.end) == self.text_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
            "created": self.created,
            "llm": dict(self.llm),
            "utterances": [u.to_dict() for u in self.utterances],
        }

    @staticmethod
    def from_dict(value: Any) -> RangeAttribution:
        if not isinstance(value, dict) or not isinstance(value.get("utterances"), list):
            raise ValueError(f"Expected a range attribution object. Value: {value!r}")
        created, llm, text_hash = value.get("created"), value.get("llm"), value.get("text_hash")
        return RangeAttribution(
            start=value.get("start"),  # type: ignore[arg-type]  (validated by __post_init__)
            end=value.get("end"),  # type: ignore[arg-type]
            text_hash=text_hash if isinstance(text_hash, str) else "",
            utterances=tuple(UtteranceAttribution.from_dict(u) for u in value["utterances"]),
            created=created if isinstance(created, str) else "",
            llm=llm if isinstance(llm, dict) else {},
        )


@dataclass(frozen=True)
class AttributionsFile:
    """All saved range attributions, oldest first: where ranges overlap,
    later ones take precedence. At most one entry per (start, end)."""
    ranges: tuple[RangeAttribution, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranges", tuple(self.ranges))
        keys = [r.key for r in self.ranges]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate attributed line ranges")

    def get(self, start: int, end: int) -> RangeAttribution | None:
        return next((r for r in self.ranges if r.key == (start, end)), None)

    def with_range(self, attribution: RangeAttribution) -> AttributionsFile:
        """A copy with `attribution` as the newest entry (replacing any saved
        entry for the same range)."""
        kept = tuple(r for r in self.ranges if r.key != attribution.key)
        return AttributionsFile(kept + (attribution,))

    def valid_ranges(self, book: Book) -> list[RangeAttribution]:
        """The saved ranges whose text is unchanged in `book`, oldest first."""
        return [r for r in self.ranges if r.is_valid_for(book)]

    def by_first_line(self, book: Book) -> dict[int, UtteranceAttribution]:
        """Each still-valid utterance attribution by its first line; where
        ranges overlap, the newer entry's attribution wins."""
        result: dict[int, UtteranceAttribution] = {}
        for attribution in self.valid_ranges(book):
            for utterance in attribution.utterances:
                result[utterance.first_line] = utterance
        return result

    def speakers_by_line(self, book: Book) -> dict[int, str]:
        """Speaker label by utterance first line (see by_first_line)."""
        return {line: u.speaker for line, u in self.by_first_line(book).items()}

    def speaker_of_each_line(self, book: Book) -> dict[int, str]:
        """Speaker label of every line of each still-valid utterance (not just
        its first line, unlike speakers_by_line)."""
        return {line: u.speaker for u in self.by_first_line(book).values() for line in u.lines}

    def to_dict(self) -> dict[str, Any]:
        return {"format": ATTRIBUTIONS_FILE_FORMAT, "ranges": [r.to_dict() for r in self.ranges]}

    @staticmethod
    def from_dict(value: Any) -> AttributionsFile:
        if (not isinstance(value, dict) or value.get("format") != ATTRIBUTIONS_FILE_FORMAT
                or not isinstance(value.get("ranges"), list)):
            raise ValueError(f"Not a {ATTRIBUTIONS_FILE_FORMAT} attributions file")
        return AttributionsFile(tuple(RangeAttribution.from_dict(r) for r in value["ranges"]))
