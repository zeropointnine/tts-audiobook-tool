from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
import json
import re
from typing import Any

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.speaker_assign.attribution import (
    UTTERANCE_TEXT_EXCERPT_CHARS,
    RangeAttribution,
    UtteranceAttribution,
)
from tts_audiobook_tool.speaker_assign.speaker_assign_types import UNKNOWN, Utterance
from tts_audiobook_tool.speaker_assign.speaker_label_util import SpeakerLabelUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil


@dataclass(frozen=True)
class ParsedReply:
    """An attribution reply: speaker label per utterance id (1-based, as in the
    request), the ids flagged uncertain, and requested ids the reply omitted."""
    speakers: dict[int, str]
    uncertain: frozenset[int]
    missing: tuple[int, ...]


class AttributionReplyUtil:
    """Parses attribution replies and turns them into RangeAttributions."""

    @staticmethod
    def extract_json(reply: str) -> dict[str, Any]:
        """The JSON object in an LLM reply (a ```json fence is allowed).
        Raises ValueError if there is none."""
        text = reply.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in LLM reply")
        value = json.loads(text[start:end + 1])
        if not isinstance(value, dict):
            raise ValueError("LLM reply JSON is not an object")
        return value

    @staticmethod
    def parse_reply(value: Mapping[str, Any], requested_ids: Set[int], roster_ids: Set[str]) -> ParsedReply:
        """
        Speakers from a reply's {"speakers": {...}, "uncertain": [...]}.
        Answers are normalized to speaker labels (see SpeakerLabelUtil);
        answers for ids that weren't requested are ignored. Raises ValueError
        if the reply has no "speakers" object.
        """
        raw_speakers = value.get("speakers")
        if not isinstance(raw_speakers, dict):
            raise ValueError("Reply has no 'speakers' object")
        speakers: dict[int, str] = {}
        for key, raw in raw_speakers.items():
            try:
                utterance_id = int(str(key).strip())
            except ValueError:
                continue
            if utterance_id in requested_ids:
                speakers[utterance_id] = SpeakerLabelUtil.normalize(raw, roster_ids)
        uncertain: set[int] = set()
        raw_uncertain = value.get("uncertain")
        if isinstance(raw_uncertain, list):
            for item in raw_uncertain:
                try:
                    uncertain.add(int(str(item).strip()))
                except ValueError:
                    pass
        missing = tuple(sorted(set(requested_ids) - speakers.keys()))
        return ParsedReply(speakers, frozenset(uncertain & set(requested_ids)), missing)

    @staticmethod
    def merge(first: ParsedReply, follow_up: ParsedReply) -> ParsedReply:
        """A reply completed by a follow-up reply for its missing ids."""
        speakers = {**first.speakers, **follow_up.speakers}
        return ParsedReply(
            speakers,
            first.uncertain | follow_up.uncertain,
            tuple(i for i in first.missing if i not in speakers),
        )

    @staticmethod
    def make_range_attribution(
            book: Book,
            start: int,
            end: int,
            utterances: list[Utterance],
            reply: ParsedReply,
            created: str = "",
            llm: Mapping[str, Any] | None = None,
    ) -> RangeAttribution:
        """The saved form of a parsed reply for [start, end); `utterances` are
        UtteranceUtil.find_utterances(book, start, end) (reply ids are their
        1-based positions). Utterances the reply has no answer for are UNKNOWN."""
        return RangeAttribution(
            start=start,
            end=end,
            text_hash=UtteranceUtil.range_text_hash(book, start, end),
            utterances=tuple(
                UtteranceAttribution(
                    lines=utterance.lines,
                    speaker=reply.speakers.get(number, UNKNOWN),
                    uncertain=number in reply.uncertain,
                    text=utterance.text.strip()[:UTTERANCE_TEXT_EXCERPT_CHARS],
                )
                for number, utterance in enumerate(utterances, 1)
            ),
            created=created,
            llm=dict(llm or {}),
        )
