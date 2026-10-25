from __future__ import annotations

from collections.abc import Mapping, Sequence

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.speaker_assign.roster import NarrativePerson, Roster
from tts_audiobook_tool.speaker_assign.speaker_assign_types import Utterance
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil


ATTRIBUTION_SYSTEM_PROMPT = """\
You are a careful literary analyst performing quotation attribution for a
multi-voice audiobook: for each quoted dialog passage you decide which
character speaks it.\
"""

ATTRIBUTION_INSTRUCTIONS = """\
Below is a passage of a book (a chapter, part of one, or several). Each
detected dialog passage is wrapped in <q id="N">...</q>. For EVERY id, decide
who speaks it.

The passage may be preceded by context text. In the context, dialog that was
already attributed is wrapped in <q speaker="...">...</q>; use it to follow the
conversation, but give answers only for the ids in the passage.

Allowed answers:
- a character id from the roster
- "OTHER:<short name>" for a speaker not in the roster (eg "OTHER:guard")
- "UNKNOWN" if the speaker cannot be determined
- "NOT_SPEECH" if the tagged text is not spoken dialog at all (eg a quoted
  title, sign, label, scare quote or written text)

Guidance:
- Follow explicit attributions ("said X", "X asks") and pronoun attributions
  by resolving who the pronoun refers to.
- For untagged lines in a back-and-forth exchange, track turn-taking.
- Consecutive tags in the same paragraph separated only by an attribution
  (eg <q>"Hi,"</q> she said, <q>"there."</q>) almost always share a speaker.
- In first-person narration, dialog attributed to "I" belongs to the narrator character.

Respond with only a JSON object, no commentary:
{"speakers": {"1": "<answer>", "2": "<answer>", ...}, "uncertain": [<ids you are unsure about>]}\
"""

# Approximate size of the context text given before each attributed range
CONTEXT_CHARS = 2500


class AttributionPromptUtil:
    """
    Builds the attribution request for a line range [start, end). Utterance
    ids in the request are the utterances' 1-based positions in
    UtteranceUtil.find_utterances(book, start, end).
    """

    @staticmethod
    def build_message(
            book: Book,
            start: int,
            end: int,
            roster: Roster,
            speakers_by_line: Mapping[int, str],
    ) -> str:
        """
        The user message: instructions and the full roster first (identical
        across requests, so provider prompt caching applies), then the context
        before the range with already-attributed utterances tagged by speaker
        (speakers_by_line: utterance first line -> label), then the range with
        each utterance tagged <q id="N">.
        """
        parts = [ATTRIBUTION_INSTRUCTIONS, "", "Character roster:", AttributionPromptUtil.roster_text(roster)]
        if roster.narration.person is not NarrativePerson.UNKNOWN:
            parts.append(f"\nThe book is narrated in the {roster.narration.person.value} person.")
        context = AttributionPromptUtil.context_before(book, start, speakers_by_line)
        if context:
            parts += ["", "===== CONTEXT: PRECEDING TEXT (do not attribute) =====", context]
        passage = UtteranceUtil.annotate_range(
            book, start, end, lambda number, _: f'<q id="{number}">', section_headers=True)
        parts += [
            "",
            f"===== TEXT TO ATTRIBUTE: {UtteranceUtil.range_label(book, start, end)} =====",
            passage.strip(),
            "===== TEXT ENDS =====",
            "",
            "Now respond with the JSON object covering every id.",
        ]
        return "\n".join(parts)

    @staticmethod
    def roster_text(roster: Roster) -> str:
        """One line per character, LLM-facing fields included."""
        lines = []
        for character in roster.characters:
            narrator = " (the first-person narrator)" if character.is_narrator else ""
            aliases = ", ".join(character.aliases)
            lines.append(
                f"- {character.id}: {character.name}{narrator}; {character.gender}; "
                f"aliases: {aliases or '-'}; {character.description}"
            )
        return "\n".join(lines)

    @staticmethod
    def context_before(book: Book, start: int, speakers_by_line: Mapping[int, str]) -> str:
        """Whole lines directly before `start` (about CONTEXT_CHARS of text),
        with already-attributed utterances tagged <q speaker="ID">. This
        carries the conversation state across range boundaries."""
        first = start
        size = 0
        while first > 0 and size < CONTEXT_CHARS:
            first -= 1
            size += len(book.flat_phrase_groups[first].text)
        if first == start:
            return ""

        def open_tag(_: int, utterance: Utterance) -> str | None:
            speaker = speakers_by_line.get(utterance.first_line)
            return f'<q speaker="{speaker}">' if speaker else None

        return UtteranceUtil.annotate_range(book, first, start, open_tag, section_headers=True).strip()

    @staticmethod
    def missing_ids_message(missing_ids: Sequence[int]) -> str:
        """A follow-up message asking for ids the reply omitted."""
        return (
            f"Your reply omitted these ids: {', '.join(map(str, missing_ids))}. "
            'Respond with only a JSON object {"speakers": {...}, "uncertain": [...]} for those ids.'
        )
