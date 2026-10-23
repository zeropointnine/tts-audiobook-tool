from __future__ import annotations

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.app_types.phrase import Reason


DEFAULT_CHUNK_WORDS = 5000
# Smaller break-delimited pieces (chapter headings, dedications) join the following text
MIN_PIECE_WORDS = 20
CHUNK_BREAK_REASONS = (Reason.SPACE_BREAK, Reason.SECTION_BREAK)
PARAGRAPH_END_REASONS = (Reason.PARAGRAPH, Reason.HEADING_END)


class ChunkPlanUtil:
    """
    Splits a book into line ranges ("chunks") for whole-book attribution, one
    LLM request per chunk. Ranges are half-open (start, end).
    """

    @staticmethod
    def plan_chunks(book: Book, chunk_words: int = DEFAULT_CHUNK_WORDS) -> list[tuple[int, int]]:
        """
        Contiguous ranges covering the whole book, of up to about chunk_words words:
        1) cut after every space/section break and at section boundaries; pieces
           under MIN_PIECE_WORDS (chapter headings, dedications) join the
           following piece, so a chunk never ends on a heading;
        2) split pieces over chunk_words in two, repeatedly, at a paragraph end
           near the middle with narration around it (see _choose_split_line);
        3) merge consecutive pieces while the total stays within chunk_words.
        """
        groups = book.flat_phrase_groups
        if not groups:
            return []
        words = [len(group.text.split()) for group in groups]
        section_starts = set(book.section_start_indices())
        line_count = len(groups)

        pieces: list[tuple[int, int]] = []
        piece_start = 0
        for line, group in enumerate(groups):
            if (line + 1 == line_count or line + 1 in section_starts
                    or group.phrases[-1].reason in CHUNK_BREAK_REASONS):
                pieces.append((piece_start, line + 1))
                piece_start = line + 1

        joined: list[tuple[int, int]] = []
        pending_start: int | None = None
        for start, end in pieces:
            if pending_start is not None:
                start, pending_start = pending_start, None
            if sum(words[start:end]) < MIN_PIECE_WORDS and end != line_count:
                pending_start = start
                continue
            joined.append((start, end))

        split: list[tuple[int, int]] = []
        for start, end in joined:
            split.extend(ChunkPlanUtil._split_piece(book, words, start, end, chunk_words))

        chunks: list[tuple[int, int]] = []
        for start, end in split:
            if chunks and sum(words[chunks[-1][0]:end]) <= chunk_words:
                chunks[-1] = (chunks[-1][0], end)
            else:
                chunks.append((start, end))
        return chunks

    @staticmethod
    def _split_piece(
            book: Book, words: list[int], start: int, end: int, chunk_words: int,
    ) -> list[tuple[int, int]]:
        if end - start <= 1 or sum(words[start:end]) <= chunk_words:
            return [(start, end)]
        cut = ChunkPlanUtil._choose_split_line(book, words, start, end)
        return (ChunkPlanUtil._split_piece(book, words, start, cut + 1, chunk_words)
                + ChunkPlanUtil._split_piece(book, words, cut + 1, end, chunk_words))

    @staticmethod
    def _choose_split_line(book: Book, words: list[int], start: int, end: int) -> int:
        """
        The line after which to split [start, end) in two, avoiding cutting a
        conversation: a paragraph end within the middle half of the piece's
        words with the most narration (non-dialog lines) around it, then
        nearest the middle. Falls back to the paragraph end, then sentence end,
        nearest the middle, then the middle line.
        """
        groups = book.flat_phrase_groups
        total = sum(words[start:end])

        def is_dialog(line: int) -> bool:
            return groups[line].voice_index >= 1

        def ends_with(line: int, reasons: tuple[Reason, ...]) -> bool:
            return groups[line].phrases[-1].reason in reasons

        def narration_around(line: int) -> int:
            """Consecutive non-dialog lines, up to 6 on each side of the cut."""
            count = 0
            for other in range(line, max(start, line - 5) - 1, -1):
                if is_dialog(other):
                    break
                count += 1
            for other in range(line + 1, min(end, line + 7)):
                if is_dialog(other):
                    break
                count += 1
            return count

        # (line, distance of a cut after it from the middle, as a fraction of the words)
        candidates: list[tuple[int, float]] = []
        running = 0
        for line in range(start, end - 1):
            running += words[line]
            candidates.append((line, abs(running - total / 2) / total))

        middle = [c for c in candidates if c[1] <= 0.25 and ends_with(c[0], PARAGRAPH_END_REASONS)]
        if middle:
            return max(middle, key=lambda c: (narration_around(c[0]), -c[1]))[0]
        for reasons in (PARAGRAPH_END_REASONS, (Reason.SENTENCE,)):
            options = [c for c in candidates if ends_with(c[0], reasons)]
            if options:
                return min(options, key=lambda c: c[1])[0]
        return min(candidates, key=lambda c: c[1])[0]
