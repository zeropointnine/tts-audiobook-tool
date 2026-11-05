from __future__ import annotations

"""
Anchored global forced alignment for the enhance flow.

Aligns the whole flattened source text against the whole stitched transcript
at once. It replaces the greedy per-phrase cursor search
(`enhance_alignment.align_phrases_with_state`), which is kept and selectable
with `ENHANCE_ANCHORED_ALIGNMENT` for A/B testing.

Steps:

1. Tokenize both sides into comparable word tokens. Each source token knows
   its phrase; each transcript token knows the Whisper word (and so the time)
   it came from. Digits are spelled out per token, so "150" and "one hundred
   and fifty" compare equal without losing the token-to-time mapping.
2. Anchor recursively, patience-diff style: within a window, word n-grams that
   occur exactly once on both sides are anchor candidates; the longest chain
   increasing on both sides is kept, and each gap between anchors is aligned
   the same way. Text present on only one side (eg narrated credits, or
   unnarrated front matter) becomes a gap instead of derailing a cursor.
3. Once a window is small, align it with a word-level edit-distance DP
   (exact, fuzzy, merge/split, substitute, insert, delete).
4. Time each phrase from the transcript words its tokens aligned to. Phrases
   that do not align stay unmatched (0.0, 0.0), except interjection-only and
   non-verbal phrases, which are attached to their timed neighbors.

See docs-dev/enhance-audiobook-flow.md ("Forced alignment").
"""

import bisect
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Sequence

from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import Word
from tts_audiobook_tool.app_types.phrase import Phrase
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.text_ops.language_util import normalize_language_code


# Windows of at most this many source x transcript cells are aligned with the
# DP; larger windows are first split by anchors. (Gaps between anchors in real
# audiobooks measured at most ~3.4k cells.)
DP_MAX_CELLS = 20_000

# Anchor n-gram sizes, tried in order until one yields a dense enough chain
ANCHOR_NGRAM_SIZES = (3, 2, 1)
# A window is split only by an anchor chain covering at least this fraction of
# its shorter side; a sparser chain is coincidence between unrelated text (eg
# the wrong book), and the window is left unaligned. Measured: ~0.95 for a
# matching book and transcript, at most ~0.007 for mismatched ones.
MIN_ANCHOR_DENSITY = 0.1

# Safety net on total DP work per alignment, in cells per token of both
# sides (real audiobooks use ~0.2); gaps beyond it are left unaligned
DP_CELL_BUDGET_PER_TOKEN = 10
DP_CELL_BUDGET_MIN = 100_000

# Token comparison costs: exact match 0, substitution/insertion/deletion 1.
# A fuzzy match (misspelled name, stutter, homophone) counts as matched.
FUZZY_MATCH_COST = 0.3
FUZZY_MATCH_MIN_RATIO = 0.75
FUZZY_MATCH_MIN_LENGTH = 3

# A phrase is timed when at least this fraction of its tokens matched...
MIN_MATCHED_FRACTION = 1 / 3
# ...or, when no more than this many tokens long, when it aligned at all and
# its neighbors are timed by that rule (eg "Craeme," heard as "Kramer")
SHORT_PHRASE_MAX_TOKENS = 2

# Shortest pause between timed neighbors that attached phrases are placed in;
# below this they get a zero-length span at the boundary
MIN_ATTACHED_GAP_SECONDS = 0.1

STATUS_TIMED = "timed"
STATUS_ATTACHED = "attached"
STATUS_UNMATCHED = "unmatched"


@dataclass(frozen=True)
class PhraseAlignment:
    status: str
    token_count: int
    # Tokens aligned to a transcript token, including substitutions
    aligned_count: int
    # Tokens matched exactly or fuzzily
    matched_count: int


@dataclass
class AnchoredAlignmentResult:
    timed_phrases: list[TimedPhrase]
    phrase_alignments: list[PhraseAlignment]
    did_interrupt: bool = False

    @property
    def unmatched_count(self) -> int:
        return sum(1 for item in self.phrase_alignments if item.status == STATUS_UNMATCHED)

    @property
    def attached_count(self) -> int:
        return sum(1 for item in self.phrase_alignments if item.status == STATUS_ATTACHED)


def align_phrases(
    phrases: Sequence[Phrase],
    words: Sequence[Word],
    language_code: str = "",
) -> tuple[list[TimedPhrase], bool]:
    """Returns timed phrases parallel to `phrases`, and whether interrupted."""
    result = align_phrases_detailed(phrases, words, language_code)
    return result.timed_phrases, result.did_interrupt


def align_phrases_detailed(
    phrases: Sequence[Phrase],
    words: Sequence[Word],
    language_code: str = "",
) -> AnchoredAlignmentResult:
    language = normalize_language_code(language_code)

    source_tokens: list[str] = []
    phrase_ranges: list[tuple[int, int]] = []
    for phrase in phrases:
        tokens = tokenize(phrase.text, language)
        phrase_ranges.append((len(source_tokens), len(source_tokens) + len(tokens)))
        source_tokens.extend(tokens)

    transcript_tokens: list[str] = []
    token_word: list[int] = []
    for word_index, word in enumerate(words):
        tokens = tokenize(word.word, language)
        transcript_tokens.extend(tokens)
        token_word.extend([word_index] * len(tokens))

    alignment = _align_tokens(source_tokens, transcript_tokens, phonetic=(language == "en"))
    if alignment is None:
        return AnchoredAlignmentResult([], [], did_interrupt=True)

    return _time_phrases(phrases, words, phrase_ranges, transcript_tokens, token_word, alignment)


# --- Tokenization -------------------------------------------------------------

_APOSTROPHES_RE = re.compile(r"[\u2019\u2018'`\u02bc]")
_SEPARATORS_RE = re.compile(r"[\W_]+")
# Three or more of the same letter: "Ahhhhh", "Nooooo", "Ummmm"
_ELONGATION_RE = re.compile(r"([^\W\d_])\1{2,}")
# Hyphenated stutter prefix of a longer following word: "wh-wh-what", "I-I'm",
# "a-all"; not a repetition of a whole word ("ha-ha", "so-so")
_STUTTER_RE = re.compile(
    r"\b([^\W\d_]{1,3})[-\u2010\u2011](?=(?:\1[-\u2010\u2011])*\1[^\W\d_])"
)
_ORDINAL_EN_RE = re.compile(r"(\d{1,6})(st|nd|rd|th)")


def tokenize(text: str, language_code: str = "") -> list[str]:
    """
    Comparable word tokens: casefolded, apostrophes dropped ("don't" -> "dont"),
    stutters collapsed ("Wh-wh-what" -> "what"), other punctuation as
    separators, letter elongations collapsed ("Ahhhh" -> "ah"), and numbers
    spelled out per token ("150" -> "one hundred and fifty").
    """
    text = _STUTTER_RE.sub("", _APOSTROPHES_RE.sub("", text.casefold()))
    tokens: list[str] = []
    for raw in _SEPARATORS_RE.split(text):
        if raw:
            tokens.extend(_expand_token(_ELONGATION_RE.sub(r"\1", raw), language_code))
    return tokens


@lru_cache(maxsize=1 << 16)
def _expand_token(token: str, language_code: str) -> tuple[str, ...]:
    spoken = ""
    if token.isascii() and token.isdigit() and len(token) <= 6:
        spoken = _number_to_words(int(token), language_code, "cardinal")
    elif language_code in ("", "en"):
        match = _ORDINAL_EN_RE.fullmatch(token)
        if match:
            spoken = _number_to_words(int(match.group(1)), "en", "ordinal")
    if not spoken:
        return (token,)
    spoken = _APOSTROPHES_RE.sub("", spoken.casefold())
    return tuple(item for item in _SEPARATORS_RE.split(spoken) if item) or (token,)


def _number_to_words(value: int, language_code: str, to: str) -> str:
    try:
        from num2words import num2words
        return num2words(value, lang=language_code or "en", to=to)
    except Exception:
        # Unsupported language (NotImplementedError) or conversion: keep digits
        return ""


# --- Interjections --------------------------------------------------------------

# Compared after elongation collapse ("Ahhhh" -> "ah", "Hmmm" -> "hm")
_INTERJECTIONS = frozenset("""
    aa aah ack agh ah ahh aha ahem argh au aw aww
    bah e eh ehh eek eep er erm err
    fu fue fueh fufu gah gya gyah ha hah haha hau hawa he hee heh hehe hm hmm hmph hrm
    ho hoo huh hwa hwah hya hyah kya m mhm mm mph mu mwa n ngh nn nngh nya
    o oh ohh oho ooh oof oops ow pff pfft phew psst pst sh shh
    tch tee tsk tut u ugh uh uhh uhm um umm uu uwa uwah wa wah wahh weh whoa
""".split())
# A syllable of one to three letters repeated at least three times, eg
# "hahaha", "ohohoho", "dururururu"
_REPEATED_SYLLABLE_RE = re.compile(r"[^\W\d_]{0,2}?([^\W\d_]{1,3})\1{2,}[^\W\d_]{0,2}")


def is_interjection_only(text: str) -> bool:
    """
    True when every word of `text` is a vocalization: a common interjection
    ("um", "Ahhh", "Heh-heh", "U-um"), a repeated syllable ("Hahaha"), or any
    elongated word ("Nooooo!", "Whaaaat?!").
    """
    text = _STUTTER_RE.sub("", _APOSTROPHES_RE.sub("", text.casefold()))
    raw_tokens = [item for item in _SEPARATORS_RE.split(text) if item]
    return bool(raw_tokens) and all(_is_vocalization(item) for item in raw_tokens)


def _is_vocalization(raw: str) -> bool:
    if raw.isdigit():
        return False
    if _ELONGATION_RE.search(raw):
        return True
    collapsed = _ELONGATION_RE.sub(r"\1", raw)
    return collapsed in _INTERJECTIONS or bool(_REPEATED_SYLLABLE_RE.fullmatch(collapsed))


# --- Token alignment ------------------------------------------------------------

@dataclass
class _TokenAlignment:
    # Per source token: first/last aligned transcript token index, or -1
    first: list[int]
    last: list[int]
    cost: list[float]


def _align_tokens(
    source: list[str], transcript: list[str], phonetic: bool
) -> _TokenAlignment | None:
    """Aligns token sequences; returns None if interrupted."""
    alignment = _TokenAlignment(
        first=[-1] * len(source), last=[-1] * len(source), cost=[1.0] * len(source)
    )
    budget = max(DP_CELL_BUDGET_MIN, DP_CELL_BUDGET_PER_TOKEN * (len(source) + len(transcript)))
    skipped_cells = 0
    stack = [(0, len(source), 0, len(transcript))]
    steps = 0
    while stack:
        steps += 1
        if steps % 256 == 1 and Interrupts().did_interrupt:
            return None
        s0, s1, t0, t1 = stack.pop()
        if s1 <= s0 or t1 <= t0:
            continue  # one side is empty: all insertions or deletions
        cells = (s1 - s0) * (t1 - t0)
        if cells <= DP_MAX_CELLS:
            if cells > budget:
                skipped_cells += cells
                continue
            budget -= cells
            _align_window_dp(source, transcript, s0, s1, t0, t1, alignment, phonetic)
            continue
        pairs = _anchor_pairs(source, transcript, s0, s1, t0, t1)
        if not pairs:
            continue  # large window without a dense anchor chain: leave unaligned
        previous_s, previous_t = s0, t0
        for s, t in pairs:
            stack.append((previous_s, s, previous_t, t))
            alignment.first[s] = alignment.last[s] = t
            alignment.cost[s] = 0.0
            previous_s, previous_t = s + 1, t + 1
        stack.append((previous_s, s1, previous_t, t1))
    if skipped_cells:
        logging.getLogger("tts-audiobook-tool").warning(
            "[enhance] alignment DP budget exhausted; left %s cells unaligned", skipped_cells
        )
    _prefer_later_duplicates(transcript, alignment)
    return alignment


def _prefer_later_duplicates(transcript: list[str], alignment: _TokenAlignment) -> None:
    """
    Moves a source token aligned to a transcript token that is immediately
    followed by an unaligned identical copy ("dont dont you think", typically
    a transcript stitching seam) to the later copy. Either copy is an equally
    good match, but the earlier one would start the phrase at the duplicate.
    """
    used = bytearray(len(transcript))
    for first, last in zip(alignment.first, alignment.last):
        if first >= 0:
            used[first:last + 1] = b"\x01" * (last - first + 1)
    next_first = len(transcript)
    for k in range(len(alignment.first) - 1, -1, -1):
        t = alignment.first[k]
        if t < 0:
            continue
        if t == alignment.last[k]:
            while t + 1 < next_first and not used[t + 1] and transcript[t + 1] == transcript[t]:
                used[t], used[t + 1] = 0, 1
                t += 1
            alignment.first[k] = alignment.last[k] = t
        next_first = alignment.first[k]


def _anchor_pairs(
    source: list[str], transcript: list[str], s0: int, s1: int, t0: int, t1: int
) -> list[tuple[int, int]]:
    """
    Token pairs (source index, transcript index), strictly increasing on both
    sides, from n-grams that occur exactly once in both windows. Empty unless
    the chain covers MIN_ANCHOR_DENSITY of the window's shorter side.
    """
    for n in ANCHOR_NGRAM_SIZES:
        if s1 - s0 < n or t1 - t0 < n:
            continue
        source_counts: dict[tuple[str, ...], int] = {}
        for i in range(s0, s1 - n + 1):
            key = tuple(source[i:i + n])
            source_counts[key] = source_counts.get(key, 0) + 1
        transcript_counts: dict[tuple[str, ...], int] = {}
        transcript_position: dict[tuple[str, ...], int] = {}
        for j in range(t0, t1 - n + 1):
            key = tuple(transcript[j:j + n])
            if source_counts.get(key) == 1:
                transcript_counts[key] = transcript_counts.get(key, 0) + 1
                transcript_position[key] = j
        candidates = [
            (i, transcript_position[key])
            for i in range(s0, s1 - n + 1)
            if transcript_counts.get(key := tuple(source[i:i + n])) == 1
        ]
        chain = _longest_increasing_chain(candidates)
        pairs: list[tuple[int, int]] = []
        for i, j in chain:
            # Overlapping n-grams on different diagonals: the later one wins
            # the overlap. With a duplicated transcript word ("the the sound"),
            # the n-gram ending at the first copy and the one starting at the
            # second both anchor; the source word takes the later copy.
            while pairs and (pairs[-1][0] >= i or pairs[-1][1] >= j):
                pairs.pop()
            pairs.extend((i + k, j + k) for k in range(n))
        if pairs and len(pairs) >= MIN_ANCHOR_DENSITY * min(s1 - s0, t1 - t0):
            return pairs
    return []


def _longest_increasing_chain(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Longest subsequence of `candidates` (sorted by source index) increasing in transcript index."""
    tails: list[int] = []  # smallest transcript index ending a chain of each length
    tail_indices: list[int] = []
    previous = [-1] * len(candidates)
    for index, (_, t) in enumerate(candidates):
        position = bisect.bisect_left(tails, t)
        if position == len(tails):
            tails.append(t)
            tail_indices.append(index)
        else:
            tails[position] = t
            tail_indices[position] = index
        previous[index] = tail_indices[position - 1] if position else -1
    chain: list[tuple[int, int]] = []
    index = tail_indices[-1] if tail_indices else -1
    while index != -1:
        chain.append(candidates[index])
        index = previous[index]
    chain.reverse()
    return chain


_DIAGONAL, _DELETE, _INSERT, _MERGE_SOURCE, _MERGE_TRANSCRIPT = 1, 2, 3, 4, 5


def _align_window_dp(
    source: list[str],
    transcript: list[str],
    s0: int,
    s1: int,
    t0: int,
    t1: int,
    alignment: _TokenAlignment,
    phonetic: bool,
) -> None:
    """
    Word-level edit distance over one window. Ties prefer the diagonal, so
    when the transcript repeats a token (eg a duplicated word) the source token
    takes the later copy.
    """
    a = source[s0:s1]
    b = transcript[t0:t1]
    n, m = len(a), len(b)
    infinity = float("inf")
    dp = [[infinity] * (m + 1) for _ in range(n + 1)]
    moves = [bytearray(m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        row = dp[i]
        move_row = moves[i]
        above = dp[i - 1] if i else None
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = infinity
            move = 0
            if above is not None and j:
                value = above[j - 1] + _token_cost(a[i - 1], b[j - 1], phonetic)
                if value < best:
                    best, move = value, _DIAGONAL
            # Two tokens against one: "high school" <-> "highschool", and
            # fuzzily "Arc Ridill" <-> "Arcredil"
            if i >= 2 and j:
                value = dp[i - 2][j - 1] + _merge_cost(a[i - 2], a[i - 1], b[j - 1], phonetic)
                if value < best:
                    best, move = value, _MERGE_SOURCE
            if above is not None and j >= 2:
                value = above[j - 2] + _merge_cost(b[j - 2], b[j - 1], a[i - 1], phonetic)
                if value < best:
                    best, move = value, _MERGE_TRANSCRIPT
            if above is not None:
                value = above[j] + 1.0
                if value < best:
                    best, move = value, _DELETE
            if j:
                value = row[j - 1] + 1.0
                if value < best:
                    best, move = value, _INSERT
            row[j] = best
            move_row[j] = move

    i, j = n, m
    while i or j:
        move = moves[i][j]
        if move == _DIAGONAL:
            alignment.first[s0 + i - 1] = alignment.last[s0 + i - 1] = t0 + j - 1
            alignment.cost[s0 + i - 1] = _token_cost(a[i - 1], b[j - 1], phonetic)
            i, j = i - 1, j - 1
        elif move == _MERGE_SOURCE:
            merge_cost = _merge_cost(a[i - 2], a[i - 1], b[j - 1], phonetic)
            for k in (i - 1, i - 2):
                alignment.first[s0 + k] = alignment.last[s0 + k] = t0 + j - 1
                alignment.cost[s0 + k] = merge_cost
            i, j = i - 2, j - 1
        elif move == _MERGE_TRANSCRIPT:
            alignment.first[s0 + i - 1] = t0 + j - 2
            alignment.last[s0 + i - 1] = t0 + j - 1
            alignment.cost[s0 + i - 1] = _merge_cost(b[j - 2], b[j - 1], a[i - 1], phonetic)
            i, j = i - 1, j - 2
        elif move == _DELETE:
            i -= 1
        else:
            j -= 1


@lru_cache(maxsize=1 << 16)
def _token_cost(a: str, b: str, phonetic: bool) -> float:
    if a == b:
        return 0.0
    if len(a) >= FUZZY_MATCH_MIN_LENGTH and len(b) >= FUZZY_MATCH_MIN_LENGTH:
        if phonetic and _metaphone(a) and _metaphone(a) == _metaphone(b):
            return FUZZY_MATCH_COST
        # 2 * shorter / total bounds the ratio; skips most unrelated pairs cheaply
        if 2 * min(len(a), len(b)) >= FUZZY_MATCH_MIN_RATIO * (len(a) + len(b)):
            if SequenceMatcher(None, a, b).ratio() >= FUZZY_MATCH_MIN_RATIO:
                return FUZZY_MATCH_COST
    return 1.0


@lru_cache(maxsize=1 << 16)
def _merge_cost(first: str, second: str, joined: str, phonetic: bool) -> float:
    """
    Cost of aligning two tokens with one; infinite unless it is a (fuzzy)
    match. A fuzzy merge also requires that neither part matches alone, or
    "the" + "sound" would fuzzily swallow a duplicated "the" before "sound".
    """
    if first + second == joined:
        return 0.0
    cost = _token_cost(first + second, joined, phonetic)
    if cost >= 1.0:
        return float("inf")
    if _token_cost(first, joined, phonetic) < 1.0 or _token_cost(second, joined, phonetic) < 1.0:
        return float("inf")
    return cost


@lru_cache(maxsize=1 << 16)
def _metaphone(token: str) -> str:
    from metaphone import doublemetaphone
    return doublemetaphone(token)[0]


# --- Phrase timing --------------------------------------------------------------

@dataclass
class _Span:
    start: float
    end: float
    # End of the phrase's last aligned word, before the gap to the next word
    last_word_end: float


def _time_phrases(
    phrases: Sequence[Phrase],
    words: Sequence[Word],
    phrase_ranges: list[tuple[int, int]],
    transcript: list[str],
    token_word: list[int],
    alignment: _TokenAlignment,
) -> AnchoredAlignmentResult:
    # Per phrase: (token count, aligned count, matched count)
    details: list[tuple[int, int, int]] = []
    aligned_tokens: list[list[int]] = []
    # Interjection-only or non-verbal: timed only by a true match, else attached
    attachable: list[bool] = []
    # Enough tokens matched to be timed on its own
    strong: list[bool] = []
    for phrase, (a, b) in zip(phrases, phrase_ranges):
        token_count = b - a
        aligned = [k for k in range(a, b) if alignment.first[k] >= 0]
        matched_count = sum(1 for k in aligned if alignment.cost[k] < 1.0)
        aligned_tokens.append(aligned)
        details.append((token_count, len(aligned), matched_count))
        attachable.append(token_count == 0 or is_interjection_only(phrase.text))
        strong.append(
            matched_count > 0 and matched_count >= MIN_MATCHED_FRACTION * token_count - 1e-9
        )

    def neighbor_is_strong(index: int, step: int) -> bool:
        # The nearest phrase that is not a vocalization; none counts as strong
        index += step
        while 0 <= index < len(phrases) and attachable[index]:
            index += step
        return not (0 <= index < len(phrases)) or strong[index]

    aligned_transcript = bytearray(len(transcript))
    for first, last in zip(alignment.first, alignment.last):
        if first >= 0:
            aligned_transcript[first:last + 1] = b"\x01" * (last - first + 1)

    statuses: list[str] = []
    spans: list[_Span | None] = []
    for index in range(len(phrases)):
        aligned = aligned_tokens[index]
        token_count, _, matched_count = details[index]
        is_vocal = attachable[index]
        if is_vocal:
            # Vocalizations count only when they truly matched: a substitution
            # pairs them with an arbitrary word
            is_timed = matched_count > 0
        else:
            is_timed = strong[index] or (
                bool(aligned)
                and token_count <= SHORT_PHRASE_MAX_TOKENS
                and neighbor_is_strong(index, -1)
                and neighbor_is_strong(index, 1)
            )
        if not is_timed:
            statuses.append(STATUS_UNMATCHED)
            spans.append(None)
            continue
        used = [k for k in aligned if alignment.cost[k] < 1.0] if is_vocal else aligned
        first_token = _include_leading_stutter(
            min(alignment.first[k] for k in used), transcript, aligned_transcript
        )
        first_word = token_word[first_token]
        last_word = token_word[max(alignment.last[k] for k in used)]
        end = words[last_word + 1].start if last_word + 1 < len(words) else words[last_word].end
        statuses.append(STATUS_TIMED)
        spans.append(_Span(words[first_word].start, end, words[last_word].end))

    _clip_overlaps(spans)
    _attach_vocalizations(spans, statuses, attachable)

    timed_phrases = [
        TimedPhrase(phrase.text, span.start, span.end, fmt=phrase.fmt)
        if span is not None
        else TimedPhrase(phrase.text, 0.0, 0.0, fmt=phrase.fmt)
        for phrase, span in zip(phrases, spans)
    ]
    phrase_alignments = [
        PhraseAlignment(status, token_count, aligned_count, matched_count)
        for status, (token_count, aligned_count, matched_count) in zip(statuses, details)
    ]
    return AnchoredAlignmentResult(timed_phrases, phrase_alignments)


def _include_leading_stutter(first: int, transcript: list[str], aligned: bytearray) -> int:
    """
    Moves a phrase's first transcript token back over unaligned stutter
    prefixes of it. Source stutters collapse within a word ("M-Monica" ->
    "monica"), but Whisper may emit the stutter as its own word ("M-", "Monica").
    """
    while (
        first > 0
        and not aligned[first - 1]
        and len(transcript[first - 1]) <= 3
        and transcript[first - 1] != transcript[first]
        and transcript[first].startswith(transcript[first - 1])
    ):
        first -= 1
    return first


def _clip_overlaps(spans: list[_Span | None]) -> None:
    """
    Keeps timed spans ordered and non-overlapping. A span normally ends where
    the next transcript word starts; when two phrases share a Whisper word (eg
    "twenty-plus" split by segmentation), the earlier span is cut at the later
    one's start.
    """
    following_start: float | None = None
    for span in reversed(spans):
        if span is None:
            continue
        if following_start is not None and span.end > following_start:
            span.end = max(span.start, following_start)
        following_start = span.start


def _attach_vocalizations(
    spans: list[_Span | None], statuses: list[str], attachable: list[bool]
) -> None:
    """
    Times runs of untimed interjection-only or non-verbal phrases from their
    neighbors. Whisper usually omits such sounds, so the run is placed in the
    pause between the previous phrase's last word and the next phrase, split
    evenly; a pause too short for that gives a zero-length span at the boundary.
    A run that also holds an unmatched ordinary phrase stays unmatched with it.
    """
    index = 0
    while index < len(spans):
        if spans[index] is not None:
            index += 1
            continue
        run_end = index
        while run_end < len(spans) and spans[run_end] is None:
            run_end += 1
        run = range(index, run_end)
        previous = spans[index - 1] if index > 0 else None
        following = spans[run_end] if run_end < len(spans) else None
        if all(attachable[k] for k in run) and (previous or following):
            if previous and following:
                gap_start = previous.last_word_end
                gap_end = following.start
                if gap_end - gap_start >= MIN_ATTACHED_GAP_SECONDS:
                    previous.end = min(previous.end, gap_start)
                    step = (gap_end - gap_start) / len(run)
                    for offset, k in enumerate(run):
                        start = gap_start + offset * step
                        spans[k] = _Span(start, start + step, start + step)
                        statuses[k] = STATUS_ATTACHED
                    index = run_end
                    continue
            if following is not None:
                boundary = following.start
            else:
                assert previous is not None
                boundary = previous.end
            for k in run:
                spans[k] = _Span(boundary, boundary, boundary)
                statuses[k] = STATUS_ATTACHED
        index = run_end
