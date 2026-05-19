from dataclasses import dataclass
from typing import Any

from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.util import *


@dataclass
class SegmentTranscriptData:
    """Persisted STT-derived sidecar info for a generated sound segment."""

    # Schema version for this sidecar file format.
    version: int

    # Schema/type discriminator for this sidecar payload.
    type: str

    # Project language code used for STT and text normalization.
    language_code: str

    # One-based phrase group index, matching user-facing line numbering.
    index_1b: int

    # Source text from the phrase group that this segment should speak.
    source: str

    # Final TTS prompt text used for generation.
    prompt: str

    # Raw STT transcript text reconstructed from transcript words.
    transcript: str

    # Normalized source text used for word-error comparison.
    normalized_source: str

    # Normalized STT transcript text used for word-error comparison.
    normalized_transcript: str

    # Word error count detected at generation time, matching the saved filename tag.
    generation_word_error_count: int

    # Phrase-level timings used to subdivide app metadata during concat.
    timed_phrases: list[TimedPhrase]

    # Word-level STT transcript data, including word text and timestamps.
    transcript_words: list[dict[str, Any]]

    # Exceptional compromise reason for the whole segment, if any.
    exception: str | None
