"""
Attaches display formatting (`Phrase.fmt`) to already-segmented phrases.

Formatting is attached *after* segmentation instead of being threaded through
the segmenter, grouper and dialog segmenter: those stages only rearrange
whitespace, so the styles of a source text can be projected onto the
concatenated phrase text by aligning words. See docs-dev/text-formatting-layer.md
"""

from __future__ import annotations

from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup
from tts_audiobook_tool.text_ops import text_format
from tts_audiobook_tool.text_ops.text_format import MaskAlignment, StyledText


def attach_formatting(phrases: list[Phrase], source: StyledText) -> MaskAlignment:
    """
    Sets `fmt` on every phrase from the styles of `source`, replacing any
    existing value (None where a phrase ends up unstyled). `phrases` should be
    one contiguous run of text, such as a section, whose words match `source`.
    """
    target = "".join(phrase.text for phrase in phrases)
    alignment = text_format.align_masks(source, target)
    offset = 0
    for phrase in phrases:
        length = len(phrase.text)
        phrase.fmt = text_format.render(phrase.text, alignment.masks[offset:offset + length])
        offset += length
    return alignment


def attach_formatting_to_groups(groups: list[PhraseGroup], source: StyledText) -> MaskAlignment:
    return attach_formatting(PhraseGroup.flatten_groups(groups), source)
