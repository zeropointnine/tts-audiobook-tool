import numpy as np

from tts_audiobook_tool.app_types import ConcreteSegment
from tts_audiobook_tool.conversation.prompt_draft import PromptDraft


def segment(text: str):
    return [ConcreteSegment(0.0, 1.0, text, [])]


def test_prompt_draft_add_select_delete_and_finalize() -> None:
    draft = PromptDraft()
    assert draft.add_transcription(
        segment(" hello "), np.array([0.1], dtype=np.float32)
    )
    assert draft.add_phrase("world", np.array([0.2], dtype=np.float32))
    assert draft.text == "hello world"
    # The highlight was on the last chunk, so it followed "world".
    assert draft.selected_index == 1
    draft.select_previous()
    assert draft.selected_index == 0
    assert draft.delete_selected()
    assert draft.text == "world"
    assert draft.selected_index == 0
    submission = draft.finalize()
    assert submission is not None
    assert submission.text == "world"
    assert submission.audio is not None
    assert np.isclose(np.max(np.abs(submission.audio)), 1.0)
    assert draft.is_empty


def test_add_phrase_follows_tail_and_keeps_moved_selection() -> None:
    draft = PromptDraft()
    assert draft.add_phrase("first")
    assert draft.add_phrase("second")
    # The highlight sat on the last chunk, so it followed each append.
    assert draft.selected_index == 1
    draft.select_previous()
    assert draft.selected_index == 0
    assert draft.add_phrase("third")
    # The user moved off the tail, so the highlight stays on "first"
    # instead of jumping to the newest chunk.
    assert draft.selected_index == 0
    assert draft.text == "first second third"


def test_prompt_draft_copies_audio_and_ignores_blank_input() -> None:
    draft = PromptDraft()
    audio = np.array([0.1, 0.2], dtype=np.float32)
    assert draft.add_transcription(segment("phrase"), audio)
    audio[0] = 9.0
    assert draft.audio_chunks[0] is not None
    assert draft.audio_chunks[0][0] == np.float32(0.1)
    assert not draft.add_transcription(segment("  "))


def test_empty_prompt_draft_does_not_finalize() -> None:
    draft = PromptDraft()
    assert draft.finalize() is None
    assert not draft.delete_selected()
