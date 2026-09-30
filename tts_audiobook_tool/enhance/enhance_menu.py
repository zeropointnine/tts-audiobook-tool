from __future__ import annotations

"""Presentation for the resumable enhance-audiobook workflow."""

from pathlib import Path

from tts_audiobook_tool import text_util
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM
from tts_audiobook_tool.enhance import enhance_flow
from tts_audiobook_tool.enhance.enhance_text import flatten_book
from tts_audiobook_tool.enhance.enhance_artifacts import EnhanceState, make_enhance_state_cached
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import ellipsize_path_for_menu, make_currently_string


INTRO = (
    f"This feature allows you to combine the audio of a pre-existing audiobook\n"
    f"with its original source text to create an {COL_ACCENT}.abr.m4a{COL_DEFAULT} / {COL_ACCENT}.abr.m4b{COL_DEFAULT}\n"
    f"audiobook file which can be played using the tts-audiobook-tool\n"
    f"player/reader, just like the TTS audiobooks created by the main app."
)


def audio_status(snapshot: EnhanceState) -> str:
    if not snapshot.has_audio_path or snapshot.artifacts is None:
        return ""
    path = snapshot.artifacts.audio_path
    value = ellipsize_path_for_menu(str(path))
    if snapshot.audio_exists:
        value = text_util.make_terminal_hyperlink(str(path), value, is_file=True)
    return make_currently_string(value)


def text_status(snapshot: EnhanceState) -> str:
    book = snapshot.book
    if book is None:
        return ""
    count = len(flatten_book(book).phrases)
    noun = "line" if count == 1 else "lines"
    return make_currently_string(f"{count} {noun}")


def output_status(snapshot: EnhanceState) -> str:
    path = snapshot.expected_output_path
    if path is None or not snapshot.output_exists:
        return ""
    value = ellipsize_path_for_menu(str(path))
    value = text_util.make_terminal_hyperlink(str(path), value, is_file=True)
    return f"{COL_DIM}({COL_ACCENT}{value}{COL_DIM})"


def output_suffix(snapshot: EnhanceState) -> str:
    if snapshot.book is not None and snapshot.book.text_source_kind == "plain_text":
        return ".abr.m4a"
    return ".abr.m4b"


def _checkbox(done: bool) -> str:
    return ("✅" if done else "⬜") + " "


def build_items(state: State, snapshot: EnhanceState) -> list[MenuItem]:
    audio_value = audio_status(snapshot)
    text_value = text_status(snapshot)
    suffix = output_suffix(snapshot)

    # The items form two visually grouped input tracks that converge on the
    # output step: the audio track (enter audio -> transcribe) and the text
    # track (enter text -> align). Alignment consumes the transcription, so
    # the audio track must complete before the text track can finish.
    items = [
        MenuItem(
            _checkbox(snapshot.audio_exists)
            + " ".join(filter(None, ["Enter source audiobook file path", audio_value])),
            lambda _, __: enhance_flow.select_audio(state),
        ),
        MenuItem(
            _checkbox(snapshot.transcription_valid) + "Transcribe",
            lambda _, __: enhance_flow.transcribe(state),
        ),
        MenuItem(
            _checkbox(snapshot.book is not None)
            + " ".join(filter(None, ["Enter text or EPUB file path", text_value])),
            lambda _, __: enhance_flow.select_text(state),
            blank_line_before=True,
        ),
        MenuItem(
            # An alignment built from a book that is no longer present cannot
            # be used to create output, so show it as incomplete.
            _checkbox(snapshot.timed_phrases_valid and snapshot.book is not None)
            + "Align source text with transcription",
            lambda _, __: enhance_flow.align_source_text(state),
        ),
        MenuItem(
            _checkbox(snapshot.output_valid)
            + " ".join(filter(None, [f"Create the \"{suffix}\" file", output_status(snapshot)])),
            lambda _, __: enhance_flow.create_output(state),
            blank_line_before=True,
        ),
    ]

    optional_items: list[MenuItem] = []
    if snapshot.output_valid:
        count = snapshot.misalignment_count
        total = snapshot.output_phrase_count
        optional_items.append(
            MenuItem(
                f"Review unmatched lines {COL_DIM}({count} of {total} lines){COL_DEFAULT}",
                lambda _, __: enhance_flow.review_discontinuities(state),
            )
        )
    if snapshot.has_audio_path:
        optional_items.append(
            MenuItem(
                "Clear",
                lambda _, __: enhance_flow.clear_selection(state),
            )
        )
    if optional_items:
        optional_items[0].superlabel = "Options"
        optional_items[0].superlabel_no_blank_line = True
        optional_items[0].blank_line_before = True
    items.extend(optional_items)
    return items


def menu(state: State) -> None:
    # Drop a stale selection whose audio file no longer exists.
    current = state.prefs.enhance_audio_path
    if current and not Path(current).is_file():
        state.prefs.enhance_audio_path = ""
        state.prefs.save()

    def make_items(_: State) -> list[MenuItem]:
        # One defensive filesystem snapshot drives every label and visibility
        # decision for this render; operations revalidate before doing work.
        # The cached variant skips re-deriving when no source file changed.
        snapshot = make_enhance_state_cached(state.prefs.enhance_audio_path)
        return build_items(state, snapshot)

    MenuUtil.menu(
        state,
        "Enhance a pre-existing audiobook",
        make_items,
        subheading=INTRO,
        breadcrumb="Enhance audiobook",
    )
