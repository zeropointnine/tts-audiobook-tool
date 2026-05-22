import os
import platform

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.util import *


def get_from_text_file(
        max_words: int,
        segmentation_strategy: SegmentationStrategy,
        pysbd_language: str,
        prefs: Prefs,
) -> tuple[list[PhraseGroup], str, str]:
    """
    Ask the user for a text file path and return phrase groups plus raw text.

    Side effects:
        Shows feedback if no text segments are returned.
        Updates prefs.last_text_dir if a file is opened.
    """
    if prefs.last_text_dir and os.path.exists(prefs.last_text_dir):
        initial_dir = prefs.last_text_dir
    else:
        initial_dir = ""

    path = ask.ask_file_path(
        "Enter text file path: ", "Select text file", initialdir=initial_dir
    )
    if not path:
        return [], "", ""
    if not os.path.exists(path):
        ask.ask_error("No such file")
        return [], "", ""

    try:
        with open(path, "r", encoding="utf-8") as file:
            raw_text = file.read()
    except Exception as e:
        ask.ask_error(f"Error: {e}")
        return [], "", ""

    prefs.last_text_dir = str(Path(path).parent)

    print(f"{COL_DIM_ITALICS}Segmenting text... ", end="", flush=True)
    phrase_groups = PhraseGrouper.text_to_groups(
        raw_text,
        pysbd_lang=pysbd_language,
        max_words=max_words,
        strategy=segmentation_strategy,
    )
    print(f"\r{Ansi.ERASE_REST_OF_LINE}", end="", flush=True)

    if not phrase_groups:
        ask.ask_enter_to_continue("No text segments.")
        return [], raw_text, ""

    title = os.path.splitext(os.path.basename(path))[0]
    return phrase_groups, raw_text, title


def get_from_std_in(
        max_words: int,
        segmentation_strategy: SegmentationStrategy,
        pysbd_language: str,
) -> tuple[list[PhraseGroup], str]:
    """
    Ask the user to input or paste text and return phrase groups plus raw text.
    """
    printt("Enter/paste text of any length.")
    if platform.system() == "Windows":
        s = f"Finish with {COL_ACCENT}[CTRL-Z + ENTER] {COL_DEFAULT}on its own line"
    else:
        s = "Finish with [ENTER + CTRL-D]"
    printt(s)
    printt()
    raw_text = ask.ask_multiline()
    printt()
    if not raw_text:
        return [], ""

    phrase_groups = PhraseGrouper.text_to_groups(
        raw_text,
        pysbd_lang=pysbd_language,
        max_words=max_words,
        strategy=segmentation_strategy,
    )

    if not phrase_groups:
        ask.ask_enter_to_continue("No text segments.")
        return [], raw_text

    return phrase_groups, raw_text

