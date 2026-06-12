from dataclasses import dataclass
import os
from pathlib import Path

from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.text_ops.epub_extractor import EpubExtractor, EpubImportResult
from tts_audiobook_tool.util import *


@dataclass
class EpubPromptMessages:
    console_message: str
    dialog_title: str


class EpubMenuUtil:

    DEFAULT_PROMPT_MESSAGES = EpubPromptMessages(
        console_message="Enter EPUB file path: ",
        dialog_title="Select EPUB file",
    )

    @staticmethod
    def ask_epub_path(
            prefs: Prefs,
            prompt_messages: EpubPromptMessages | None = None,
    ) -> str:
        prompt_messages = prompt_messages or EpubMenuUtil.DEFAULT_PROMPT_MESSAGES
        if prefs.last_text_dir and os.path.exists(prefs.last_text_dir):
            initial_dir = prefs.last_text_dir
        else:
            initial_dir = ""
        path = ask.ask_file_path(
            prompt_messages.console_message,
            prompt_messages.dialog_title,
            filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        if not path:
            return ""
        if not os.path.exists(path):
            ask.ask_error("No such file")
            return ""
        if os.path.splitext(path)[1].lower() != ".epub":
            ask.ask_error("Must select an .epub file")
            return ""
        prefs.last_text_dir = str(Path(path).parent)
        return path

    @staticmethod
    def import_epub(
            epub_path: str,
            max_words: int,
            segmentation_strategy: SegmentationStrategy,
            language_code: str,
    ) -> EpubImportResult | None:
        try:
            printt(f"{COL_DIM_ITALICS}Importing epub file... ")
            return EpubExtractor.import_epub(
                epub_path=epub_path,
                max_words=max_words,
                segmentation_strategy=segmentation_strategy,
                language_code=language_code,
            )
        except ImportError as e:
            printt()
            ask.ask_error(str(e))
            return None
        except Exception as e:
            printt()
            ask.ask_error(f"Error importing EPUB: {e}")
            return None

    @staticmethod
    def print_import_info(epub_import_result: EpubImportResult) -> None:
        printt()
        printt(f"{COL_ACCENT}Import info:{COL_DEFAULT}")
        for warning in epub_import_result.significant_warnings:
            printt(f"- {warning}")
        num_sections = len(epub_import_result.chapters)
        noun = make_noun("EPUB section", "EPUB sections", num_sections)
        printt(f"- Imported {COL_ACCENT}{num_sections}{COL_DEFAULT} {noun} using the EPUB's built-in structure.")
        printt()

    @staticmethod
    def make_text_file_path(epub_path: str) -> str:
        return make_unique_file_path(str(Path(epub_path).with_suffix(".txt")))

    @staticmethod
    def save_text_file(text: str, file_path: str) -> str:
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
        except Exception as e:
            return f"Error writing text file: {e}"
        return ""

    @staticmethod
    def print_text_file_created(file_path: str) -> None:
        link = text_util.make_terminal_hyperlink(file_path, file_path, is_file=True)
        printt(f"Text file created at {link}")
        printt()
