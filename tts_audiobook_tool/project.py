from __future__ import annotations
import json
import os
from typing import Any
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.text_segment import TextSegment

class Project:
    """
    """

    def __init__(
            self,
            dir_path: str = "",
            voice: dict | None = None,
            text_segments: list[TextSegment] = [],
            chapter_indices: list[int] = []
    ) -> None:
        self.dir_path = dir_path
        self.voice = voice
        self._text_segments = text_segments
        self._section_dividers: list[int] = chapter_indices

    @staticmethod
    def load(dir_path: str) -> Project:
        """
        Loads project state from files in a directory
        """

        # Defaults
        voice = None
        text_segments = []
        chapter_indices = []

        # Load voice from file
        from tts_audiobook_tool.voice_util import VoiceUtil
        voice_path = os.path.join(dir_path, PROJECT_VOICE_FILE_NAME)
        result = VoiceUtil.load_voice(voice_path)
        if isinstance(result, dict):
            voice = result

        # Load settings json for the other values
        settings_dict = None
        settings_path = os.path.join(dir_path, PROJECT_SETTINGS_FILE_NAME)
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as file:
                    settings_dict = json.load(file)
            except Exception as e:
                L.e(f"Error loading settings: {e}")

        if settings_dict:
            if not isinstance(settings_dict, dict):
                L.e(f"Project settings file bad type: {type(settings_dict)}")
            else:
                # Text segments
                if not "text_segments" in settings_dict:
                    L.e("Missing text_segments")
                else:
                    l = settings_dict["text_segments"]
                    text_segments = Project._extract_text_segments(l)
                # Chapter indices
                if not "chapter_indices" in settings_dict:
                    L.e("Missing chapter_indices")
                else:
                    lst = settings_dict["chapter_indices"]
                    is_list_valid = True
                    for index in lst:
                        is_item_valid = isinstance(index, int) and index >= 0 and index <= len(text_segments)
                        if not is_item_valid:
                            is_list_valid = False
                            L.e(f"Bad value for chapter index: {index}")
                            break
                    if is_list_valid:
                        chapter_indices = lst

        return Project(dir_path, voice, text_segments, chapter_indices)

    def _save(self) -> None:
        """ Saves project settings (sans voice) to concomitant files in project directory """

        dict = {
            "text_segments": TextSegment.to_dict_list(self._text_segments),
            "chapter_indices": self._section_dividers
        }
        file_path = os.path.join(self.dir_path, PROJECT_SETTINGS_FILE_NAME)
        try:
            with open(file_path, "w") as file:
                json.dump(dict, file, indent=4)
        except Exception as e:
            L.e(f"Save error: {e}")

    @property
    def text_segments(self) -> list[TextSegment]:
        return self._text_segments

    def set_text_segments(self, text_segments: list[TextSegment], raw_text: str) -> None:
        self._text_segments = text_segments
        # Setting text segments invalidates section dividers
        self._section_dividers =[]
        self._save()
        # Save raw text as well for reference
        self._save_raw_text(raw_text)

    def _save_raw_text(self, raw_text: str) -> None:
        file_path = os.path.join(self.dir_path, PROJECT_RAW_TEXT_FILE_NAME)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(raw_text)
        except Exception as e:
            L.e(f"Error saving raw text: {e}") # TODO need to return error

    def load_raw_text(self) -> str:
        file_path = os.path.join(self.dir_path, PROJECT_RAW_TEXT_FILE_NAME)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            L.e(f"Error saving raw text: {e}") # TODO need to return error
            return ""

    @property
    def section_dividers(self) -> list[int]:
        return self._section_dividers

    @section_dividers.setter
    def section_dividers(self, value: list[int]) -> None:
        self._section_dividers = value
        self._save()

    @staticmethod
    def _extract_text_segments(object: list[dict]) -> list[TextSegment]:
        if not object or not isinstance(object, list):
            L.e(f"bad text segments object: {object}")
            return []
        text_segments = []
        for item in object:
            if not isinstance(item, dict):
                L.e(f"bad type: {item}")
                return []
            if not "text" in item or not "index_start" in item or not "index_end" in item:
                L.e(f"missing required property in item: {item}")
                return []
            try:
                start = int(item["index_start"])
                end = int(item["index_end"])
            except:
                L.e(f"parse float error: {item}")
                return []
            text_segment = TextSegment(item["text"], start, end)
            text_segments.append(text_segment)
        return text_segments
