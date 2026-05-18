from __future__ import annotations

import base64
import json
from pathlib import Path
import zlib

from tts_audiobook_tool.app_types import *
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.util import *

class AppMetadata(NamedTuple):
    """
    Metadata of the app-generated audio file.
    Plus serialization util functions.
    """

    # The list of Phrases that make up the audiobook text, including timing info
    timed_phrases: list[TimedPhrase]

    # ABR metadata spec version; missing in old files implies version 1
    version: int

    # Bookmark indices
    bookmark_indices: list[int]

    # The unmassaged input text 
    raw_text: str

    # Is True when the audio includes page turn sound effect at so-called section breaks
    has_section_break_audio: bool | None

    # Export-time snapshot of project settings for possible future import flows
    project_snapshot: dict

    def to_json_string(self) -> str:

        # Convert raw text to base64'ed compressed data blob
        bytes = self.raw_text.encode('utf-8')
        data = zlib.compress(bytes, level=6)
        raw_text_base64 = base64.urlsafe_b64encode(data).decode('ascii')

        dic = {
            "version": self.version,
            "raw_text": raw_text_base64,
            "bookmarks": self.bookmark_indices,
            "text_segments": TimedPhrase.timed_phrases_to_dicts(self.timed_phrases),
            "has_section_break_audio": bool(self.has_section_break_audio),
            "project_snapshot": self.project_snapshot
        }
        string = json.dumps(dic)
        return string

    @staticmethod
    def get_from_json_string(json_string: str) -> AppMetadata | str:
        """
        Returns tuple or error string.
        Does some validation.
        """

        try:
            o = json.loads(json_string)
        except Exception as e:
            return f"{e}"
        if not isinstance(o, dict):
            return f"Bad type: {type(o)}"
        if not "raw_text" in o or not "text_segments" in o:
            return f"Missing required field in {o}"

        version = o.get("version", 1)
        if not isinstance(version, int) or version < 1:
            return f"Bad type/value for 'version': {version}"

        # raw_text - Decode base64 string to bytes, decompress, and convert back to utf8
        s = o["raw_text"]
        data = base64.urlsafe_b64decode(s.encode('ascii'))
        bytes_data = zlib.decompress(data)
        raw_text = bytes_data.decode('utf-8')

        phrase_dicts = o["text_segments"]
        if not isinstance(phrase_dicts, list):
            return f"Bad type for 'text_segments': {type(phrase_dicts)}"
        if not phrase_dicts:
            return f"text_segments list is empty"
        result = TimedPhrase.dicts_to_timed_phrases(phrase_dicts)
        if isinstance(result, str):
            return result
        timed_phrases = result

        bookmarks = o.get("bookmarks", [])
        if not isinstance(bookmarks, list):
            return f"Bad type for 'bookmarks': {type(bookmarks)}"
        try:
            bookmarks = [int(item) for item in bookmarks]
        except Exception as e:
            return "Bad item in bookmarks: " + make_error_string(e)

        has_section_break_audio = o.get("has_section_break_audio", False)

        project_snapshot = o.get("project_snapshot", {})
        if not isinstance(project_snapshot, dict):
            return f"Bad type for 'project_snapshot': {type(project_snapshot)}"

        return AppMetadata(
            timed_phrases=timed_phrases, 
            version=version,
            bookmark_indices=bookmarks,
            raw_text=raw_text, 
            has_section_break_audio=has_section_break_audio,
            project_snapshot=project_snapshot
        )

    @staticmethod
    def load_from_file(path: str) -> AppMetadata | None:
        suffix = Path(path).suffix.lower()
        if suffix == ".flac":
            return AppMetadata.load_from_flac(path)
        elif suffix in AAC_SUFFIXES:
            return AppMetadata.load_from_mp4(path)
        else:
            return None

    @staticmethod
    def load_from_flac(path: str) -> AppMetadata | None:
        string = AudioMetaUtil.get_flac_metadata_field(path, APP_META_FLAC_FIELD)
        if not string:
            return None
        result = AppMetadata.get_from_json_string(string)
        if isinstance(result, str):
            return None
        else:
            return result

    @staticmethod
    def load_from_mp4(path: str) -> AppMetadata | None:
        string, error = AudioMetaUtil.get_mp4_metadata_tag(path, APP_META_MP4_MEAN, APP_META_MP4_TAG)
        if error:
            return None
        result = AppMetadata.get_from_json_string(string)
        if isinstance(result, str):
            return None
        return result

    @staticmethod
    def save_to_flac(app_meta: AppMetadata, src_path: str, dest_path: str="") -> str:
        """
        Returns error string on fail
        """
        string = app_meta.to_json_string()
        error = AudioMetaUtil.set_flac_custom_metadata_field(
            src_path=src_path,
            field_name=APP_META_FLAC_FIELD,
            value=string,
            dest_path=dest_path
        )
        return error

    @staticmethod
    def save_to_mp4(app_meta: AppMetadata, src_path: str,  dest_path: str="") -> str:
        """
        Returns error string on fail
        """
        string = app_meta.to_json_string()
        error = AudioMetaUtil.set_mp4_metadata_tag(
            src_path=src_path,
            mean=APP_META_MP4_MEAN,
            tag=APP_META_MP4_TAG,
            value=string,
            dest_path=dest_path
        )
        return error
