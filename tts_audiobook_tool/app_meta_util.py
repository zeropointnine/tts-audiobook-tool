import base64
import json
from pathlib import Path
import zlib

from tts_audiobook_tool.app_types import *
from tts_audiobook_tool.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.timed_text_segment import TimedTextSegment

class AppMetaUtil:

    @staticmethod
    def get_app_metadata(path: str) -> AppMetadata | None:
        suffix = Path(path).suffix.lower()
        if suffix == ".flac":
            return AppMetaUtil.get_app_metadata_flac(path)
        elif suffix in AAC_SUFFIXES:
            return AppMetaUtil.get_app_metadata_mp4(path)

    @staticmethod
    def get_app_metadata_flac(path: str) -> AppMetadata | None:
        """
        TODO: untested
        """
        string = AudioMetaUtil.get_flac_metadata_field(path, APP_META_FLAC_FIELD)
        if not string:
            return None
        result = AppMetaUtil.get_app_metadata_from_json_string(string)
        if isinstance(result, str):
            return None
        else:
            return result

    @staticmethod
    def set_flac_app_metadata(path: str, raw_text: str, timed_text_segments: list[TimedTextSegment]) -> str:
        """
        Returns error string on fail
        """
        string = AppMetaUtil._make_app_metadata_json_string(raw_text, timed_text_segments)
        error = AudioMetaUtil.set_flac_custom_metadata_field(path, APP_META_FLAC_FIELD, string)
        return error

    @staticmethod
    def get_app_metadata_mp4(path: str) -> AppMetadata | None:
        string, error = AudioMetaUtil.get_mp4_metadata_tag(path, APP_META_MP4_MEAN, APP_META_MP4_TAG)
        if error:
            return None
        result = AppMetaUtil.get_app_metadata_from_json_string(string)
        if isinstance(result, str):
            return None
        return result

    @staticmethod
    def get_app_metadata_from_json_string(json_string: str) -> AppMetadata | str:
        """
        Returns tuple or error string.
        Does some validation.

        TODO: untested
        """

        try:
            o = json.loads(json_string)
        except Exception as e:
            return f"{e}"
        if not isinstance(o, dict):
            return f"Bad type: {type(o)}"
        if not "raw_text" in o or not "text_segments" in o:
            return f"Missing top-level field in {o}"

        # raw_text - Decode base64 string to bytes, decompress, and convert back to utf8
        s = o["raw_text"]
        data = base64.urlsafe_b64decode(s.encode('ascii'))
        bytes_data = zlib.decompress(data)
        raw_text = bytes_data.decode('utf-8')

        text_segment_dicts = o["text_segments"]
        if not isinstance(text_segment_dicts, list):
            return f"Bad type for text_segments: {type(text_segment_dicts)}"
        if not text_segment_dicts:
            return f"text_segments list is empty"
        result = TimedTextSegment.list_from_dict_list(text_segment_dicts)
        if isinstance(result, str):
            return result
        timed_text_segments = result

        return AppMetadata(raw_text, timed_text_segments)


    @staticmethod
    def set_mp4_app_metadata(
        path: str, raw_text: str, timed_text_segments: list[TimedTextSegment], dest_path: str=""
     ) -> str:
        """
        Returns error string on fail
        """
        string = AppMetaUtil._make_app_metadata_json_string(raw_text, timed_text_segments)
        error = AudioMetaUtil.set_mp4_metadata_tag(
            path, APP_META_MP4_MEAN, APP_META_MP4_TAG, string, dest_path
        )
        return error

    @staticmethod
    def _make_app_metadata_json_string(raw_text: str, timed_text_segments: list[TimedTextSegment]) -> str:

        # Convert raw text to base64'ed compressed data blob
        bytes = raw_text.encode('utf-8')
        data = zlib.compress(bytes, level=6)
        raw_text_base64 = base64.urlsafe_b64encode(data).decode('ascii')

        dic = {
            "raw_text": raw_text_base64,
            "text_segments": TimedTextSegment.to_dict_list(timed_text_segments)
        }
        string = json.dumps(dic)
        return string
