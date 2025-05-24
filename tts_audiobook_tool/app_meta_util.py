import base64
import json
from typing import cast
import zlib
from mutagen.flac import FLAC, FLACNoHeaderError, FLACVorbisError
import mutagen.mp4
from mutagen.mp4 import MP4, MP4Tags, MP4FreeForm, AtomDataType

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.text_segment import TimedTextSegment

class AppMetaUtil:

    @staticmethod
    def set_flac_app_metadata(flac_path: str, raw_text: str, timed_text_segments: list[TimedTextSegment]) -> str:
        """
        Returns error string on fail
        """
        string = AppMetaUtil._make_app_metadata_json_string(raw_text, timed_text_segments)
        error = AppMetaUtil.set_flac_custom_metadata_field(flac_path, APP_META_FLAC_FIELD, string)
        return error

    @staticmethod
    def _make_app_metadata_json_string(raw_text: str, timed_text_segments: list[TimedTextSegment]) -> str:

        # Convert raw text to base64'ed compressed data blob
        bytes = raw_text.encode('utf-8')
        data = zlib.compress(bytes, level=6)
        raw_text_base64 = base64.urlsafe_b64encode(data).decode('ascii')

        # TODO: text segments may yet need to be properly escaped or maybe even blob-ized, maybe alright
        dic = {
            "raw_text": raw_text_base64,
            "text_segments": TimedTextSegment.to_dict_list(timed_text_segments)
        }
        string = json.dumps(dic)
        return string

    @staticmethod
    def set_flac_custom_metadata_field(flac_path: str, field_name: str, value: str) -> str:
        """
        Adds string metadata to a FLAC file
        Returns error message on fail
        """

        # Normalize field name to uppercase (FLAC standard)
        field_name = field_name.upper()

        try:
            flac = FLAC(flac_path)

            if field_name in flac:
                del flac[field_name]

            # Mutagen requires list even for single values
            flac[field_name] = [value]

            # Save changes
            flac.save()

        except (FLACNoHeaderError, FLACVorbisError, IOError, Exception) as e:
            return str(e)

        return "" # success

    @staticmethod
    def get_flac_app_metadata(flac_path: str) -> dict | None:
        """
        Returns dict with
            "raw_text"
            "text_segments" (which are dicts not TimedTextSegment instances)
        Does some validation.
        Returns None on fail.

        TODO: untested
        """
        string = AppMetaUtil.get_flac_metadata_field(flac_path, APP_META_FLAC_FIELD)
        if not string:
            return None
        try:
            o = json.loads(string)
        except Exception as e:
            L.e(f"{e}")
            return None
        if not isinstance(o, dict):
            L.e(f"Bad type: {type(o)}")
            return None
        if not "raw_text" in o or not "text_segments" in o:
            L.e(f"Missing top-level field in {o}")
            return None

        text_segments = o["text_segments"]
        if not isinstance(text_segments, list):
            L.e(f"Bad type for text_segments: {type(text_segments)}")
            return None
        if not text_segments:
            L.e(f"text_segments list is empty")
            return None
        for key in ["index_start", "index_end", "time_start", "time_end"]: # (comes from TimedTextSegment actually)
            if not key in text_segments[0]:
                return L.e(f"Missing field in text_segments element: {key}")

        # Decode the base64 string to bytes, decompress, and convert back to utf8
        s = o["raw_text"]
        data = base64.urlsafe_b64decode(s.encode('ascii'))
        bytes_data = zlib.decompress(data)
        raw_text = bytes_data.decode('utf-8')

        # text_segments = []
        # for item in lst:
        #     try:
        #         text_segment = TextSegment(**item)
        #     except Exception as e:
        #         L.e("error with item: {item} {e}")
        #         return None
        #     text_segments.append(text_segment)

        return {
            "raw_text": raw_text,
            "text_segments": text_segments
        }

    @staticmethod
    def get_flac_metadata_field(path: str, field_name: str) -> str:

        # Normalize field name to uppercase (FLAC standard)
        field_name = field_name.upper()

        try:
            flac = FLAC(path)
        except Exception:
            return ""
        if not flac.tags:
            return ""
        result = flac[field_name][0] if field_name in flac else ""
        return result

    @staticmethod
    def get_flac_duration(path: str) -> float | None:
        """
        Returns the duration in seconds of a FLAC file, or None
        """
        try:
            flac = FLAC(path)
        except Exception as e:
            L.e(f"{e}")
            return None
        if flac.info:
            return flac.info.length
        else:
            return None

    # ---

    @staticmethod
    def set_mp4_metadata_tag(mp4_path: str, mean: str, tag: str, value: str) -> str:
        """
        Returns error string on fail
        """

        tag_key = f"----:{mean}:{tag}"

        try:
            utf8_bytes = value.encode('utf-8')
        except UnicodeEncodeError as e:
            return str(e)

        try:
            mp4 = MP4(mp4_path)
        except mutagen.mp4.MP4StreamInfoError as e:
            return f"Couldn't open mp4 file with mutagen: {e}"
        except FileNotFoundError:
            return f"File not found: {mp4_path}"

        if mp4.tags is None:
            mp4.add_tags()

        tags = cast(MP4Tags, mp4.tags)
        tags[tag_key] = MP4FreeForm(utf8_bytes, dataformat=AtomDataType.UTF8)

        try:
            mp4.save()
        except Exception as e:
            return f"{e}"

        return ""

    @staticmethod
    def get_mp4_metadata_tag(mp4_path: str, mean: str, tag: str) -> str:

        tag_key = f"----:{mean}:{tag}"

        try:
            audio = MP4(mp4_path)
            if audio.tags is None or tag_key not in audio.tags:
                return f"Tag '{tag_key}' not found in {mp4_path}"

            freeform_tag_list = audio.tags[tag_key]
            if not freeform_tag_list:
                return f"Tag '{tag_key}' is empty in {mp4_path}"

            utf8_bytes = freeform_tag_list[0]
            string = str(utf8_bytes, encoding='utf-8')
        except (FileNotFoundError, KeyError, UnicodeDecodeError, Exception) as e:
            return str(e)

        return string
