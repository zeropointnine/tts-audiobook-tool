import base64
import json
import zlib
from mutagen.flac import FLAC, FLACNoHeaderError, FLACVorbisError

from tts_audiobook_tool.l import L
from tts_audiobook_tool.text_segment import TextSegment, TimedTextSegment

class FlacMetaUtil:

    APP_METADATA_FIELD = "TTS_AUDIOBOOK_TOOL"

    @staticmethod
    def set_app_metadata(flac_path: str, raw_text: str, timed_text_segments: list[TimedTextSegment]) -> str:
        """
        Returns error string on fail
        """

        # Convert raw text to base64'ed compressed data blob
        bytes = raw_text.encode('utf-8')
        data = zlib.compress(bytes, level=6)
        raw_text_base64 = base64.urlsafe_b64encode(data).decode('ascii')

        # TODO: text segments need to be properly escaped or maybe even blob-ized
        dic = {
            "raw_text": raw_text_base64,
            "text_segments": TimedTextSegment.to_dict_list(timed_text_segments)
        }
        s = json.dumps(dic)

        error = FlacMetaUtil.set_metadata(flac_path, FlacMetaUtil.APP_METADATA_FIELD, s)
        return error

    @staticmethod
    def get_app_metadata(flac_path: str) -> tuple[str, list[TextSegment]] | None:
        """
        Returns tuple of [0] raw text, [1] list of TextSegments
        Or None on fail
        """
        string = FlacMetaUtil.get_metadata(flac_path, FlacMetaUtil.APP_METADATA_FIELD)
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
            L.e(f"Missing field: {o}")
            return None

        lst = o["text_segments"]
        if not isinstance(lst, list):
            L.e(f"Bad type for text_segments: {type(lst)}")
            return None

        # Decode the base64 string to bytes, decompress, and convert back to utf8
        s = o["raw_text"]
        data = base64.urlsafe_b64decode(s.encode('ascii'))
        bytes_data = zlib.decompress(data)
        raw_text = bytes_data.decode('utf-8')

        text_segments = []
        for item in lst:
            try:
                text_segment = TextSegment(**item)
            except Exception as e:
                L.e("error with item: {item} {e}")
                return None
            text_segments.append(text_segment)

        return raw_text, text_segments

    # ---

    @staticmethod
    def get_metadata(path: str, field_name: str) -> str:

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
    def set_metadata(file_path: str, field_name: str, value: str) -> str:
        """
        Adds string metadata to a FLAC file
        Returns error message on fail
        """

        # Normalize field name to uppercase (FLAC standard)
        field_name = field_name.upper()

        try:
            flac = FLAC(file_path)

            if field_name in flac:
                del flac[field_name]

            # Mutagen requires list even for single values
            flac[field_name] = [value]

            # Save changes
            flac.save()
            return ""

        except (FLACNoHeaderError, FLACVorbisError) as e:
            # Not a valid FLAC file or corrupt metadata
            return str(e)
        except IOError as e:
            # File access problems
            return str(e)
        except Exception as e:
            return str(e)

    # ---

    @staticmethod
    def get_duration(path: str) -> float | None:
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
