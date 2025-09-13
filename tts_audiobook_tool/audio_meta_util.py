from pathlib import Path
from typing import cast
import shutil
import mutagen.mp4
from mutagen.mp4 import MP4, MP4Tags, MP4FreeForm, AtomDataType
from mutagen.flac import FLAC, FLACNoHeaderError, FLACVorbisError

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.util import *

class AudioMetaUtil:

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


    # TODO: add optional dest file path, matching mp4 version
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
            return make_error_string(e)

        return "" # success


    @staticmethod
    def get_mp4_metadata_tag(mp4_path: str, mean: str, tag: str) -> tuple[str, str]:
        """
        Returns content and error message, mutually exclusive
        """

        tag_key = f"----:{mean}:{tag}"

        try:
            audio = MP4(mp4_path)
            if audio.tags is None or tag_key not in audio.tags:
                return "", f"Tag '{tag_key}' not found in {mp4_path}"

            freeform_tag_list = audio.tags[tag_key]
            if not freeform_tag_list:
                return "", f"Tag '{tag_key}' is empty in {mp4_path}"

            utf8_bytes = freeform_tag_list[0]
            string = str(utf8_bytes, encoding='utf-8')
        except (FileNotFoundError, KeyError, UnicodeDecodeError, Exception) as e:
            return "", make_error_string(e)

        return string, ""

    @staticmethod
    def set_mp4_metadata_tag(src_path: str, mean: str, tag: str, value: str, dest_path: str = "") -> str:
        """
        Adds tag to pre-existing or to copied mp4 file
        Returns error string on fail
        """

        tag_key = f"----:{mean}:{tag}"

        try:
            utf8_bytes = value.encode('utf-8')
        except UnicodeEncodeError as e:
            return make_error_string(e)

        try:
            file_to_modify = dest_path if dest_path else src_path
            if dest_path and src_path != dest_path:
                shutil.copy2(src_path, dest_path)
            mp4 = MP4(file_to_modify)
        except mutagen.mp4.MP4StreamInfoError as e:
            return f"Couldn't open mp4 file with mutagen: {e}"
        except FileNotFoundError:
            return f"File not found: {src_path}"

        if mp4.tags is None:
            mp4.add_tags()

        tags = cast(MP4Tags, mp4.tags)
        tags[tag_key] = MP4FreeForm(utf8_bytes, dataformat=AtomDataType.UTF8)
        mp4.save(file_to_modify)

        try:
            mp4.save()
        except Exception as e:
            return f"{e}"

        return ""

    @staticmethod
    def get_audio_duration(path: str) -> float | None:
        """
        Returns the duration in seconds of a FLAC or MP4 file, or None
        """
        suffix = Path(path).suffix.lower()
        if suffix == ".flac":
            try:
                flac = FLAC(path)
            except Exception as e:
                L.e(f"{e}")
                return None
            if flac.info:
                return flac.info.length
            else:
                return None
        elif suffix in AAC_SUFFIXES:
            try:
                mp4 = MP4(path)
            except Exception as e:
                L.e(f"{e}")
                return None
            if mp4.info:
                return mp4.info.length
            else:
                return None
        else:
            return None
