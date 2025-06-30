import json
import os
import xxhash

from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class HashFileUtil:

    @staticmethod
    def make_segment_file_path(index: int, project: Project) -> str:
        fn = HashFileUtil.make_segment_file_name(
            index, project.text_segments[index], project.get_voice_label()
        )
        return os.path.join(project.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR, fn)

    @staticmethod
    def make_segment_file_name(index: int, text_segment: TextSegment, voice_label: str, suffix=".flac") -> str:
        hash = HashFileUtil.calc_segment_hash(index, text_segment.text)
        one_based_index = index + 1
        s1 = "[" + str(one_based_index).zfill(5) + "] "
        s2 = "[" + hash + "] "
        s3 = "[" + voice_label + "] "
        s4 = sanitize_for_filename(text_segment.text[:50])
        s = s1 + s2 + s3 + s4 + suffix
        return s

    @staticmethod
    def calc_hash(string: str) -> str:
        return xxhash.xxh3_64(string).hexdigest()

    @staticmethod
    def calc_segment_hash(index: int, text: str) -> str:
        """
        Returns a hash for audio segment filename
        Simply hashes the combination of: index value, text
        """
        s = str(index) + " " + text
        return xxhash.xxh3_64(s).hexdigest()

    # ---

    @staticmethod
    def calc_full_hash(text_segments: list[str], voice: dict) -> str:
        s = json.dumps(text_segments, indent=4)
        s += " " + voice.get("hash", "")
        if not "hash" in voice:
            L.w("voice dict is missing hash property")
        return xxhash.xxh3_64(s).hexdigest()

    # ---

    @staticmethod
    def calc_json_object_hash(json_object: dict) -> str:
        json_string = json.dumps(json_object, indent=4)
        return xxhash.xxh3_64(json_string).hexdigest()
