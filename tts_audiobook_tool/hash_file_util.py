import json
import os
from pathlib import Path
from re import Match
from typing import cast
import xxhash

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class HashFileUtil:

    @staticmethod
    def make_segment_file_path(index, state: State) -> str:
        fn = HashFileUtil.make_segment_file_name(index, state.project.text_segments[index], cast(dict, state.project.voice))
        return os.path.join(state.prefs.project_dir, AUDIO_SEGMENTS_SUBDIR, fn)

    @staticmethod
    def make_segment_file_name(index: int, text_segment: str, voice: dict, suffix=".flac") -> str:
        hash = HashFileUtil.calc_segment_hash(index, text_segment, voice)
        voice_id = voice.get("identifier", "voice")
        s1 = "[" + voice_id + "] "
        s2 = "[" + str(index).zfill(5) + "] "
        s3 = "[" + hash + "] "
        s4 = sanitize_for_filename(text_segment)
        s = s1 + s2 + s3 + s4
        s = s[:100]
        s = s.rstrip("_")
        s = s.rstrip(".")
        s += suffix
        return s

    @staticmethod
    def calc_segment_hash(index: int, text_segment: str, voice: dict) -> str:
        """
        Returns a hash for a text/audio segment
        Hashes the combination of: index value, text, and voice dict's precalculated hash value
        """
        s = str(index) + " " + text_segment + " " + voice.get("hash", "")
        if not "hash" in voice:
            L.w("voice dict is missing hash property")
        return xxhash.xxh3_64(s).hexdigest()

    @staticmethod
    def extract_index_and_hash_from_segment_file_name(file_name: str) -> tuple[int, str]:
        match = AUDIO_SEGMENT_FILE_NAME_PATTERN.match(file_name)
        assert isinstance(match, Match)
        index = int(match.group(1))
        hash = match.group(2)
        return index, hash

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
    def get_voice_hash(voice: dict) -> str:
        json_string = json.dumps(voice, indent=4)
        return xxhash.xxh3_64(json_string).hexdigest()
