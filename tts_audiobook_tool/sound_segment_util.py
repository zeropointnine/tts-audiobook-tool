from __future__ import annotations
from pathlib import Path
from typing import NamedTuple
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.phrase import PhraseGroup
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.tts_model import TtsModelInfo, TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.validation_result import MusicFailResult, ValidationResult, WordErrorResult


class SoundSegmentUtil:
    """
    Logic for managing sound segment files 
    (generated audio for a chunk of project text, saved as a file in the project dir)
    """

    @staticmethod
    def make_sound_segments_map(project: Project) -> dict[int, list[SoundSegment]]:
        """
        """

        if not project.dir_path:
            return {}
        if not os.path.exists(project.sound_segments_path):
            return {}

        map = dict[int, list[SoundSegment]]()

        for path in Path(project.sound_segments_path).iterdir():

            if not path.is_file():
                continue
            if os.path.getsize(path) == 0:
                continue
            if path.suffix.lower() != ".flac":
                continue

            sound_segment = SoundSegment.from_file_name(path.name)
            if sound_segment is None:
                continue
            index = sound_segment.idx
            if index >= len(project.phrase_groups):
                continue

            phrase_group = project.phrase_groups[sound_segment.idx]
            segment_hash = SoundSegmentUtil.calc_segment_hash(index, phrase_group.text)
            if sound_segment.hash != segment_hash:
                continue

            if not map.get(index, []):
                map[sound_segment.idx] = []
            map[index].append(sound_segment)

        return map

    @staticmethod
    def make_file_name(
        index: int,
        phrase_group: PhraseGroup,
        project: Project,
        tts_model_info: TtsModelInfo,
        validation_result: ValidationResult,
        is_real_time: bool,
        suffix=".flac",
        is_debug_json: bool=False
    ) -> str:

        idx = str(index + 1).zfill(5)
        model = tts_model_info.file_tag
        voice = project.get_voice_label()
        text = " " + TextUtil.sanitize_for_filename(phrase_group.presentable_text[:50])
        
        if isinstance(validation_result, WordErrorResult):
            num_fails_tag = f" [{validation_result.num_errors}]" 
        elif isinstance(validation_result, MusicFailResult):
            num_fails_tag = f" [99]" # good enough for now
        else:
            num_fails_tag = ""
        
        fail_tag = " [fail]" if validation_result.is_fail else ""

        if is_debug_json:
            suffix = ".debug.json"

        if is_real_time:
            timestamp = SoundSegmentUtil.make_timestamp_string()
            path = f"[{timestamp}] [{idx}] [{model}] [{voice}]{num_fails_tag}{fail_tag}{text}{suffix}"
        else:
            hash_string = SoundSegmentUtil.calc_segment_hash(index, phrase_group.text)
            path = f"[{idx}] [{hash_string}] [{model}] [{voice}]{num_fails_tag}{fail_tag}{text}{suffix}"

        return path
        
    @staticmethod
    def get_common_model_tag(paths: list[str]) -> str:
        result = ""
        for path in paths:
            sound_segment = SoundSegment.from_file_name(Path(path).name)
            if not sound_segment or not sound_segment.model:
                return ""
            if not result:
                result = sound_segment.model
            elif sound_segment.model != result:
                return ""
        return result

    @staticmethod
    def get_common_voice_tag(paths: list[str]) -> str:
        result = ""
        for path in paths:
            sound_segment = SoundSegment.from_file_name(Path(path).name)
            if not sound_segment or not sound_segment.voice:
                return ""
            if not result:
                result = sound_segment.voice
            elif sound_segment.voice != result:
                return ""
        return result

    @staticmethod
    def calc_segment_hash(index: int, text: str) -> str:
        """
        Returns a hash for the sound segment filename
        Simply hashes the combination of: index value and text
        """
        s = str(index) + " " + text
        return AppUtil.calc_hash_string(s)

    @staticmethod
    def extract_tags_from_file_name(file_name: str) -> list[str] | None:
        """
        Extracts tags from a file name string.

        A file_name must look like this to be valid: "[tag1] [tag2] [tag3] optionally-anything-else",
        where "tag" is a series of characters without whitespace enclosed in square brackets (eg, "[hello_1234]").
        There must be one or more so-called tags, and they must occur consecutively at the beginning of the string.
        Anything after the last identified tag is permitted and can be ignored.

        Args:
            file_name: The string of the file name to process.

        Returns:
            A list of the extracted tag contents (e.g., ["tag1", "tag2"]) if the file_name is valid,
            otherwise returns None.
        """
        # A regular expression for a single valid tag.
        # It must start with '[', end with ']', and contain one or more
        # characters that are NOT whitespace or brackets.
        tag_pattern = re.compile(r"\[([^\[\]\s]+)\]")

        tags_found = []
        # We work on a copy of the string, consuming it from the left.
        # We strip leading whitespace to handle cases like " [tag1]..."
        remaining_name = file_name.lstrip()

        # Loop as long as the remaining string could potentially start with a tag.
        while remaining_name.startswith('['):
            # Attempt to match a single tag at the beginning of the current string.
            match = tag_pattern.match(remaining_name)

            if not match:
                # The string starts with '[' but is not a valid tag
                # (e.g., "[tag with space]", "[]", "[malformed").
                # Since tags must be consecutive, we stop processing.
                break

            # If a valid tag is matched, add its content to our list.
            # group(1) captures the content inside the brackets.
            tags_found.append(match.group(1))

            # "Consume" the matched tag from the string.
            # We also strip leading whitespace to prepare for the next tag
            # or to separate the tags from the rest of the file name.
            remaining_name = remaining_name[match.end():].lstrip()

        # According to the rules, there must be one or more tags.
        # If our list is empty after the loop, the file_name is invalid.
        if not tags_found:
            return None

        return tags_found
    
    @staticmethod
    def make_timestamp_string() -> str:
        # "0" ensures debug sound files shows up alphabetically before sound non-debug files in segments dir, yes rly
        return "0" + str(int(time.time() * 1000)) 

# ---

class SoundSegment(NamedTuple):
    """
    Extracts the data from the so-called tags from a sound segment filename
    """

    # The pre-existing file_name from which the SoundSegment fields were extracted
    file_name: str

    # Zero-based index
    idx: int 
    # Hash of the source text used to generate the sound file
    hash: str
    num_errors: int
    is_fail: bool
    model: str
    voice: str

    @staticmethod
    def from_file_name(file_name: str) -> SoundSegment | None:
        """
        Expecting:
            [0] index, [1] hash, [2] model, [3] voice, [4] num-fails (optional), ...
        Eg:
            [00024] [3ae0f21b9de65a3c] [vibevoice] [sy_even_if_ch1_c] [5] [fail] With_Lord_knows_what_s_beyond.flac
        """

        # In case argument is a full file path
        file_name = Path(file_name).name 

        tags = SoundSegmentUtil.extract_tags_from_file_name(file_name)
        if not tags:
            return None

        if len(tags) < 2:
            # Index and hash are required
            return None

        try:
            idx_1b = int(tags[0])
        except:
            return None
        if idx_1b < 1:
            return None
        index_0b = idx_1b - 1

        hash = tags[1]
        if not AppUtil.is_app_hash(hash):
            return None

        model = tags[2] if len(tags) >= 3 and tags[2] in TtsModelInfos.all_file_tags() else ""
        voice = tags[3] if len(tags) >= 4 else ""
        num_errors = int(tags[4]) if len(tags) >= 5 and tags[4].isdigit() else -1
        is_fail = ("[fail]" in file_name)

        return SoundSegment(
            file_name=file_name, idx=index_0b, hash=hash, is_fail=is_fail, 
            voice=voice, model=model, num_errors=num_errors
        )

