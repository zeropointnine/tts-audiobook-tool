from __future__ import annotations
from pathlib import Path
from typing import NamedTuple
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.phrase import PhraseGroup
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *


class SoundSegmentUtil:
    """
    Logic for managing sound segment files
    """

    @staticmethod
    def get_project_sound_segments(project: Project) -> dict[int, str]:
        """
        Returns dict (key = text segment index, value = file path)
        of sound segment file paths in project.
        """

        if not project.dir_path:
            return {}
        if not os.path.exists(project.sound_segments_dir_path):
            return {}

        result = dict[int, str]()
        groups = project.phrase_groups

        for path in Path(project.sound_segments_dir_path).iterdir():

            if not path.is_file():
                continue
            if os.path.getsize(path) == 0:
                continue
            if path.suffix.lower() != ".flac":
                continue

            parts = SoundSegmentUtil.extract_parts_from_file_name(path.name)

            if parts is None:
                continue
            if parts.zb_index >= len(groups):
                continue

            text_group = groups[parts.zb_index]
            segment_hash = SoundSegmentUtil.calc_segment_hash(parts.zb_index, text_group.text)
            if parts.hash != segment_hash:
                continue

            if parts.zb_index in result:
                # in case of duplicate
                if "fail" in parts:
                    continue

            result[parts.zb_index] = str(path)

        return result

    @staticmethod
    def extract_parts_from_file_name(file_name: str) -> FileNameParts | None:
        """
        Must be either:
            [index, hash, voice, ...] <-- legacy format
            [index, hash, model, voice, ...]
        """
        tags = extract_tags_from_file_name(file_name)
        if not tags:
            return None

        if len(tags) < 3:
            return None

        try:
            one_based_index = int(tags[0])
        except:
            return None
        if one_based_index < 1:
            return None
        zero_based_index = one_based_index - 1

        hash = tags[1]
        if not AppUtil.is_app_hash(hash):
            return None

        if tags[2] in TtsModelInfos.all_file_tags():
            # Must be [index, hash, model, voice, ...]
            model = tags[2]
            if len(tags) < 4:
                return None
            voice = tags[3]
        else:
            # Must be [index, hash, voice, ...]
            voice = tags[2]
            model = ""

        return FileNameParts(zb_index=zero_based_index, hash=hash, voice=voice, model=model)


    @staticmethod
    def get_common_model_tag(paths: list[str]) -> str:
        """
        """
        result = ""
        for path in paths:
            stem = Path(path).stem
            parts = SoundSegmentUtil.extract_parts_from_file_name(stem)
            if not parts or not parts.model:
                return ""
            if not result:
                result = parts.model
            elif parts.model != result:
                return ""
        return result

    @staticmethod
    def get_common_voice_tag(paths: list[str]) -> str:
        """
        """
        result = ""
        for path in paths:
            stem = Path(path).stem
            parts = SoundSegmentUtil.extract_parts_from_file_name(stem)
            if not parts or not parts.voice:
                return ""
            if not result:
                result = parts.voice
            elif parts.voice != result:
                return ""
        return result

    @staticmethod
    def is_valid_file_name(file_name: str) -> bool:
        if not file_name.lower().endswith(".flac"):
            return False
        match = AUDIO_SEGMENT_FILE_NAME_PATTERN.fullmatch(file_name)
        return bool(match)

    @staticmethod
    def make_segment_file_path(index: int, project: Project) -> str:
        fn = SoundSegmentUtil.make_file_name(
            index=index,
            phrase_group=project.phrase_groups[index],
            model_tag=Tts.get_type().value.file_tag,
            voice_tag=project.get_voice_label()
        )
        return os.path.join(project.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR, fn)

    @staticmethod
    def make_file_name(
        index: int, phrase_group: PhraseGroup, model_tag: str, voice_tag: str, suffix=".flac"
    ) -> str:
        index_tag = "[" + str(index + 1).zfill(5) + "]" # one-based-index
        hash_tag = "[" + SoundSegmentUtil.calc_segment_hash(index, phrase_group.text) + "]"
        model_tag = "[" + model_tag + "]"
        voice_tag = "[" + voice_tag + "]"
        sanitized_text = sanitize_for_filename(phrase_group.presentable_text[:50])
        s = f"{index_tag} {hash_tag} {model_tag} {voice_tag} {sanitized_text}{suffix}"
        return s

    @staticmethod
    def calc_segment_hash(index: int, text: str) -> str:
        """
        Returns a hash for the sound segment filename
        Simply hashes the combination of: index value and text
        """
        s = str(index) + " " + text
        return AppUtil.calc_hash_string(s)

# ---

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

# ---

class FileNameParts(NamedTuple):
    zb_index: int # zero-based index
    hash: str
    voice: str
    model: str
