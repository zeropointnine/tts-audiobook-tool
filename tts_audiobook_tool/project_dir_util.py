import os
from pathlib import Path
from re import Match

from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class ProjectDirUtil:
    """ Functions for getting audio segment files in the project directory """

    @staticmethod
    def check_dir_valid(project_dir: str) -> str:
        """ Returns error feedback text or empty string if success """

        if not os.path.exists(project_dir):
            return f"Doesn't exist: {project_dir}"

        items = os.listdir(project_dir)

        # Empty directory is considered valid
        if not items:
            return ""

        # Directory with a voice and/or text json file considered valid
        if PROJECT_JSON_FILE_NAME in items:
            return ""

        return f"{project_dir} does not appear to be a project directory"

    @staticmethod
    def has_generated_any(project: Project) -> bool:
        dic = ProjectDirUtil.get_items(project)
        return bool(dic)

    @staticmethod
    def num_generated(project: Project) -> int:
        dic = ProjectDirUtil.get_items(project)
        return len( list( dic.keys() ) )

    @staticmethod
    def count_num_generated_in(project: Project, set_: set[int]) -> int:
        dic = ProjectDirUtil.get_items(project)
        count = 0
        for key in dic.keys():
            if key in set_:
                count += 1
        return count

    @staticmethod
    def get_items_with_tag(project: Project, tag: str) -> dict[int, str]:
        """
        "tag"
            is a so-called "bracket tag"
            (app nomenclature for strings like "[fail]" inserted in filename)
        """
        tag = tag.lstrip("[")
        tag = tag.rstrip("]")
        tag = f"[{tag}]"

        items = ProjectDirUtil.get_items(project)
        result = { index: path for index, path in items.items() if tag in path }
        return result

    @staticmethod
    def get_items(project: Project) -> dict[int, str]:
        """
        Returns dict (key = text segment index, value = file path)
        of valid project audio files found in the project directory.
        """

        if not project.dir_path:
            return {}

        audio_segments_path = os.path.join(project.dir_path, AUDIO_SEGMENTS_SUBDIR)
        os.makedirs(Path(audio_segments_path), exist_ok=True)
        file_paths = ProjectDirUtil._get_all_paths(audio_segments_path)

        result = dict[int, str]()
        text_segments = project.text_segments

        for path in file_paths:

            file_name = Path(path).name

            parts = ProjectDirUtil.extract_parts_from_file_name(file_name)
            if parts is None:
                continue

            index, hash, _ = parts
            if index >= len(text_segments):
                continue

            text_segment = text_segments[index]
            segment_hash = HashFileUtil.calc_segment_hash(index, text_segment.text)
            if hash != segment_hash:
                continue

            if index in result:
                # in case of duplicate
                if "[fail]" in path:
                    continue

            result[index] = str(path)

        return result

    @staticmethod
    def extract_parts_from_file_name(file_name: str) -> tuple[int, str, str] | None:
        """ Returns segment index, hash value, and voice label """
        try:
            match = AUDIO_SEGMENT_FILE_NAME_PATTERN.match(file_name)
            assert isinstance(match, Match)
            one_based_index = int(match.group(1))
            segment_index = one_based_index - 1
            hash = match.group(2)
            voice_label = str(match.group(3))
            return segment_index, hash, voice_label
        except Exception as e:
            return None


    @staticmethod
    def get_common_voice_label(paths: list[str]) -> str:
        """
        """
        result = ""
        for path in paths:
            stem = Path(path).stem
            parts = ProjectDirUtil.extract_parts_from_file_name(stem)
            if not parts or not parts[2]:
                return ""
            voice = parts[2]
            if not result:
                result = voice
            elif voice != result:
                return ""
        return result

    @staticmethod
    def _get_all_paths(dir_path: str) -> list[str]:
        """
        Returns the file paths of audio files in directory that have app's "audio segment" filename format.
        Does not filter by content hash etc.
        """
        result = []
        for path in Path(dir_path).iterdir():
            if path.suffix != ".flac":
                continue
            if not path.is_file():
                continue
            if os.path.getsize(path) == 0:
                continue
            match = AUDIO_SEGMENT_FILE_NAME_PATTERN.fullmatch(path.name) # TODO make function for this
            if not match:
                continue
            result.append( os.path.join(dir_path, path.name) )

        result.sort()
        return result
