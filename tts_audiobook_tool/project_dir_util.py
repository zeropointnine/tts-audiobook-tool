import os
from pathlib import Path
from re import Match
from typing import cast

from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.state import State
from .util import *
from .constants import *

class ProjectDirUtil:
    """ Methods that get or infer state from the contents of the project directory """

    @staticmethod
    def check_project_dir_valid(project_dir: str) -> str:
        """ Returns error feedback text or empty string if success """

        if not os.path.exists(project_dir):
            return f"Doesn't exist: {project_dir}"

        items = os.listdir(project_dir)

        # Empty directory is considered valid
        if not items:
            return ""

        # Directory with a voice and/or text json file considered valid
        if PROJECT_VOICE_FILE_NAME in items or PROJECT_TEXT_FILE_NAME in items:
            return ""

        return f"{project_dir} does not appear to be a project directory"

    @staticmethod
    def has_audio_segment_files(state: State) -> bool:
        dic = ProjectDirUtil.get_project_audio_segment_file_paths(state)
        return bool(dic)

    @staticmethod
    def num_audio_segment_files(state: State) -> int:
        dic = ProjectDirUtil.get_project_audio_segment_file_paths(state)
        return len( list( dic.keys() ) )

    @staticmethod
    def get_project_audio_segment_file_paths(state: State) -> dict[int, str]:
        """
        Returns dict (key = text segment index, value = file path)
        of valid audio files in the project directory.
        """
        if not state.project_dir:
            return {}

        paths = ProjectDirUtil._get_all_audio_segment_file_paths(state.project_dir)
        # print("paths", paths)

        result = dict[int, str]()
        text_segments = state.text_segments

        for path in paths:

            file_name = Path(path).name
            # print("fn", file_name)

            # Note we do not simply compare against the full file name,
            # because the full file name includes the text segment in massaged form,
            # which we don't rly want to treat as 'idempotent'
            index_from_file_name, hash_from_file_name = HashFileUtil.extract_index_and_hash_from_segment_file_name(file_name)

            if index_from_file_name >= len(text_segments):
                continue

            text_segment = text_segments[index_from_file_name]
            segment_hash = HashFileUtil.calc_segment_hash(index_from_file_name, text_segment, cast(dict, state.voice))

            if hash_from_file_name != segment_hash:
                continue
            if index_from_file_name in result:
                continue

            result[index_from_file_name] = str(path)

        return result

    @staticmethod
    def _get_all_audio_segment_file_paths(project_dir: str) -> list[str]:
        """
        Returns the file paths of all audio segment files in project dir.
        Filters out audio files that don't have the "audio segment" filename format.
        Does not filter by content hash etc.
        """
        result = []
        for file in Path(project_dir).iterdir():
            if not file.name.endswith(".flac"):
                continue
            if not file.is_file():
                continue
            if  not os.path.getsize( os.path.join(project_dir, file) ):
                continue

            match = AUDIO_SEGMENT_FILE_NAME_PATTERN.fullmatch(file.name)
            if not match:
                continue

            result.append( os.path.join(project_dir, file.name) )

        result.sort()
        return result
