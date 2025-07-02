from pathlib import Path
from re import Match
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.project import Project


class SoundSegmentFileUtil:

    @staticmethod
    def get_project_sound_segments(project: Project) -> dict[int, str]:
        """
        Returns dict (key = text segment index, value = file path)
        of sound segment file paths in project.
        """

        if not project.dir_path:
            return {}

        audio_segments_path = os.path.join(project.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)
        os.makedirs(Path(audio_segments_path), exist_ok=True)
        file_paths = SoundSegmentFileUtil.get_sound_segment_paths_from_dir(audio_segments_path)

        result = dict[int, str]()
        text_segments = project.text_segments

        for path in file_paths:

            file_name = Path(path).name

            parts = SoundSegmentFileUtil.extract_parts_from_file_name(file_name)
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
            parts = SoundSegmentFileUtil.extract_parts_from_file_name(stem)
            if not parts or not parts[2]:
                return ""
            voice = parts[2]
            if not result:
                result = voice
            elif voice != result:
                return ""
        return result

    @staticmethod
    def get_sound_segment_paths_from_dir(dir_path: str) -> list[str]:
        """
        Returns the file paths in a directory that conform to the app's "sound segment" filename format
        and appear to be valid files
        """
        result = []
        for path in Path(dir_path).iterdir():
            if SoundSegmentFileUtil.is_valid_file_name(path.name) \
                    and path.is_file() \
                    and os.path.getsize(path) > 0:
                result.append( os.path.join(dir_path, path.name) )
        result.sort()
        return result

    @staticmethod
    def is_valid_file_name(file_name: str) -> bool:
        if not file_name.lower().endswith(".flac"):
            return False
        match = AUDIO_SEGMENT_FILE_NAME_PATTERN.fullmatch(file_name)
        return bool(match)

