from __future__ import annotations
import json
import os
import shutil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.oute_util import OuteUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.tts_info import TtsType
from tts_audiobook_tool.util import *

class Project:
    """
    """
    dir_path: str

    text_segments: list[TextSegment] = []
    section_dividers: list[int] = []

    oute_voice_file_name: str = ""
    oute_voice_json: dict = {} # is loaded from external file, `oute_voice_file_name`
    oute_temperature: float = -1

    chatterbox_voice_file_name: str = ""
    chatterbox_temperature: float = -1
    chatterbox_cfg: float = -1
    chatterbox_exaggeration: float = -1

    fish_voice_file_name: str = ""
    fish_voice_transcript: str = ""
    fish_temperature: float = -1

    higgs_voice_file_name: str = ""
    higgs_voice_transcript: str = ""
    higgs_temperature: float = -1

    generate_range_string: str = ""


    def __init__(self, dir_path: str):
        self.dir_path = dir_path

        from tts_audiobook_tool.project_sound_segments import ProjectSoundSegments

        if dir_path:
            # Ensure subdir exists
            ss_path = os.path.join(dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)
            if not os.path.exists(ss_path):
                try:
                    os.mkdir(ss_path)
                except:
                    ...
                if not os.path.exists(ss_path):
                    return Exception(f"Couldn't create required subdirectory {ss_path}")

        self.sound_segments = ProjectSoundSegments(self)


    @staticmethod
    def load_using_dir_path(dir_path: str) -> Project | str:
        """
        Loads project json from directory path and returns parsed project instance.
        Returns error string if json is unviable.
        Else, on some parse errors, falls back to defaults and prints info along the way.
        """

        if not os.path.exists(dir_path):
            return f"No such directory: {dir_path}"

        project_dict_path = os.path.join(dir_path, PROJECT_JSON_FILE_NAME)
        try:
            with open(project_dict_path, 'r', encoding='utf-8') as file:
                d = json.load(file)
        except Exception as e:
            return f"Error loading project settings: {e}"

        if not isinstance(d, dict):
            return f"Project settings file bad type: {type(d)}"

        project = Project(dir_path)

        # Text segments
        if "text_segments" in d:
            lst = d["text_segments"]
            project.text_segments = TextSegment.dict_list_to_list(lst)

        # Chapter indices
        if "chapter_indices" in d:
            lst = d["chapter_indices"]
            is_list_valid = True
            for index in lst:
                is_item_valid = isinstance(index, int) and index >= 0 and index <= len(project.text_segments)
                if not is_item_valid:
                    is_list_valid = False
                    break
            if not is_list_valid:
                printt(f"File cut points invalid: {lst}")
            else:
                project.section_dividers = lst

        # Generate range string
        project.generate_range_string = d.get("generate_range", "")

        # Oute
        project.oute_voice_file_name = d.get("oute_voice_file_name", "")
        project.oute_temperature = d.get("oute_temperature", -1)

        if not project.oute_voice_file_name:
            # Pre-existing project has no oute voice set, so set it Oute default
            result = OuteUtil.load_oute_voice_json(OUTE_DEFAULT_VOICE_JSON_FILE_PATH)
            if isinstance(result, str):
                ask_error(result) # not ideal
            else:
                project.set_oute_voice_and_save(result, "default")
        else:
            # Load specified oute voice json file
            voice_path = os.path.join(dir_path, project.oute_voice_file_name)
            result = OuteUtil.load_oute_voice_json(voice_path)
            if isinstance(result, str):
                printt(f"Problem loading Oute voice json file {project.oute_voice_file_name}") # not ideal
            else:
                project.oute_voice_json = result


        # Chatterbox
        project.chatterbox_voice_file_name = d.get("chatterbox_voice_file_name", "")
        project.chatterbox_temperature = d.get("chatterbox_temperature", -1)
        project.chatterbox_cfg = d.get("chatterbox_cfg", -1)
        project.chatterbox_exaggeration = d.get("chatterbox_exaggeration", -1)

        # Fish
        project.fish_voice_file_name = d.get("fish_voice_file_name", "")
        project.fish_voice_transcript = d.get("fish_voice_text", "")
        project.fish_temperature = d.get("fish_temperature", -1)

        # Higgs
        project.higgs_voice_file_name = d.get("higgs_voice_file_name", "")
        project.higgs_voice_transcript = d.get("higgs_voice_text", "")
        project.higgs_temperature = d.get("higgs_temperature", -1)

        return project

    def save(self) -> None:

        d = {
            "dir_path": self.dir_path,
            "text_segments": TextSegment.list_to_dict_list(self.text_segments),
            "chapter_indices": self.section_dividers,
            "generate_range": self.generate_range_string,
            "oute_voice_file_name": self.oute_voice_file_name,
            "oute_temperature": self.oute_temperature,
            "chatterbox_voice_file_name": self.chatterbox_voice_file_name,
            "chatterbox_temperature": self.chatterbox_temperature,
            "chatterbox_cfg": self.chatterbox_cfg,
            "chatterbox_exaggeration": self.chatterbox_exaggeration,
            "fish_voice_file_name": self.fish_voice_file_name,
            "fish_voice_text": self.fish_voice_transcript,
            "fish_temperature": self.fish_temperature,
            "higgs_voice_file_name": self.higgs_voice_file_name,
            "higgs_voice_text": self.higgs_voice_transcript,
            "higgs_temperature": self.higgs_temperature
        }

        file_path = os.path.join(self.dir_path, PROJECT_JSON_FILE_NAME)
        try:
            with open(file_path, "w") as file:
                json.dump(d, file, indent=4)
        except Exception as e:
            L.e(f"Save error: {e}") # TODO: need to handle this

    def set_text_segments_and_save(self, text_segments: list[TextSegment], raw_text: str) -> None:

        self.text_segments = text_segments
        # Setting text segments invalidates some things
        self.section_dividers =[]
        self.generate_range_string = ""
        self.save()
        # Save raw text as well for reference
        self.save_raw_text(raw_text)

    def save_raw_text(self, raw_text: str) -> None:
        file_path = os.path.join(self.dir_path, PROJECT_TEXT_RAW_FILE_NAME)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(raw_text)
        except Exception as e:
            L.e(f"Error saving raw text: {e}") # TODO need to return error

    def load_raw_text(self) -> str:
        file_path = os.path.join(self.dir_path, PROJECT_TEXT_RAW_FILE_NAME)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            L.e(f"Error saving raw text: {e}") # TODO need to return error
            return ""

    def set_oute_voice_and_save(self, voice_json: dict, dest_file_stem: str) -> None:
        file_name = dest_file_stem + ".json"
        err = save_json(voice_json, os.path.join(self.dir_path, file_name))
        if err:
            ask_error(err)
            return
        self.oute_voice_file_name = file_name
        self.oute_voice_json = voice_json
        self.save()

    def set_chatterbox_voice_and_save(self, src_path: str) -> str:
        """ Returns error string on fail """
        source_path = Path(src_path)
        file_name = source_path.name
        dest_path = Path(self.dir_path) / file_name
        if source_path != dest_path:
            try:
                shutil.copy(source_path, dest_path)
            except Exception as e:
                return str(e)
        self.chatterbox_voice_file_name = file_name
        self.save()
        return ""

    def set_fish_voice_and_save(self, src_path: str, text: str) -> str:
        """ Returns error string on fail """
        source_path = Path(src_path)
        file_name = source_path.name
        dest_path = Path(self.dir_path) / file_name
        if source_path != dest_path:
            try:
                shutil.copy(source_path, dest_path)
            except Exception as e:
                return str(e)
        self.fish_voice_file_name = file_name
        self.fish_voice_transcript = text
        self.save()
        return ""

    def clear_fish_voice_and_save(self) -> None:
        self.fish_voice_file_name = ""
        self.fish_voice_transcript = ""
        self.save()

    def set_higgs_voice_and_save(self, src_path: str, text: str) -> str:
        """ Returns error string on fail """
        source_path = Path(src_path)
        file_name = source_path.name
        dest_path = Path(self.dir_path) / file_name
        if source_path != dest_path:
            try:
                shutil.copy(source_path, dest_path)
            except Exception as e:
                return str(e)
        self.higgs_voice_file_name = file_name
        self.higgs_voice_transcript = text
        self.save()
        return ""

    def clear_higgs_voice_and_save(self) -> None:
        self.higgs_voice_file_name = ""
        self.higgs_voice_transcript = ""
        self.save()

    def get_voice_label(self) -> str:
        match Tts.get_type():
            case TtsType.OUTE:
                if self.has_voice:
                    label = Path(self.oute_voice_file_name).stem[:30]
                    label = sanitize_for_filename(label)
                    return label
                else:
                    return "none" # shouldn't happen
            case TtsType.CHATTERBOX:
                if not self.chatterbox_voice_file_name:
                    return "none"
                label = Path(self.chatterbox_voice_file_name).stem[:30]
                label = sanitize_for_filename(label)
                return label
            case TtsType.FISH:
                if not self.fish_voice_file_name:
                    return "none"
                label = Path(self.fish_voice_file_name).stem[:30]
                label = sanitize_for_filename(label)
                return label
            case TtsType.HIGGS:
                if not self.higgs_voice_file_name:
                    return "none"
                label = Path(self.higgs_voice_file_name).stem[:30]
                label = sanitize_for_filename(label)
                return label
            case TtsType.NONE:
                return "none"
            case _:
                return "none"

    @property
    def has_voice(self) -> bool:
        """
        Returns True if current state allows for outputting a "voice" of any kind.
        """
        match Tts.get_type():
            case TtsType.OUTE:
                # must have oute json file
                # (rem, we default to using Oute's own "official" default voice)
                return bool(self.oute_voice_json)
            case TtsType.CHATTERBOX:
                # always true bc does not require voice sample
                return True
            case TtsType.FISH:
                # always true bc does not require voice sample
                return True
            case TtsType.HIGGS:
                # always true bc does not require voice sample # TODO implement/verify
                return True
            case TtsType.NONE:
                return False

    @property
    def can_generate_audio(self) -> bool:
        return self.has_voice and len(self.text_segments) > 0

    @property
    def sound_segments_dir_path(self) -> str:
        if not self.dir_path:
            return "" # TODO this feels off, smth abt project not yet having a dir_path, etc
        return os.path.join(self.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)

    def get_indices_to_generate(self) -> set[int]:
        """
        Returns the set of indices to be generated,
        derived from the user-inputted and human readable "generate_range_string"
        """
        range_string = self.generate_range_string
        is_all = not range_string or range_string == "all" or range_string == "a"
        if is_all:
            result = set(range(len(self.text_segments)))
        else:
            result, _ = ParseUtil.parse_one_indexed_ranges_string(range_string, len(self.text_segments))
        return result

    @staticmethod
    def is_valid_project_dir(project_dir: str) -> str:
        """ Returns error feedback text or empty string if is-valid """

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
