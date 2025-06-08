from __future__ import annotations
import json
import os
import shutil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.util import *

class Project:
    """
    """
    dir_path: str

    # Note, currently not doing anything with this value,
    # as we are letting a project be used regardless of the runtime's active model
    model_type: str

    text_segments: list[TextSegment] = []
    section_dividers: list[int] = []

    oute_voice_file_name: str = ""
    oute_voice_json: dict = {} # is loaded from external file, `oute_voice_file_name`
    oute_temperature: float = -1

    chatterbox_voice_file_name: str = ""
    chatterbox_temperature: float = -1
    chatterbox_cfg: float = -1
    chatterbox_exaggeration: float = -1

    # Does not persist
    generate_ints: set[int] = set()


    def __init__(self, dir_path: str, model_type: str):
        self.dir_path = dir_path
        self.model_type = model_type

    @staticmethod
    def load(dir_path: str) -> Project | str:
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


        # Note: We allow project to load regardless of original 'model type'. By design?
        model_type = d.get("model_type", "")

        project = Project(dir_path, model_type)

        # Text segments
        if "text_segments" in d:
            lst = d["text_segments"]
            project.text_segments = Project.get_text_segments_from_dict(lst)

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

        # Oute
        project.oute_voice_file_name = d.get("oute_voice_file_name", "")
        project.oute_temperature = d.get("oute_temperature", -1)

        if project.oute_voice_file_name and model_type == "oute":
            from tts_audiobook_tool.voice_util import VoiceUtil
            voice_path = os.path.join(dir_path, project.oute_voice_file_name)
            result = VoiceUtil.load_voice_json(voice_path)
            if isinstance(result, str):
                printt(f"Problem loading Oute voice json file {project.oute_voice_file_name}")
            else:
                project.oute_voice_json = result

        # Chatterbox
        project.chatterbox_voice_file_name = d.get("chatterbox_voice_file_name", "")
        project.chatterbox_temperature = d.get("chatterbox_temperature", -1)
        project.chatterbox_cfg = d.get("chatterbox_cfg", -1)
        project.chatterbox_exaggeration = d.get("chatterbox_exaggeration", -1)

        return project

    def save(self) -> None:

        d = {
            "dir_path": self.dir_path,
            "model_type": self.model_type,
            "text_segments": TextSegment.to_dict_list(self.text_segments),
            "chapter_indices": self.section_dividers,
            "oute_voice_file_name": self.oute_voice_file_name,
            "oute_temperature": self.oute_temperature,
            "chatterbox_voice_file_name": self.chatterbox_voice_file_name,
            "chatterbox_temperature": self.chatterbox_temperature,
            "chatterbox_cfg": self.chatterbox_cfg,
            "chatterbox_exaggeration": self.chatterbox_exaggeration
        }

        file_path = os.path.join(self.dir_path, PROJECT_JSON_FILE_NAME)
        try:
            with open(file_path, "w") as file:
                json.dump(d, file, indent=4)
        except Exception as e:
            L.e(f"Save error: {e}") # TODO: need to handle this

    def set_text_segments_and_save(self, text_segments: list[TextSegment], raw_text: str) -> None:
        self.text_segments = text_segments
        # Setting text segments invalidates section dividers
        self.section_dividers =[]
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

    def set_oute_voice_and_save(self, voice_json: dict, source_file_stem: str) -> None:
        from tts_audiobook_tool.app_util import AppUtil
        file_name = source_file_stem + ".json"
        err = AppUtil.save_json(voice_json, os.path.join(self.dir_path, file_name))
        if err:
            printt(err, "error")
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

    def get_voice_label(self) -> str:
        if Shared.is_oute() and self.has_voice:
            label = Path(self.oute_voice_file_name).stem[:20]
            label = sanitize_for_filename(label)
            return label
        elif Shared.is_chatterbox() and self.chatterbox_voice_file_name:
            label = Path(self.chatterbox_voice_file_name).stem[:20]
            label = sanitize_for_filename(label)
            return label
        else:
            return "None"

    @property
    def is_model_oute(self) -> bool:
        return self.model_type == "oute"

    @property
    def is_model_chatterbox(self) -> bool:
        return self.model_type == "chatterbox"

    @property
    def has_voice(self) -> bool:
        if Shared.is_oute():
            return bool(self.oute_voice_json)
        elif Shared.is_chatterbox():
            return bool(self.chatterbox_voice_file_name)
        else:
            return False

    @property
    def can_generate_audio(self) -> bool:
        return self.has_voice and len(self.text_segments) > 0

    # ---

    @staticmethod
    def get_text_segments_from_dict(object: list[dict]) -> list[TextSegment]:
        if not object or not isinstance(object, list):
            L.e(f"bad text segments object: {object}")
            return []
        text_segments = []
        for item in object:
            if not isinstance(item, dict):
                L.e(f"bad type: {item}")
                return []
            if not "text" in item or not "index_start" in item or not "index_end" in item:
                L.e(f"missing required property in item: {item}")
                return []
            try:
                start = int(item["index_start"])
                end = int(item["index_end"])
            except:
                L.e(f"parse float error: {item}")
                return []
            text_segment = TextSegment(item["text"], start, end)
            text_segments.append(text_segment)
        return text_segments
