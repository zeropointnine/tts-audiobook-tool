from __future__ import annotations
import json
import os
import shutil

from tts_audiobook_tool.app_types import SegmentationStrategy, Sound
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.oute_util import OuteUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.tts_model import IndexTts2Protocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

class Project:
    """
    Project settings data-like class, with convenience functions
    """

    dir_path: str

    phrase_groups: list[PhraseGroup] = []
    # The segmentation strategy used to create the PhraseGroups from the source text
    segmentation_strategy: SegmentationStrategy | None = None
    # The max words per segment value used to create the PhraseGroups from the source text
    segmentation_max_words: int = 0
    # The language code used to create the PhraseGroups from the source text (ie, for pysbd)
    segmentation_language_code: str = ""

    section_dividers: list[int] = []
    language_code: str = PROJECT_DEFAULT_LANGUAGE

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

    vibevoice_voice_file_name: str = ""
    vibevoice_model_path: str = ""
    vibevoice_cfg: float = -1
    vibevoice_steps: int = -1

    indextts2_temperature: float = -1
    indextts2_use_fp16: bool = IndexTts2Protocol.DEFAULT_USE_FP16
    indextts2_voice_file_name: str = ""
    indextts2_emo_alpha: float = -1
    indextts2_emo_voice_file_name: str = ""
    indextts2_emo_vector: list[float] = [] # use either 0 or 8 elements

    generate_range_string: str = ""
    
    def __init__(self, dir_path: str):
        self.dir_path = dir_path

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

        from tts_audiobook_tool.project_sound_segments import ProjectSoundSegments
        self.sound_segments = ProjectSoundSegments(self)

    def kill(self) -> None:
        self.sound_segments.observer.stop()

    @staticmethod
    def load_using_dir_path(dir_path: str) -> Project | str:
        """
        Loads project json from directory path and returns parsed project instance.
        Returns error string if json is unviable.
        Else, on some parse errors, falls back to defaults and prints info along the way.
        """

        if not os.path.exists(dir_path):
            return f"Project directory doesn't exist:\n{dir_path}"

        project_dict_path = os.path.join(dir_path, PROJECT_JSON_FILE_NAME)
        try:
            with open(project_dict_path, 'r', encoding='utf-8') as file:
                d = json.load(file)
        except Exception as e:
            return f"Error loading project settings: {e}"

        if not isinstance(d, dict):
            return f"Project settings file bad type: {type(d)}"

        project = Project(dir_path)

        # Text
        if "text" in d:
            lst = d["text"] # is expected to be a list of lists
            result = PhraseGroup.phrase_groups_from_json_list(lst)
            if isinstance(result, str):
                err = result
                printt(f"{COL_ERROR}Error loading project text: {err}")
            else:
                project.phrase_groups = result
        elif "text_segments" in d:
            # Legacy format - list of dicts of "TextSegments"
            lst = d["text_segments"]
            result = Phrase.phrases_from_json_dicts(lst) # this is 'compatible' with the old TextSegment json dicts
            if isinstance(result, str):
                err = result
                printt(f"{COL_ERROR}Error loading project text legacy format: {err}")
            else:
                # Wrap each Phrase (nee TextGroup) in a PhraseGroup
                phrases = result
                project.phrase_groups = []
                for i in range(0, len(phrases)):
                    if i < len(phrases) - 1:
                        # Phrase reason is derived from the text characters _after_ the phrase,
                        # whereas with TextGroup, reason was derived from the text _before_ the phrase
                        reason = phrases[i+1].reason
                    else:
                        reason = Reason.UNDEFINED
                    phrase = Phrase(phrases[i].text, reason)
                    project.phrase_groups.append( PhraseGroup([phrase]) )
        else:
            project.phrase_groups = []

        s = d.get("segmentation_strategy", "")
        project.segmentation_strategy = SegmentationStrategy.from_json_id(s)

        i = d.get("segmentation_max_words", 0)
        if not (i >= 0):
            i = 0
        project.segmentation_max_words = i

        s = d.get("language_code", "")
        if not isinstance(s, str):
            s = ""
        project.language_code = s

        # Chapter indices
        if "chapter_indices" in d:
            lst = d["chapter_indices"]
            is_list_valid = True
            for index in lst:
                is_item_valid = isinstance(index, int) and index >= 0 and index <= len(project.phrase_groups)
                if not is_item_valid:
                    is_list_valid = False
                    break
            if not is_list_valid:
                printt(f"File cut points invalid: {lst}")
            else:
                project.section_dividers = lst

        # Generate-range string
        # TODO: should validate and set to empty if invalid
        s = d.get("generate_range", "")
        if s == "all" or s == "a":
            s = ""
        project.generate_range_string = s

        # Oute
        project.oute_voice_file_name = d.get("oute_voice_file_name", "")
        project.oute_temperature = d.get("oute_temperature", -1)

        if not project.oute_voice_file_name:
            # Pre-existing project has no oute voice set, so set it Oute default
            result = OuteUtil.load_oute_voice_json(OUTE_DEFAULT_VOICE_JSON_FILE_PATH)
            if isinstance(result, str):
                AskUtil.ask_error(result) # not ideal
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


        # TODO: need validation logic for each of these properties (especially file-related ones)

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

        # VibeVoice
        project.vibevoice_voice_file_name = d.get("vibevoice_voice_file_name", "")
        project.vibevoice_model_path = d.get("vibevoice_model_path", "")
        project.vibevoice_cfg = d.get("vibevoice_cfg", -1)
        project.vibevoice_steps = d.get("vibevoice_steps", -1)

        # IndexTTS2
        project.indextts2_voice_file_name = d.get("indextts2_voice_file_name", "")
        project.indextts2_temperature = d.get("indextts2_temperature", -1)
        project.indextts2_use_fp16 = d.get("indextts2_use_fp16", IndexTts2Protocol.DEFAULT_USE_FP16)
        project.indextts2_emo_alpha = d.get("indextts2_emo_alpha", -1)
        if project.indextts2_emo_alpha == -1 and d.get("indextts2_emo_voice_alpha", -1) >= 0:
            project.indextts2_emo_alpha = d.get("indextts2_emo_voice_alpha", -1) # legacy support
        project.indextts2_emo_voice_file_name = d.get("indextts2_emo_voice_file_name", "")

        o = d.get("indextts2_emo_vector", [])
        if not isinstance(o, list):
            o = []
        project.indextts2_emo_vector = o

        return project

    def save(self) -> None:

        if self.segmentation_strategy:
            seg_string = self.segmentation_strategy.json_id
        else:
            seg_string = ""

        d = {

            "dir_path": self.dir_path,

            "text": PhraseGroup.phrase_groups_to_json_list(self.phrase_groups),
            "segmentation_strategy": seg_string,
            "segmentation_max_words": self.segmentation_max_words,
            "segmentation_language_code": self.segmentation_language_code,

            "language_code": self.language_code,

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
            "higgs_temperature": self.higgs_temperature,

            "vibevoice_voice_file_name": self.vibevoice_voice_file_name,
            "vibevoice_model_path": self.vibevoice_model_path,
            "vibevoice_cfg": self.vibevoice_cfg,
            "vibevoice_steps": self.vibevoice_steps,

            "indextts2_voice_file_name": self.indextts2_voice_file_name,
            "indextts2_temperature": self.indextts2_temperature,
            "indextts2_emo_voice_file_name": self.indextts2_emo_voice_file_name,
            "indextts2_emo_vector": self.indextts2_emo_vector,
            "indextts2_temperature": self.indextts2_temperature,
            "indextts2_use_fp16": self.indextts2_use_fp16
        }

        file_path = os.path.join(self.dir_path, PROJECT_JSON_FILE_NAME)
        try:
            with open(file_path, "w") as file:
                json.dump(d, file, indent=4)
        except Exception as e:
            L.e(f"Save error: {e}") # TODO: need to handle this

    def set_phrase_groups_and_save(
            self,
            phrase_groups: list[PhraseGroup],
            strategy: SegmentationStrategy,
            max_words: int,
            language_code: str,
            raw_text: str
    ) -> None:

        self.phrase_groups = phrase_groups
        self.segmentation_strategy = strategy
        self.segmentation_max_words = max_words
        self.segmentation_language_code = language_code

        # Setting this invalidates some things
        self.section_dividers =[]
        self.generate_range_string = ""

        self.save()
        self.save_raw_text(raw_text) # saved for reference

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

    def set_voice_and_save(
            self,
            source_sound: Sound,
            voice_file_stem: str,
            text: str,
            tts_type: TtsModelInfos,
            is_secondary: bool=False
    ) -> str:
        """
        Saves resampled/peak-normalized voice sound file, and updates and saves project properties
        Returns error string on fail
        """

        # Resample to model's native samplerate
        target_sr = tts_type.value.sample_rate
        sound = SoundUtil.resample_if_necessary(source_sound, target_sr)
        # Peak normalization
        sound = Sound( SoundUtil.normalize(sound.data), sound.sr )

        # Add "_modelname" to filename
        dest_file_name = f"{voice_file_stem}_{tts_type.value.file_tag}.flac"
        dest_path = Path(self.dir_path) / dest_file_name
        # Save flac file to project dir
        err = SoundFileUtil.save_flac(sound, str(dest_path))
        if err:
            return err

        # Update the correct voice file name property of project, and save
        match tts_type:
            case TtsModelInfos.CHATTERBOX:
                self.chatterbox_voice_file_name = dest_file_name
                # Rem, chatterbox does not require voice sound file's transcription
            case TtsModelInfos.FISH:
                self.fish_voice_file_name = dest_file_name
                self.fish_voice_transcript = text
            case TtsModelInfos.HIGGS:
                self.higgs_voice_file_name = dest_file_name
                self.higgs_voice_transcript = text
            case TtsModelInfos.VIBEVOICE:
                self.vibevoice_voice_file_name = dest_file_name
            case TtsModelInfos.INDEXTTS2:
                if not is_secondary:
                    self.indextts2_voice_file_name = dest_file_name
                else:
                    self.indextts2_emo_voice_file_name = dest_file_name
            case _:
                raise Exception(f"Unsupported tts type {tts_type}")

        self.save()
        return ""

    def set_oute_voice_and_save(self, voice_dict: dict, dest_file_stem: str) -> None:
        file_name = dest_file_stem + ".json"
        err = save_json(voice_dict, os.path.join(self.dir_path, file_name))
        if err:
            AskUtil.ask_error(err)
            return
        self.oute_voice_file_name = file_name
        self.oute_voice_json = voice_dict
        self.save()

    def clear_voice_and_save(self, tts_type: TtsModelInfos, is_secondary: bool=False) -> None:
        match tts_type:
            case TtsModelInfos.CHATTERBOX:
                self.chatterbox_voice_file_name = ""
            case TtsModelInfos.FISH:
                self.fish_voice_file_name = ""
                self.fish_voice_transcript = ""
            case TtsModelInfos.HIGGS:
                self.higgs_voice_file_name = ""
                self.higgs_voice_transcript = ""
            case TtsModelInfos.VIBEVOICE:
                self.vibevoice_voice_file_name = ""
            case TtsModelInfos.INDEXTTS2:
                if not is_secondary:
                    self.indextts2_voice_file_name = ""
                else:
                    self.indextts2_emo_voice_file_name = ""
            case _:
                raise ValueError(f"Unsupported tts_type: {tts_type}")
        self.save()

    def get_voice_label(self, is_secondary: bool=False) -> str:

        def make_label(file_name: str) -> str:
            label = Path(file_name).stem
            # Strip "_model" from end of file stem
            label = label.removesuffix("_" + Tts.get_type().value.file_tag)
            label = label[:30]
            label = sanitize_for_filename(label)
            return label

        match Tts.get_type():
            case TtsModelInfos.OUTE:
                if self.can_voice:
                    return make_label(self.oute_voice_file_name)
                else:
                    return "none" # shouldn't happen
            case TtsModelInfos.CHATTERBOX:
                if not self.chatterbox_voice_file_name:
                    return "none"
                else:
                    return make_label(self.chatterbox_voice_file_name)
            case TtsModelInfos.FISH:
                if not self.fish_voice_file_name:
                    return "none"
                else:
                    return make_label(self.fish_voice_file_name)
            case TtsModelInfos.HIGGS:
                if not self.higgs_voice_file_name:
                    return "none"
                else:
                    return make_label(self.higgs_voice_file_name)
            case TtsModelInfos.VIBEVOICE:
                if not self.vibevoice_voice_file_name:
                    return "none"
                else:
                    return make_label(self.vibevoice_voice_file_name)
            case TtsModelInfos.INDEXTTS2:
                if not is_secondary:
                    if not self.indextts2_voice_file_name:
                        return "none"
                    else:
                        return make_label(self.indextts2_voice_file_name)
                else:
                    if not self.indextts2_emo_voice_file_name:
                        return "none"
                    else:
                        return make_label(self.indextts2_emo_voice_file_name)
            case TtsModelInfos.NONE:
                return "none"

    @property
    def can_voice(self) -> bool:
        """
        Returns True if current state allows for outputting a "voice" of any kind.
        """
        match Tts.get_type():
            case TtsModelInfos.OUTE:
                # must have oute json file
                return bool(self.oute_voice_json)
            case TtsModelInfos.CHATTERBOX:
                # always true bc does not require voice sample
                return True
            case TtsModelInfos.FISH:
                # always true bc does not require voice sample
                return True
            case TtsModelInfos.HIGGS:
                # always true bc does not require voice sample
                return True
            case TtsModelInfos.VIBEVOICE:
                # requires voice sample (in my implementation, for reasons)
                return bool(self.vibevoice_voice_file_name)
            case TtsModelInfos.INDEXTTS2:
                # this model requires a voice sample
                return bool(self.indextts2_voice_file_name)
            case TtsModelInfos.NONE:
                return False

    @property
    def can_generate_audio(self) -> bool:
        return self.can_voice and len(self.phrase_groups) > 0

    @property
    def sound_segments_dir_path(self) -> str:
        if not self.dir_path:
            return "" # TODO this feels off, smth abt project not yet having a dir_path, etc
        return os.path.join(self.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)

    def get_indices_to_generate(self) -> set[int]:
        """
        Returns the set of indices to be generated,
        derived from the (human readable) "generate_range_string"
        """
        range_string = self.generate_range_string
        is_all = not range_string or range_string == "all" or range_string == "a"
        if is_all:
            result = set(range(len(self.phrase_groups)))
        else:
            result, _ = ParseUtil.parse_ranges_string(range_string, len(self.phrase_groups))
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

    @staticmethod
    def parse_emo_vector_string(string: str) -> list[float] | str:
        """
        Returns error string on parse fail
        Returns empty list to represent list of all zeroes
        """

        string = string.strip()
        if not string:
            return []
        if string.lower() == "none":
            return []

        strings = string.split(",")
        if len(strings) != 8:
            return "Requires 8 comma-delimited numbers between 0-1"

        floats = []
        for string in strings:
            try:
                flt = float(string)
            except:
                return f"Bad value: {string} - must be a number between 0-1"
            if not (0 <= flt <= 1):
                return f"Out of range: {flt} - must be between 0-1"
            floats.append(flt)
        return floats

    def emo_vector_to_string(self) -> str:
        if not self.indextts2_emo_vector or sum(self.indextts2_emo_vector) == 0:
            return "none"
        strings = []
        for item in self.indextts2_emo_vector:
            string = f"{item:.1f}".replace(".0", "") # either one or no decimal point
            strings.append(string)
        return ",".join(strings)

    def migrate_from(self, source_project: Project) -> None:
        """ 
        Copies settings from pre-existing project except directory path, and also copies 'active' internal files.
        Returns error string on fail 
        """

        # Overwrite all "data" attributes except dir_path; and ofc skip any helper objects 
        BLACKLIST = ["dir_path", "sound_segments"]
        source_vars = vars(source_project)        
        for attr_name, attr_value in source_vars.items():
            if not attr_name in BLACKLIST:
                setattr(self, attr_name, attr_value)

        # Copy 'internal' files; fail silently
        src_files = [PROJECT_TEXT_RAW_FILE_NAME]
        src_files.append(source_project.chatterbox_voice_file_name)
        src_files.append(source_project.fish_voice_file_name)
        src_files.append(source_project.higgs_voice_file_name)
        src_files.append(source_project.vibevoice_voice_file_name)
        src_files.append(source_project.indextts2_voice_file_name)
        src_files.append(source_project.indextts2_emo_voice_file_name)
        src_files = [file for file in src_files if file]

        for src_file in src_files:
            src_path = os.path.join(source_project.dir_path, src_file)
            if not Path(src_path).exists():
                continue 
            dest_path = os.path.join(self.dir_path, src_file)
            try:
                shutil.copy(src_path, dest_path)
            except Exception as e:
                ... # eat silently

        self.save()
