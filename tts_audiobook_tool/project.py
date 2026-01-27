from __future__ import annotations
import json
import os
import shutil

from tts_audiobook_tool.app_types import ChapterMode, ExportType, NormalizationType, SegmentationStrategy, Sound, Strictness
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.tts_model.oute_util import OuteUtil
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.tts_model import ChatterboxType, GlmProtocol, IndexTts2Protocol, MiraProtocol, Qwen3Protocol, VibeVoiceProtocol
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *

class Project:
    """
    Project settings data-like class, with convenience functions.
    After changing values, must manually `save()`.
    Big.
    """

    dir_path: str

    language_code: str = PROJECT_DEFAULT_LANGUAGE

    phrase_groups: list[PhraseGroup] = []

    segmentation_strategy: SegmentationStrategy = list(SegmentationStrategy)[0]
    max_words: int = MAX_WORDS_PER_SEGMENT_DEFAULT
    word_substitutions: dict[str, str] = {}

    # The segmentation strategy used to create the PhraseGroups from the source text
    applied_strategy: SegmentationStrategy | None = None
    # The max words per segment value used to create the PhraseGroups from the source text
    applied_max_words: int = 0
    # The language code used to create the PhraseGroups from the source text (ie, for pysbd)
    applied_language_code: str = ""

    generate_range_string: str = ""
    section_dividers: list[int] = []
    subdivide_phrases: bool = False
    export_type: ExportType = list(ExportType)[0]
    use_section_sound_effect: bool = PROJECT_DEFAULT_SECTION_SOUND_EFFECT
    normalization_type: NormalizationType = list(NormalizationType)[0]
    realtime_save: bool = PROJECT_DEFAULT_REALTIME_SAVE
    strictness: Strictness = list(Strictness)[0] 
    max_retries: int = PROJECT_MAX_RETRIES_DEFAULT
    chapter_mode: ChapterMode = list(ChapterMode)[0]

    oute_voice_file_name: str = ""
    oute_voice_json: dict = {} # is loaded from external file, `oute_voice_file_name`
    oute_temperature: float = -1

    chatterbox_type: ChatterboxType = list(ChatterboxType)[0]
    chatterbox_voice_file_name: str = ""
    chatterbox_temperature: float = -1
    chatterbox_cfg: float = -1
    chatterbox_exaggeration: float = -1
    chatterbox_seed: int = -1

    fish_voice_file_name: str = ""
    fish_voice_transcript: str = ""
    fish_temperature: float = -1
    fish_seed: int = -1

    higgs_voice_file_name: str = ""
    higgs_voice_transcript: str = ""
    higgs_temperature: float = -1

    vibevoice_voice_file_name: str = ""
    vibevoice_target: str = ""
    vibevoice_lora_target: str = "" 
    vibevoice_cfg: float = -1
    vibevoice_steps: int = -1
    vibevoice_batch_size: int = 1
    vibevoice_seed: int = -1

    indextts2_temperature: float = -1
    indextts2_use_fp16: bool = IndexTts2Protocol.DEFAULT_USE_FP16
    indextts2_voice_file_name: str = ""
    indextts2_emo_alpha: float = -1
    indextts2_emo_voice_file_name: str = ""
    indextts2_emo_vector: list[float] = [] # use either 0 or 8 elements

    glm_voice_file_name: str = ""
    glm_voice_transcript: str = ""
    glm_sr: int = GlmProtocol.SAMPLE_RATES[0]
    glm_seed: int = -1

    mira_voice_file_name: str = ""
    mira_temperature: float = MiraProtocol.TEMPERATURE_DEFAULT # TODO: should use "-1 pattern"
    mira_batch_size: int = 1

    qwen3_target: str = ""
    qwen3_voice_file_name: str = ""
    qwen3_voice_transcript: str = ""
    qwen3_speaker_id: str = ""
    qwen3_instructions: str = ""
    qwen3_batch_size: int = 1
    qwen3_temperature: float = -1
    qwen3_seed: int = -1

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

        from tts_audiobook_tool.project_util import ProjectUtil

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

        warnings: list[str] = []
        def add_warning(attr_name: str, defaulting_to: Any):
            s = f"Warning: Missing or invalid value for: {attr_name}\n"
            s += "This can occur if a new project property has been added to the app since the last time you opened the project.\n"
            s += f"Setting to default: {defaulting_to}"
            warnings.append(s)

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
                printt()
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

        s = d.get("language_code", "")
        if not isinstance(s, str):
            s = ""
        project.language_code = s

        s = d.get("segmentation_strategy", "")
        project.segmentation_strategy = SegmentationStrategy.from_id(s) or list(SegmentationStrategy)[0]

        i = d.get("max_words", MAX_WORDS_PER_SEGMENT_DEFAULT)
        if not isinstance(i, int) and not isinstance(i, float):
            i = MAX_WORDS_PER_SEGMENT_DEFAULT
        i = int(i)
        if not (MAX_WORDS_PER_SEGMENT_MIN <= i <= MAX_WORDS_PER_SEGMENT_MAX):
            i = MAX_WORDS_PER_SEGMENT_DEFAULT
        project.max_words = i

        s = d.get("word_substitutions_json_string", "")
        result = ProjectUtil.parse_word_substitutions_json_string(s)
        if isinstance(result, str): # error
            project.word_substitutions = {}
        else:
            project.word_substitutions = result

        s = d.get("applied_language_code", "")
        if not isinstance(s, str):
            s = ""
        project.applied_language_code = s

        s = d.get("applied_strategy", "")
        project.applied_strategy = SegmentationStrategy.from_id(s)

        i = d.get("applied_max_words", 0)
        if not (i >= 0):
            i = 0
        project.applied_max_words = i

        # Generate-range string
        # TODO: should validate and set to empty if invalid
        s = d.get("generate_range", "")
        if s == "all" or s == "a":
            s = ""
        project.generate_range_string = s

        # Section dividers / chapter indices
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

        # Subdivide into phrases
        b = d.get("subdivide_phrases", False)
        if not isinstance(b, bool):
            b = False
        project.subdivide_phrases = b

        # Export type
        s = d.get("export_type", "")
        project.export_type = ExportType.get_by_id(s) or list(ExportType)[0]

        # Section break sound effect
        b = d.get("use_section_sound_effect", False)
        if not isinstance(b, bool):
            b = PROJECT_DEFAULT_SECTION_SOUND_EFFECT
        project.use_section_sound_effect = b
        
        # Normalization type
        s = d.get("normalization_type", False)
        project.normalization_type = NormalizationType.from_id(s) or list(NormalizationType)[0]

        # Realtime save output
        b = d.get("realtime_save", False)
        if not isinstance(b, bool):
            b = PROJECT_DEFAULT_REALTIME_SAVE
        project.realtime_save = b

        # Validation strictness
        s = d.get("strictness", "")
        value = Strictness.get_by_id(s)
        if value is None:
            value = Strictness.get_recommended_default(project.language_code)
            add_warning("strictness", value)
        project.strictness = value

        # Max retries
        value = d.get("max_retries", None)
        if not isinstance(value, int) or value < PROJECT_MAX_RETRIES_MIN or value > PROJECT_MAX_RETRIES_MAX:
            add_warning("max_retries", PROJECT_MAX_RETRIES_DEFAULT)
            value = PROJECT_MAX_RETRIES_DEFAULT # note, not using "unset" value of -1 here b/c
        project.max_retries = value

        # Chapter mode
        s = d.get("chapter_mode", "")
        value = ChapterMode.get_by_id(s)
        if value is None:
            value = list(ChapterMode)[0]
            add_warning("chapter_mode", value)
        project.chapter_mode = value

        # Oute
        project.oute_voice_file_name = d.get("oute_voice_file_name", "")
        project.oute_temperature = d.get("oute_temperature", -1)
        if Tts.get_type() == TtsModelInfos.OUTE:
            voice_path = os.path.join(dir_path, project.oute_voice_file_name)
            if not project.oute_voice_file_name or not os.path.exists(voice_path):
                # Set it Oute default
                result = OuteUtil.load_oute_voice_json(OUTE_DEFAULT_VOICE_JSON_FILE_PATH)
                if isinstance(result, str):
                    from tts_audiobook_tool.ask_util import AskUtil
                    AskUtil.ask_error(result) # not ideal
                else:
                    project.set_oute_voice_and_save(result, "default")
            else:
                # Load specified oute voice json file
                result = OuteUtil.load_oute_voice_json(voice_path)
                if isinstance(result, str):
                    printt(f"Problem loading Oute voice json file {project.oute_voice_file_name}: {result}") # not ideal
                    printt()
                else:
                    project.oute_voice_json = result

        # TODO: need validation logic for each of these properties (especially file-related ones)

        # Chatterbox
        s = d.get("chatterbox_type", "")
        chatterbox_type = ChatterboxType.get_by_id(s)
        if not chatterbox_type:
            chatterbox_type = list(ChatterboxType)[0]
            # Don't show warning if current model is not chatterbox
            if Tts.get_type() == TtsModelInfos.CHATTERBOX:
                add_warning("chatterbox_type", chatterbox_type.id)
        project.chatterbox_type = chatterbox_type

        project.chatterbox_voice_file_name = d.get("chatterbox_voice_file_name", "")
        project.chatterbox_temperature = d.get("chatterbox_temperature", -1)
        project.chatterbox_cfg = d.get("chatterbox_cfg", -1)
        project.chatterbox_exaggeration = d.get("chatterbox_exaggeration", -1)
        
        project.chatterbox_seed = d.get("chatterbox_seed", -1)
        if not (-1 <= project.chatterbox_seed <= 2**32 - 1):
            add_warning("chatterbox_seed", -1)
            project.chatterbox_seed = -1

        # Fish
        project.fish_voice_file_name = d.get("fish_voice_file_name", "")
        project.fish_voice_transcript = d.get("fish_voice_text", "")
        project.fish_temperature = d.get("fish_temperature", -1)
        project.fish_seed = d.get("fish_seed", -1)
        if not (-1 <= project.fish_seed <= 2**32 - 1):
            add_warning("fish_seed", -1)
            project.fish_seed = -1

        # Higgs
        project.higgs_voice_file_name = d.get("higgs_voice_file_name", "")
        project.higgs_voice_transcript = d.get("higgs_voice_text", "")
        project.higgs_temperature = d.get("higgs_temperature", -1)

        # VibeVoice
        project.vibevoice_voice_file_name = d.get("vibevoice_voice_file_name", "")
        project.vibevoice_target = d.get("vibevoice_target", "") or d.get("vibevoice_model_path", "") # legacy key compat
        project.vibevoice_lora_target = d.get("vibevoice_lora_path", "")
        project.vibevoice_cfg = d.get("vibevoice_cfg", -1)
        project.vibevoice_steps = d.get("vibevoice_steps", -1)

        value = d.get("vibevoice_batch_size", -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (1 <= value <= PROJECT_BATCH_SIZE_MAX):
                value = PROJECT_BATCH_SIZE_DEFAULT
                add_warning("vibevoice_batch_size", value)
            value = int(value)
        project.vibevoice_batch_size = value

        project.vibevoice_seed = d.get("vibevoice_seed", -1)
        if not (-1 <= project.vibevoice_seed <= 2**32 - 1):
            add_warning("vibevoice_seed", -1)
            project.vibevoice_seed = -1

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

        # GLM
        project.glm_voice_file_name = d.get("glm_voice_file_name", "")
        project.glm_voice_transcript = d.get("glm_voice_text", "")
        project.glm_sr = d.get("glm_sr", 0)
        if not project.glm_sr in GlmProtocol.SAMPLE_RATES:
            project.glm_sr = GlmProtocol.SAMPLE_RATES[0]
        seed = d.get("glm_seed", -1)
        if not isinstance(seed, (int, float)) or not (seed >= -1):
            seed = -1
        project.glm_seed = int(seed)

        # Mira
        project.mira_voice_file_name = d.get("mira_voice_file_name", "")
        
        value = d.get("mira_temperature", -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (MiraProtocol.TEMPERATURE_MIN <= value <= MiraProtocol.TEMPERATURE_MAX):
                value = MiraProtocol.TEMPERATURE_DEFAULT
                add_warning("mira_temperature", value)
        project.mira_temperature = value

        value = d.get("mira_batch_size", -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (1 <= value <= PROJECT_BATCH_SIZE_MAX):
                value = PROJECT_BATCH_SIZE_DEFAULT
                add_warning("mira_batch_size", value)
            value = int(value)
        project.mira_batch_size = value

        # Qwen3-TTS
        project.qwen3_target = d.get("qwen3_target", "") or d.get("qwen3_path_or_id", "") # legacy key compat
        project.qwen3_voice_file_name = d.get("qwen3_voice_file_name", "")
        project.qwen3_voice_transcript = d.get("qwen3_voice_text", "")
        project.qwen3_speaker_id = d.get("qwen3_speaker_id", "")
        project.qwen3_instructions = d.get("qwen3_instructions", "")

        value = d.get("qwen3_batch_size", -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (1 <= value <= PROJECT_BATCH_SIZE_MAX):
                value = PROJECT_BATCH_SIZE_DEFAULT
                add_warning("qwen3_batch_size", value)
            value = int(value)
        project.qwen3_batch_size = value

        value = d.get("qwen3_temperature", -1)
        if value != -1:
            if not isinstance(value, (float, int)) or not (Qwen3Protocol.TEMPERATURE_MIN <= value <= Qwen3Protocol.TEMPERATURE_MAX):
                value = -1
                add_warning("qwen3_temperature", value)
        project.qwen3_temperature = value

        seed = d.get("qwen3_seed", -1)
        if not (-1 <= seed <= 2**32 - 1):
            add_warning("qwen3_seed", -1)
            project.qwen3_seed = -1
        seed = int(seed)
        project.qwen3_seed = seed

        # ---

        if warnings:
            project.save()
            for item in warnings:
                printt(item)
            printt()
            from tts_audiobook_tool.ask_util import AskUtil
            AskUtil.ask_enter_to_continue()

        return project

    def save(self) -> None:

        d = {
            "dir_path": self.dir_path,

            "language_code": self.language_code,

            "text": PhraseGroup.phrase_groups_to_json_list(self.phrase_groups),
            "segmentation_strategy": self.segmentation_strategy.id,
            "max_words": self.max_words,
            "word_substitutions_json_string": json.dumps(self.word_substitutions),
            "applied_language_code": self.applied_language_code,
            "applied_strategy": self.applied_strategy.id if self.applied_strategy else "",
            "applied_max_words": self.applied_max_words,

            "generate_range": self.generate_range_string,
            "chapter_indices": self.section_dividers,
            "subdivide_phrases": self.subdivide_phrases,
            "export_type": self.export_type.id,
            "use_section_sound_effect": self.use_section_sound_effect,
            "normalization_type": self.normalization_type.value.id,
            "realtime_save": self.realtime_save,
            "strictness": self.strictness.id,
            "max_retries": self.max_retries,
            "chapter_mode": self.chapter_mode.id,

            "oute_voice_file_name": self.oute_voice_file_name,
            "oute_temperature": self.oute_temperature,

            "chatterbox_type": self.chatterbox_type.id,
            "chatterbox_voice_file_name": self.chatterbox_voice_file_name,
            "chatterbox_temperature": self.chatterbox_temperature,
            "chatterbox_cfg": self.chatterbox_cfg,
            "chatterbox_exaggeration": self.chatterbox_exaggeration,
            "chatterbox_seed": self.chatterbox_seed,

            "fish_voice_file_name": self.fish_voice_file_name,
            "fish_voice_text": self.fish_voice_transcript,
            "fish_temperature": self.fish_temperature,
            "fish_seed": self.fish_seed,

            "higgs_voice_file_name": self.higgs_voice_file_name,
            "higgs_voice_text": self.higgs_voice_transcript,
            "higgs_temperature": self.higgs_temperature,

            "vibevoice_voice_file_name": self.vibevoice_voice_file_name,
            "vibevoice_target": self.vibevoice_target,
            "vibevoice_lora_path": self.vibevoice_lora_target,
            "vibevoice_cfg": self.vibevoice_cfg,
            "vibevoice_steps": self.vibevoice_steps,
            "vibevoice_batch_size": self.vibevoice_batch_size,
            "vibevoice_seed": self.vibevoice_seed,

            "indextts2_voice_file_name": self.indextts2_voice_file_name,
            "indextts2_temperature": self.indextts2_temperature,
            "indextts2_emo_voice_file_name": self.indextts2_emo_voice_file_name,
            "indextts2_emo_vector": self.indextts2_emo_vector,
            "indextts2_emo_alpha": self.indextts2_emo_alpha,
            "indextts2_temperature": self.indextts2_temperature,
            "indextts2_use_fp16": self.indextts2_use_fp16,

            "glm_voice_file_name": self.glm_voice_file_name,
            "glm_voice_text": self.glm_voice_transcript,
            "glm_sr": self.glm_sr,
            "glm_seed": self.glm_seed,

            "mira_voice_file_name": self.mira_voice_file_name,
            "mira_temperature": self.mira_temperature,
            "mira_batch_size": self.mira_batch_size,

            "qwen3_target": self.qwen3_target,
            "qwen3_voice_file_name": self.qwen3_voice_file_name,
            "qwen3_voice_text": self.qwen3_voice_transcript,
            "qwen3_speaker_id": self.qwen3_speaker_id,
            "qwen3_instructions": self.qwen3_instructions,
            "qwen3_batch_size": self.qwen3_batch_size,
            "qwen3_temperature": self.qwen3_temperature,
            "qwen3_seed": self.qwen3_seed,
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
        self.applied_strategy = strategy
        self.applied_max_words = max_words
        self.applied_language_code = language_code

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
            transcript: str,
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
                self.fish_voice_transcript = transcript
            case TtsModelInfos.HIGGS:
                self.higgs_voice_file_name = dest_file_name
                self.higgs_voice_transcript = transcript
            case TtsModelInfos.VIBEVOICE:
                self.vibevoice_voice_file_name = dest_file_name
            case TtsModelInfos.INDEXTTS2:
                if not is_secondary:
                    self.indextts2_voice_file_name = dest_file_name
                else:
                    self.indextts2_emo_voice_file_name = dest_file_name
            case TtsModelInfos.GLM:
                self.glm_voice_file_name = dest_file_name
                self.glm_voice_transcript = transcript
            case TtsModelInfos.MIRA:
                self.mira_voice_file_name = dest_file_name
            case TtsModelInfos.QWEN3TTS:
                self.qwen3_voice_file_name = dest_file_name
                self.qwen3_voice_transcript = transcript
            case _:
                raise Exception(f"Unsupported tts type {tts_type}")

        self.save()
        return ""

    def set_oute_voice_and_save(self, voice_dict: dict, dest_file_stem: str) -> None:
        file_name = dest_file_stem + ".json"
        err = save_json(voice_dict, os.path.join(self.dir_path, file_name))
        if err:
            from tts_audiobook_tool.ask_util import AskUtil
            AskUtil.ask_error(err)
            return
        self.oute_voice_file_name = file_name
        self.oute_voice_json = voice_dict
        self.save()

    # TODO: Refactor next three methods. Generalize using TtsModel or TtsModelInfos. 
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
            case TtsModelInfos.GLM:
                self.glm_voice_file_name = ""
                self.glm_voice_transcript = ""
            case TtsModelInfos.MIRA:
                self.mira_voice_file_name = ""
            case TtsModelInfos.QWEN3TTS:
                self.qwen3_voice_file_name = ""
            case _:
                raise ValueError(f"Unsupported tts_type: {tts_type}")
        self.save()

    def get_voice_label(self, is_secondary: bool=False) -> str:
        """
        """
        def make_label(file_name: str) -> str:
            label = Path(file_name).stem
            # Strip "_model" from end of file stem
            label = label.removesuffix("_" + Tts.get_type().value.file_tag)
            label = TextUtil.sanitize_for_filename(label)
            label = label[:30]
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
            case TtsModelInfos.GLM:
                if not self.glm_voice_file_name:
                    return "none"
                else:
                    return make_label(self.glm_voice_file_name)
            case TtsModelInfos.MIRA:
                if not self.mira_voice_file_name:
                    return "none"
                else:
                    return make_label(self.mira_voice_file_name)
            case TtsModelInfos.QWEN3TTS:
                if not self.qwen3_voice_file_name:
                    return "none"
                else:
                    return make_label(self.qwen3_voice_file_name)
            case TtsModelInfos.NONE:
                return "none"

    @property
    def has_voice(self) -> bool:
        match Tts.get_type(): # TODO: adopt this 'pattern' in mumblemumble
            case TtsModelInfos.NONE:
                has_voice = False
            case TtsModelInfos.OUTE:
                has_voice = bool(self.oute_voice_json)
            case TtsModelInfos.CHATTERBOX:
                has_voice = bool(self.chatterbox_voice_file_name)
            case TtsModelInfos.FISH:
                has_voice = bool(self.fish_voice_file_name)
            case TtsModelInfos.HIGGS:
                has_voice = bool(self.higgs_voice_file_name)
            case TtsModelInfos.VIBEVOICE:
                has_voice = bool(self.vibevoice_voice_file_name)
            case TtsModelInfos.INDEXTTS2:
                has_voice = bool(self.indextts2_voice_file_name)
            case TtsModelInfos.GLM:
                has_voice = bool(self.glm_voice_file_name)
            case TtsModelInfos.MIRA:
                has_voice = bool(self.mira_voice_file_name)
            case TtsModelInfos.QWEN3TTS:
                has_voice = bool(self.qwen3_voice_file_name)
        return has_voice

    @property
    def can_voice(self) -> bool:
        """
        Returns True if current state allows for outputting a "voice" of any kind.
        """        
        if not Tts.get_type().value.requires_voice:
            return True
        else:
            return self.has_voice

    @property
    def can_generate_audio(self) -> bool:
        return self.can_voice and len(self.phrase_groups) > 0

    @property
    def sound_segments_path(self) -> str:
        if not self.dir_path:
            return "" # TODO smth abt project not yet having a dir_path, etc
        return os.path.join(self.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)

    @property
    def concat_path(self) -> str:
        if not self.dir_path:
            return ""
        return os.path.join(self.dir_path, PROJECT_CONCAT_SUBDIR)

    @property
    def realtime_path(self) -> str:
        if not self.dir_path:
            return ""
        return os.path.join(self.dir_path, PROJECT_REALTIME_SUBDIR)

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

    def get_selected_indices_not_generated(self) -> set[int]:
        """ 
        From the currently selected range of indices, 
        returns the indicies for which no sound segment exists.
        """
        selected_indices_all = self.get_indices_to_generate()
        selected_indices_generated = set( self.sound_segments.sound_segments_map.keys() )
        selected_indices_not_generated = selected_indices_all - selected_indices_generated
        return selected_indices_not_generated

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

    def verify_voice_files_exist(self) -> bool:
        """
        Checks if the voice file/s for the current TTS model exist, and if they don't, 
        clears the field, prints feedback, and re-saves project, and returns True
        """
        attribs = []
        match Tts.get_type().value:
            case TtsModelInfos.CHATTERBOX:
                attribs = ["chatterbox_voice_file_name"] 
            case TtsModelInfos.FISH:
                attribs = ["fish_voice_file_name"] 
            case TtsModelInfos.HIGGS:
                attribs = ["higgs_voice_file_name"] 
            case TtsModelInfos.VIBEVOICE:
                attribs = ["vibevoice_voice_file_name"] 
            case TtsModelInfos.INDEXTTS2:
                attribs = ["indextts2_voice_file_name", "indextts2_emo_voice_file_name"] 
            case TtsModelInfos.GLM:
                attribs = ["glm_voice_file_name"] 
            case TtsModelInfos.MIRA:
                attribs = ["mira_voice_file_name"] 
        if not attribs:
            return False
        
        did_delete = False
        for attrib in attribs:
            file_name = getattr(self, attrib)
            if file_name and not os.path.exists(os.path.join(self.dir_path, file_name)):
                printt(f"{COL_ERROR}Missing voice file: {COL_DEFAULT}{file_name}")
                setattr(self, attrib, "")
                did_delete = True
        return did_delete

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

        # Copy 'internal' files
        src_files = [
            PROJECT_TEXT_RAW_FILE_NAME,
            source_project.chatterbox_voice_file_name,
            source_project.fish_voice_file_name,
            source_project.higgs_voice_file_name,
            source_project.vibevoice_voice_file_name,
            source_project.indextts2_voice_file_name,
            source_project.indextts2_emo_voice_file_name,
            source_project.glm_voice_file_name,
            source_project.mira_voice_file_name
        ]                                 
        src_files = [file for file in src_files if file]

        for src_file in src_files: 
            src_path = os.path.join(source_project.dir_path, src_file)
            if not Path(src_path).exists():
                continue 
            dest_path = os.path.join(self.dir_path, src_file)
            try:
                shutil.copy(src_path, dest_path)
            except Exception as e:
                ... # eat

        self.save()

    def get_batch_size(self) -> int:
        field = Tts.get_type().value.batch_size_project_field
        if not field:
            return 1
        if not hasattr(self, field):
            raise ValueError(f"Unrecognized attribute {field}")
        value = getattr(self, field)
        if value == -1:
            value = PROJECT_BATCH_SIZE_DEFAULT
        elif value > PROJECT_BATCH_SIZE_MAX:
            value = PROJECT_BATCH_SIZE_MAX # clamp silently
        return value
    
    def set_batch_size_and_save(self, value: int) -> None:
        field = Tts.get_type().value.batch_size_project_field
        if not field:
            raise ValueError(f"No support for batch_size for the current model")
        if not hasattr(self, field):
            raise ValueError(f"Unrecognized attribute {field}")
        if value > PROJECT_BATCH_SIZE_MAX:
            value = PROJECT_BATCH_SIZE_MAX # clamp silently
        setattr(self, field, value)
        self.save()

    @property
    def is_language_cjk(self) -> bool:
        if self.language_code in ["zh", "ja", "ko"]:
            return True
        if self.language_code.startswith(("zh-", "ja-", "ko-")):
            return True
        return False
