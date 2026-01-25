from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Protocol

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.tts_model_info import TtsModelInfo
from tts_audiobook_tool.util import *


class TtsModel(ABC):
    """
    Base class for a TTS model

    Note: generate() method 
        Must be implementation-specific b/c variable params. 
        Must return Sound with np.dtype("float32")
    """

    def __init__(self, info: TtsModelInfo):
        self.info = info

    @abstractmethod
    def kill(self) -> None:
        """
        Performs any clean-up (nulling out local member variables, etc)
        that might help with garbage collection.
        """
        ...

    def massage_for_inference(self, text: str) -> str:
        """
        Applies text transformations from `self.info.substitutions` (usually single character punctuation)
        to address any model-specific idiosyncrasies, etc.
        Concrete class may want to override-and-super with extra logic.
        """
        for before, after in self.info.substitutions:
            text = text.replace(before, after)
        return text

# ---

class OuteProtocol(Protocol):

    DEFAULT_TEMPERATURE = 0.4 # from oute library code

    def create_speaker(self, path: str) -> dict | str:
        ...

    def generate(
            self,
            prompt: str,
            voice: dict,
            temperature: float = -1
    ) -> Sound | str:
        ...

class OuteModelProtocol(TtsModel, OuteProtocol):
        ...


class ChatterboxProtocol(Protocol):

    DEFAULT_EXAGGERATION = 0.5 # from chatterbox library code
    DEFAULT_CFG = 0.5
    DEFAULT_TEMPERATURE = 0.8

    def generate(
            self,
            text: str,
            voice_path: str,
            exaggeration: float,
            cfg: float,
            temperature: float,
            seed: int,  
            language_id: str = ""
    ) -> Sound | str:
        ...

class ChatterboxModelProtocol(TtsModel, ChatterboxProtocol):
        ...

class ChatterboxType(tuple[str, str, str], Enum):
    MULTILINGUAL = "multilingual", "Chatterbox-Multilingual", "Supports multiple languages"
    TURBO = "turbo", "Chatterbox-Turbo", "Distilled, en only"

    @property
    def id(self) -> str:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.value[1]
    
    @property
    def description(self) -> str:
        return self.value[2]

    @staticmethod
    def get_by_id(id: str) -> ChatterboxType | None:
        for item in list(ChatterboxType):
            if id == item.id:
                return item
        return None


class FishProtocol(Protocol):

    DEFAULT_TEMPERATURE = 0.8 # from fish gradio demo

    def set_voice_clone_using(self, source_path: str, transcribed_text: str) -> None:
        ...

    def clear_voice_clone(self) -> None:
        ...

    def generate(self, text: str, temperature: float, seed: int) -> Sound | str:
        ...

class FishModelProtocol(TtsModel, FishProtocol):
    ...


class HiggsProtocol(Protocol):

    # from higgs project README example code
    # (nb, higgs library code uses a default of 1.0 which is much too high for narration)
    DEFAULT_TEMPERATURE = 0.3

    def generate(
            self,
            p_voice_path: str,
            p_voice_transcript: str,
            text: str,
            seed: int,
            temperature: float
    ) -> Sound | str:
        ...

class HiggsModelProtocol(TtsModel, HiggsProtocol):
    ...


class VibeVoiceProtocol(Protocol):

    DEFAULT_MODEL_PATH = "microsoft/VibeVoice-1.5b"
    DEFAULT_MODEL_NAME = "VibeVoice 1.5B"

    # nb, their gradio demo default is 1.3, which is IMO much too low
    CFG_DEFAULT = 3.0
    CFG_MIN = 1.0
    CFG_MAX = 7.0

    DEFAULT_NUM_STEPS = 10 # from vibevoice library code

    # Must accommodate worst-case prompt size (app limit 80 words)
    MAX_TOKENS = 250

    def generate(
            self,
            texts: list[str],
            voice_path: str,
            cfg_scale: float = 3.0,
            num_steps: int = 10,
            seed: int = -1
    ) -> list[Sound] | str:
        ...

    @property
    def has_lora(self) -> bool:
        ...

class VibeVoiceModelProtocol(TtsModel, VibeVoiceProtocol):
    ...

class IndexTts2Protocol(Protocol):

    DEFAULT_EMO_VOICE_ALPHA = 0.65 # project gradio demo default
    DEFAULT_TEMPERATURE = 0.8 # project api default
    DEFAULT_USE_FP16 = False # project api default

    def generate(
            self,
            text: str,
            voice_path: str,
            temperature: float,
            emo_alpha: float,
            emo_voice_path: str,
            emo_vector: list[float]
    ) -> Sound | str:
        ...

class IndexTts2ModelProtocol(TtsModel, IndexTts2Protocol):
    ...

class GlmProtocol(Protocol):

    SAMPLE_RATES = [24000, 32000]

    def generate(
        self,
        prompt_text: str,
        prompt_speech: str,
        syn_text: str,
        seed: int
    ) -> Sound | str:
        ...

class GlmModelProtocol(TtsModel, GlmProtocol):
    ...

class MiraProtocol(Protocol):

    TEMPERATURE_DEFAULT = 0.7
    TEMPERATURE_MIN = 0.0
    TEMPERATURE_MAX = 2.0

    MAX_NEW_TOKENS = 2048 # default is 1024, which is enough for ~60 words

class MiraModelProtocol(TtsModel, MiraProtocol):
    def set_voice_clone(self, path: str) -> None:
        ...
    def clear_voice_clone(self) -> None:
        ...
    def set_params(self, temperature: float, max_new_tokens: int) -> None:
        ...
    def generate(self, prompt: str) -> Sound | str:
        ...
    def generate_batch(self, prompts: list[str]) -> list[Sound] | str:
        ...

# ---

class Qwen3ModelType(str, Enum):
    BASE = "base"
    CUSTOM_VOICE = "custom_voice"
    VOICE_DESIGN = "voice_design"
    UNKNOWN = "unknown"

class Qwen3Protocol(Protocol):

    REPO_ID_BASE_DEFAULT = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    TEMPERATURE_FALLBACK_DEFAULT = 0.9 # (Real default is embedded in the loaded model)
    TEMPERATURE_MIN = 0.01 
    TEMPERATURE_MAX = 3.0 # sane max IMO; at high values, gens can fail to terminate for a very long time

    # MAX_NEW_TOKENS = 2048 # default is 1024, which is enough for ~60 words
    ...

    @staticmethod
    def get_display_path_or_id(path_or_id: str) -> str:
        """ Returns path/id shortened/formatted for display """
        if not path_or_id:
            path_or_id = Qwen3Protocol.REPO_ID_BASE_DEFAULT
        COMMON_ID_PREFIX = "Qwen/"
        if path_or_id.startswith(COMMON_ID_PREFIX):
            path_or_id = path_or_id[len(COMMON_ID_PREFIX):]
        else:
            if os.path.exists(path_or_id):
                path_or_id = AppUtil.local_path_for_display(path_or_id)
        return path_or_id
        
class Qwen3ModelProtocol(TtsModel, Qwen3Protocol):

    def generate_base(
            self, 
            prompts: list[str], 
            voice_info: tuple[str, str], 
            language_code: str,
            temperature: float,
            seed: int
    ) -> list[Sound] | str:
        ...

    def generate_custom_voice(
        self, 
        prompts: list[str], 
        speaker_id: str, 
        instruct: str, 
        language_code: str,
        temperature: float,
        seed: int
    ) -> list[Sound] | str:
        ...

    def generate_voice_design(
        self, 
        prompts: list[str], 
        instruct: str, 
        language_code: str,
        temperature: float,
        seed: int
    ) -> list[Sound] | str:
        ...

    def clear_voice(self) -> None:
        ...

    @property
    def path_or_id(self) -> str:
        ...

    @property
    def model_type(self) -> str:
        ...

    @property
    def supported_languages(self) -> list[str]:
        ...

    @property
    def supported_speakers(self) -> list[str]:
        ...    

    @property
    def generate_defaults(self) -> dict[str, Any]:
        ...

    @property
    def is_model_type_supported(self) -> bool:
        return self.model_type in ["base", "custom_voice", "voice_design"] # ie, all current types
    
    def resolve_language_code_and_warning(self, language_code: str) -> tuple[str, str]:
        """
        Returns qwen model's language value and warning if any.
        Project language code is expected to be a 2-letter ISO-mumblemumble language code, but
        Qwen uses english names for language value.
        """

        # This hardcoded map may need updating if model's supported languages changes with future updates.
        # Or, consider expanding this with the assumption that future models or finetunes will also
        # use english word values.
        MAP = {
            'zh': 'chinese',
            'en': 'english',
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'ja': 'japanese',
            'ko': 'korean',
            'pt': 'portuguese',
            'ru': 'russian',
            'es': 'spanish'
        }

        language_code = language_code.lower()
        qwen_language = MAP.get(language_code, "")

        if language_code.startswith("zh"):
            result = "chinese", ""
        elif language_code in MAP:
            result = MAP[language_code], ""                
        elif qwen_language and not qwen_language in self.supported_languages:
            # Probably can't happen with the current qwen3tts models
            result = "auto", f"Warning: Language `{qwen_language}` is not part of the currently loaded Qwen3-TTS model's \"supported languages\" list; will use 'auto'"
        else:
            result = "auto", f"Warning: Project language code {language_code} doesn't map to a known Qwen3-TTS language value; will use 'auto'"
        
        return result
    
    def make_main_menu_model_desc(self, project) -> str:
        
        from tts_audiobook_tool.project import Project
        assert isinstance(project, Project)

        match self.model_type:
            case "base":
                if not project.qwen3_voice_file_name:
                    desc = f"{COL_ERROR}required"
                else:                    
                    desc = f"{COL_DIM}currently: {COL_ACCENT}{project.get_voice_label()}"
            case "custom_voice":
                desc = "custom speaker: "
                if len(self.supported_speakers) == 1:
                    desc += self.supported_speakers[0]
                else:
                    if project.qwen3_speaker_id in self.supported_speakers:
                        desc += project.qwen3_speaker_id
                    else:
                        desc = "required"
            case "voice_design":
                desc = f"current instructions: "
                if not project.qwen3_instructions:
                    desc += f"{COL_ERROR}none"
                else:
                    instructions = truncate_pretty(project.qwen3_instructions, 40, middle=False, content_color=COL_ACCENT)
                    desc = f"current instructions: {COL_ACCENT}{instructions}"
            case _:
                return ""
        
        return f"{COL_DIM}({desc}{COL_DIM})"
    
    def get_post_init_error(self, project) -> str:      
        """ 
        Returns error message if prerequisite properties don't exist or are incorrect
        (ie, inference should not be attempted)
        """

        if not self.is_model_type_supported:
            return f"Unsupported model type: {self.model_type}"

        from tts_audiobook_tool.project import Project
        assert isinstance(project, Project)

        match self.model_type:
            case "base":
                if not project.qwen3_voice_file_name:
                    return "Voice sample required"
                else:
                    return ""
            case "custom_voice":
                if len(self.supported_speakers) == 1:
                    return ""
                if not (project.qwen3_speaker_id in self.supported_speakers):
                    return "Speaker id required"
                else:
                    return ""
            case "voice_design":
                return ""
            case _:
                return f"Unsupported model_type: {self.model_type}"
        
        
    def get_post_init_warning(self, project) -> str:        
        """ 
        Returns 'non-blocking' warning message if necessary 
        """

        from tts_audiobook_tool.project import Project
        assert isinstance(project, Project)

        _, warning = self.resolve_language_code_and_warning(project.language_code)
        if warning:
            return warning
        
        return ""
        
        