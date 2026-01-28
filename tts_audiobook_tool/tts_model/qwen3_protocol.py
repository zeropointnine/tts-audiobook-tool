from __future__ import annotations
from enum import Enum
from typing import Protocol

from tts_audiobook_tool.tts_model.tts_model import TtsModel
from tts_audiobook_tool.util import *

class Qwen3ModelType(str, Enum):
    BASE = "base"
    CUSTOM_VOICE = "custom_voice"
    VOICE_DESIGN = "voice_design"
    UNKNOWN = "unknown"

class Qwen3Protocol(Protocol):

    DEFAULT_REPO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    TEMPERATURE_FALLBACK_DEFAULT = 0.9 # (Real default is embedded in the loaded model)
    TEMPERATURE_MIN = 0.01 
    TEMPERATURE_MAX = 3.0 # sane max IMO; at high values, gens can fail to terminate for a very long time
            
class Qwen3ModelProtocol(TtsModel, Qwen3Protocol):

    def clear_voice(self) -> None:
        ...

    @property
    def model_target(self) -> str:
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
