from __future__ import annotations
from enum import Enum

from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object

class Qwen3BaseModel(TtsBaseModel):

    INFO = TtsModelInfos.QWEN3TTS.value

    DEFAULT_REPO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    TEMPERATURE_FALLBACK_DEFAULT = 0.9 # (Real default is embedded in the loaded model)
    TEMPERATURE_MIN = 0.01
    TEMPERATURE_MAX = 3.0 # sane max IMO; at high values, gens can fail to terminate for a very long time
    TOP_K_DEFAULT = 50
    TOP_P_DEFAULT = 1.0
    REPETITION_PENALTY_DEFAULT = 1.05
            
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

    def get_resolved_speaker_info(self, project: Project) -> tuple[str, bool]:
        """ 
        Returns resolved speaker id based on Project value, 
        and if it is valid for the currently loaded model. 
        """
        if self.model_type != "custom_voice":
            return "", False
        if len(self.supported_speakers) == 1:
            return self.supported_speakers[0], True
        return project.qwen3_speaker_id, project.qwen3_speaker_id in self.supported_speakers

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
    
    # ---

    @classmethod
    def get_prereq_errors(
            cls, project: Project, instance: TtsBaseModel | None, short_format: bool
    ) -> list[str]:

        if instance:
            assert(isinstance(instance, Qwen3BaseModel))

        errors = []

        if instance and not instance.is_model_type_supported:
            return ["unsupported model type"]

        match project.qwen3_model_type:            
            case "custom_voice":
                if not instance:
                    ... # can't know if project settings valid wo instance
                else:
                    is_valid = instance.get_resolved_speaker_info(project)[1]
                    if not is_valid:
                        err = "requires speaker" if short_format else "A valid speaker id is required"
                        errors.append(err)
            case "voice_design":
                ... # # has no requirements bc "instruction" is optional
            case "base" | _:
                if not project.qwen3_voice_file_name:
                    err = "requires voice sample" if short_format else "Voice sample required"
                    errors.append(err)
        
        return errors

    def get_prereq_warnings(self, project: Project) -> list[str]:
        
        warnings = []

        _, warning = self.resolve_language_code_and_warning(project.language_code)
        if warning:
            warnings.append(warning)

        if project.qwen3_model_type == "voice_design" and not project.qwen3_instructions:
            warning = "Model may generate random voices because no instructions defined"
            warnings.append(warning)

        return warnings

    # ---

    @classmethod
    def get_voice_display_info(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> tuple[str, str]:

        if instance:
            assert(isinstance(instance, Qwen3BaseModel))

        match project.qwen3_model_type:
            
            case "custom_voice":
                if instance:
                    resolved_speaker_id, is_valid = instance.get_resolved_speaker_info(project)
                    if not resolved_speaker_id:
                        resolved_speaker_id = "required"
                    if is_valid and len(instance.supported_speakers) > 1:
                        prefix = "current speaker"
                    else:
                        prefix = "speaker"
                    value = (COL_ACCENT if is_valid else COL_ERROR) + resolved_speaker_id
                else:
                    prefix = "speaker"
                    # Instance doesn't yet exist, so settle for incomplete info
                    value = ""
            
            case "voice_design":
                prefix = "instruction"
                if not project.qwen3_instructions:
                    value = COL_ERROR + "none"
                else:
                    value = truncate_pretty(project.qwen3_instructions, 30, middle=False)
            
            case "base " | _:
                if not project.qwen3_voice_file_name:
                    prefix, value = COL_ERROR + "required", ""
                else:
                    prefix = "current voice clone"
                    value = COL_ACCENT + ellipsize_path_for_menu(project.qwen3_voice_file_name)

        return prefix, value

    @classmethod
    def get_voice_tag(cls, project: Project) -> str:
        
        match project.qwen3_model_type:            
            case "custom_voice":
                if project.qwen3_speaker_id:
                    return TextUtil.sanitize_for_filename(project.qwen3_speaker_id)[:30]
                else:
                    return "speaker" # good enough
            case "voice_design":
                return "voice_design"
            case "base " | _:
                return super().get_voice_tag(project)

# ---

class Qwen3ModelType(str, Enum):
    BASE = "base"
    CUSTOM_VOICE = "custom_voice"
    VOICE_DESIGN = "voice_design"
    UNKNOWN = "unknown"

