from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.conversation.sound_input_device_util import SoundInputDeviceInfo
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos

if TYPE_CHECKING:
    from tts_audiobook_tool.state import State


@dataclass
class PrereqError:
    
    # Unfulfilled item as a very short noun-phrase. 
    # Eg, "voice clone". Answers question, "What is missing"
    short: str
    
    # Full sentence description following this format: 
    # "Voice clone is required"
    verbose: str


class PrereqUtil:
    """ Logic for getting unfulfilled prerequisites based on current State """

    @staticmethod
    def get_generate_prereq_errors(state: State, is_realtime_audiobook: bool=False) -> list[PrereqError]:
        """ 
        Returns list of short noun descriptions of unfulfilled items required to do generate 
        """
        from tts_audiobook_tool.tts import Tts

        items = []

        if Tts.get_type() == TtsModelInfos.NONE:
            items.append(
                PrereqError("TTS model", "TTS model is required")
            )

        if is_realtime_audiobook:
            if not state.real_time.custom_phrase_groups and not state.project.phrase_groups:
                items.append(
                    PrereqError("text", "Project text or custom text is required")
                )
        else:
            if not state.project.phrase_groups:
                items.append(
                    PrereqError("text", "Text must be imported into the project")
                )

        model_errors = Tts.get_class().get_prereq_errors(state.project, Tts.get_instance_if_exists()) 
        if model_errors:
            items.extend(model_errors)

        return items

    @staticmethod
    def get_generate_prereq_error_string(state: State, verbose=False, is_realtime_playback=False) -> str:
        items = PrereqUtil.get_generate_prereq_errors(state, is_realtime_playback)
        return PrereqUtil.prereq_errors_to_string(items, verbose)
    
    @staticmethod
    def get_chat_prereq_errors(state: State) -> list[PrereqError]:
        """ """
        errors = [] 
        
        # Hotkeys
        if not AskUtil.can_hotkey:
            errors.append(PrereqError("terminal hotkey support", "Terminal does not have hotkey capabilities"))

        # Microphone 
        sound_input_error = SoundInputDeviceInfo.get_check_error()
        if sound_input_error:
            errors.append(PrereqError("microphone", sound_input_error))

        # LLM settings
        # Rem, api key and model name may or may not be required depending on endpoint
        if not state.prefs.llm_url.strip():
            errors.append(PrereqError("LLM settings", "LLM endpoint URL not set (see Options > LLM Settings)"))

        # STT not disabled
        if state.prefs.stt_variant == SttVariant.DISABLED:
            errors.append(PrereqError("STT enabled", "Speech-to-text must be enabled (see Options menu > Whisper model)"))

        # TTS Model generation prereqs
        from tts_audiobook_tool.tts import Tts
        model_errors = Tts.get_class().get_prereq_errors(state.project, Tts.get_instance_if_exists()) 
        if model_errors:
            errors.extend(model_errors)

        return errors

    @staticmethod
    def get_chat_prereq_error_string(state: State, verbose=False) -> str:
        items = PrereqUtil.get_chat_prereq_errors(state)
        return PrereqUtil.prereq_errors_to_string(items, verbose)
    
    @staticmethod
    def prereq_errors_to_string(prereq_errors: list[PrereqError], verbose=False) -> str:
        if not prereq_errors:
            return ""
        if len(prereq_errors) == 1:
            if verbose: 
                return prereq_errors[0].verbose
            else:
                return f"requires {prereq_errors[0].short}"
        else:
            if verbose:
                verbose_items = [item.verbose for item in prereq_errors]
                return '\n'.join(verbose_items)
            else:
                short_items = [item.short for item in prereq_errors]
                return f"requires: {', '.join(short_items)}"

