from __future__ import annotations

from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import ReadinessIssue, SttVariant
from tts_audiobook_tool import ask
from tts_audiobook_tool.conversation.sound_input_device_util import SoundInputDeviceInfo

if TYPE_CHECKING:
    from tts_audiobook_tool.state import State

def get_generate_blockers(state: State, is_realtime_audiobook: bool=False) -> list[ReadinessIssue]:
    """Returns blocking issues that prevent audiobook generation from starting."""
    from tts_audiobook_tool.tts import Tts

    items = []

    if is_realtime_audiobook:
        if not state.real_time.custom_phrase_groups and not state.project.phrase_groups:
            items.append(
                ReadinessIssue("text", "Project text or custom text is required")
            )
    else:
        if not state.project.phrase_groups:
            items.append(
                ReadinessIssue("text", "Text must be imported into the project")
            )

    model_errors = Tts.get_class().get_blocking_issues(state.project, Tts.get_instance_if_exists()) 
    if model_errors:
        items.extend(model_errors)

    return items


def get_generate_blocker_text(state: State, verbose=False, is_realtime_playback=False) -> str:
    items = get_generate_blockers(state, is_realtime_playback)
    return format_issues(items, verbose)


def get_chat_blockers(state: State) -> list[ReadinessIssue]:
    """Returns blocking issues that prevent chat from starting."""
    errors = [] 
    
    # Hotkeys
    if not ask.can_hotkey:
        errors.append(ReadinessIssue("terminal hotkey support", "Terminal does not have hotkey capabilities"))

    # Microphone 
    sound_input_error = SoundInputDeviceInfo.get_check_error()
    if sound_input_error:
        errors.append(ReadinessIssue("microphone", sound_input_error))

    # LLM settings
    # Rem, api key and model name may or may not be required depending on endpoint
    if not state.prefs.llm_url.strip():
        errors.append(ReadinessIssue("LLM settings", "LLM endpoint URL not set (see Options > LLM Settings)"))

    # STT not disabled
    if state.prefs.stt_variant == SttVariant.DISABLED:
        errors.append(ReadinessIssue("Whisper model enabled", "Speech-to-text must be enabled (see Options menu > Whisper model)"))

    # TTS Model generation readiness
    from tts_audiobook_tool.tts import Tts
    model_errors = Tts.get_class().get_blocking_issues(state.project, Tts.get_instance_if_exists()) 
    if model_errors:
        errors.extend(model_errors)

    return errors


def get_chat_blocker_text(state: State, verbose=False) -> str:
    items = get_chat_blockers(state)
    return format_issues(items, verbose)


def format_issues(issues: list[ReadinessIssue], verbose=False) -> str:
    if not issues:
        return ""
    if len(issues) == 1:
        if verbose: 
            return issues[0].verbose
        else:
            return f"requires {issues[0].short}"
    else:
        if verbose:
            verbose_items = [item.verbose for item in issues]
            return '\n'.join(verbose_items)
        else:
            short_items = [item.short for item in issues]
            return f"requires: {', '.join(short_items)}"

