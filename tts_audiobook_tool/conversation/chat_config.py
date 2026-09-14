from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tts_audiobook_tool import text_util
from tts_audiobook_tool.util import ellipsize
from tts_audiobook_tool.constants import (
    ASSETS_DIR_NAME,
    CHAT_SYSTEM_PROMPTS,
    package_dir,
)
from tts_audiobook_tool.conversation.conversation_types import ChatInputMode

if TYPE_CHECKING:
    from tts_audiobook_tool.state import State


def is_text_input_mode(mode: ChatInputMode) -> bool:
    return mode == ChatInputMode.TEXT


def is_microphone_input_mode(mode: ChatInputMode) -> bool:
    return mode in (ChatInputMode.MIC_ENTER, ChatInputMode.MIC_IMMEDIATE)


def is_immediate_input_mode(mode: ChatInputMode) -> bool:
    return mode == ChatInputMode.MIC_IMMEDIATE


def get_chat_input_mode_label(mode: ChatInputMode) -> str:
    return "text" if is_text_input_mode(mode) else "microphone"


def get_system_prompt_preset_label(preset: str) -> str:
    if not preset:
        return "none"
    for file_name, label in CHAT_SYSTEM_PROMPTS:
        if file_name == preset:
            return label
    return preset


def get_system_prompt_label(state: State) -> str:
    preset = state.prefs.system_prompt_preset
    if preset:
        return get_system_prompt_preset_label(preset)
    return ellipsize(state.prefs.llm_system_prompt, 50) or "none"


def get_resolved_system_prompt(state: State) -> str:
    preset = state.prefs.system_prompt_preset
    if preset:
        path = os.path.join(package_dir, ASSETS_DIR_NAME, preset)
        return text_util.load_text_file(path)
    return state.prefs.llm_system_prompt.strip()
