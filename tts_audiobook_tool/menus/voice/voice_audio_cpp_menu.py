"""Voice clone controls for the selected audio.cpp definition."""
from __future__ import annotations

from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.model.model_audio_cpp_menu import AudioCppMenuContext, ModelAudioCppMenu
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.audio_cpp_definition import AudioCppModelDefinition
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class VoiceAudioCppMenu:
    @staticmethod
    def make_items(state: State, definition: AudioCppModelDefinition) -> list[MenuItem]:
        items: list[MenuItem] = []
        added_config_items = False
        model_type = TtsModelType.require_by_id(definition.spec.id)
        context = AudioCppMenuContext(state, definition)
        for control in definition.menu:
            if control.kind == "voice_samples":
                items.extend(VoiceMenuShared.make_voice_sample_items(state, model_type))
            elif control.target_menu == "voice" and context.is_visible(control):
                control_items = ModelAudioCppMenu.make_control_items(state, definition, control, context)
                if control_items and not added_config_items:
                    control_items[0].blank_line_before = True
                    added_config_items = True
                items.extend(control_items)
        return items

    @staticmethod
    def menu(state: State) -> None:
        VoiceMenuShared.menu_wrapper(state, VoiceMenuShared.make_remote_items)
