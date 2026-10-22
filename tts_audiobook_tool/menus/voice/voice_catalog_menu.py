"""Voice clone controls for a catalog definition (SGL-Omni or audio.cpp)."""
from __future__ import annotations

from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.model.catalog_menu_source import CatalogDefinition, catalog_menu_source
from tts_audiobook_tool.menus.model.model_catalog_menu import ModelCatalogMenu
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class VoiceCatalogMenu:
    @staticmethod
    def make_items(state: State, definition: CatalogDefinition) -> list[MenuItem]:
        source = catalog_menu_source(definition)
        items: list[MenuItem] = []
        added_config_items = False
        model_type = TtsModelType.require_by_id(source.model_id)
        for control in source.controls:
            if control.kind == "voice_samples":
                preset_choices, allow_custom_preset = source.voice_preset_choices(model_type)
                items.extend(VoiceMenuShared.make_voice_sample_items(
                    state, model_type,
                    preset_choices=preset_choices,
                    allow_custom_preset=allow_custom_preset,
                ))
            elif control.target_menu == "voice":
                # Voice-targeted settings reuse the Model menu's editors.
                control_items = ModelCatalogMenu.make_control_items(state, definition, control)
                if control_items and not added_config_items:
                    control_items[0].blank_line_before = True
                    added_config_items = True
                items.extend(control_items)
        return items

    @staticmethod
    def menu(state: State) -> None:
        VoiceMenuShared.menu_wrapper(state, VoiceMenuShared.make_remote_items)
