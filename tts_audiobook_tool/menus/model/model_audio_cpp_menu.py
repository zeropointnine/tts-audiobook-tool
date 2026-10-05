"""Model settings for the selected audio.cpp server family."""
from __future__ import annotations

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM
from tts_audiobook_tool.constants_hints import HINT_MOSS_TEMPERATURE
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppSettings
from tts_audiobook_tool.tts_models.audio_cpp_definition import (
    AudioCppMenuControl,
    AudioCppModelDefinition,
    AudioCppParameter,
    AudioCppTextParameter,
)
from tts_audiobook_tool.util import make_menu_label


class ModelAudioCppMenu:
    @staticmethod
    def make_items(state: State, definition: AudioCppModelDefinition) -> list[MenuItem]:
        items: list[MenuItem] = []
        for control in definition.menu:
            if control.target_menu == "model":
                items.extend(ModelAudioCppMenu.make_control_items(state, definition, control))
        return items

    @staticmethod
    def make_control_items(
            state: State, definition: AudioCppModelDefinition, control: AudioCppMenuControl,
    ) -> list[MenuItem]:
        """Expand a settings control identically in either destination menu."""
        if control.kind == "seed":
            return [ModelMenuShared.make_seed_item(state, SettingRef(definition.spec.id, "seed"))]
        if control.kind == "voice_instructions":
            is_omnivoice = definition.family == "omnivoice"
            return ModelMenuShared.make_voice_instructions_item(
                state, definition.spec.id, control.parameter,
                label="Voice design instructions" if is_omnivoice else "Instructions",
                validate_omnivoice=is_omnivoice)
        parameter = definition.parameters[control.parameter]
        if isinstance(parameter, AudioCppTextParameter):
            # The catalog parser only routes string parameters to the
            # `voice_instructions` control above.
            raise ValueError(f"{definition.spec.id}.{control.parameter} is not a numeric parameter")
        return [ModelAudioCppMenu.make_parameter_item(
            state, parameter, control.label,
            show_hint=definition.language_policy == "moss" and parameter.name.endswith("temperature"),
        )]

    @staticmethod
    def make_parameter_item(state: State, parameter: AudioCppParameter, label: str, *, show_hint: bool = False) -> MenuItem:
        num_decimals = 0 if parameter.type == "int" else 2

        def display(current: State) -> str:
            try:
                effective = AudioCppSettings.get(current.project, parameter)
            except ValueError as exc:
                return f"{label} (invalid: {exc})"
            return make_menu_label(label, effective, parameter.default, num_decimals=num_decimals)

        def edit(current: State, _: MenuItem) -> None:
            try:
                effective = AudioCppSettings.get(current.project, parameter)
            except ValueError as exc:
                ask.ask_error(str(exc))
                effective = parameter.default
            suffix = f" {COL_DIM}{parameter.input_prompt_suffix}{COL_DEFAULT}" if parameter.input_prompt_suffix else ""
            if show_hint:
                hints.show_hint_if_necessary(current.prefs, HINT_MOSS_TEMPERATURE)
            ask.ask_number_and_save(
                saveable=current.project,
                target=SettingRef(parameter.model_id, parameter.name),
                prompt=f"Enter {label}{suffix}",
                min_value=parameter.min,
                max_value=parameter.max,
                default_value=parameter.default,
                success_prefix="Value set:",
                is_int=parameter.type == "int",
                prefill_value=float(effective),
            )

        return MenuItem(display, edit)

    @staticmethod
    def menu(state: State) -> None:
        ModelMenuShared.menu_wrapper(state, ModelMenuShared.make_remote_items)
