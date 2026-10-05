"""Voice and model settings for the selected audio.cpp server family."""
from __future__ import annotations

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM
from tts_audiobook_tool.constants_hints import HINT_MOSS_TEMPERATURE
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppSettings
from tts_audiobook_tool.tts_models.audio_cpp_definition import (
    AudioCppModelDefinition,
    AudioCppParameter,
    AudioCppTextParameter,
)
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import make_menu_label


class VoiceAudioCppMenu:
    @staticmethod
    def make_items(state: State, definition: AudioCppModelDefinition) -> list[MenuItem]:
        items: list[MenuItem] = []
        groups: list[str] = []
        model_type = TtsModelType.require_by_id(definition.spec.id)
        for control in definition.menu:
            if control.kind == "voice_samples":
                voice_items = VoiceMenuShared.make_voice_sample_items(state, model_type)
                items.extend(voice_items)
                groups.extend([""] * len(voice_items))
            elif control.kind == "seed":
                items.append(VoiceMenuShared.make_seed_item(state, SettingRef(definition.spec.id, "seed")))
                groups.append("")
            elif control.kind == "voice_instructions":
                is_omnivoice = definition.family == "omnivoice"
                instruction_items = VoiceMenuShared.make_voice_instructions_item(
                    state, definition.spec.id, control.parameter,
                    label="Voice design instructions" if is_omnivoice else "Instructions",
                    validate_omnivoice=is_omnivoice)
                items.extend(instruction_items)
                groups.extend([control.group] * len(instruction_items))
            else:
                parameter = definition.parameters[control.parameter]
                if isinstance(parameter, AudioCppTextParameter):
                    # The catalog parser only routes string parameters to the
                    # `voice_instructions` control above.
                    raise ValueError(f"{definition.spec.id}.{control.parameter} is not a numeric parameter")
                items.append(VoiceAudioCppMenu.make_parameter_item(
                    state, parameter, control.label,
                    show_hint=definition.language_policy == "moss" and parameter.name.endswith("temperature"),
                ))
                groups.append(control.group)
        VoiceMenuShared.apply_group_superlabels(items, groups)
        return items

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
        VoiceMenuShared.menu_wrapper(state, VoiceMenuShared.make_remote_items)
