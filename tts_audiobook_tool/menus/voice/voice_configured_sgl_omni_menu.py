"""Voice menu rendered from the active validated SGL-Omni definition."""
from __future__ import annotations

from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredSettings
from tts_audiobook_tool.tts_models.sgl_omni_definition import NumericParameter, SglOmniModelDefinition
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.constants import VOICE_ADVANCED_SUPERLABEL
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.util import make_menu_label, print_feedback, printt


class VoiceConfiguredSglOmniMenu:
    @staticmethod
    def make_items(state: State, definition: SglOmniModelDefinition) -> list[MenuItem]:
        items: list[MenuItem] = []
        model_type = TtsModelType.get_by_id(definition.spec.id)
        for control in definition.menu:
            if control.kind == "voice_samples":
                items.extend(VoiceMenuShared.make_voice_sample_items(state, model_type))
                continue
            if control.kind == "seed":
                items.append(VoiceMenuShared.make_seed_item(
                    state, SettingRef(definition.spec.id, "seed"), add_batch_warning=True))
                continue
            parameter = definition.parameters[control.parameter]
            item = VoiceConfiguredSglOmniMenu.make_parameter_item(state, parameter, control.label, definition.spec.id)
            if control.group == "advanced":
                item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(item)
        return items

    @staticmethod
    def make_parameter_item(state: State, parameter: NumericParameter, label: str, model_id: str | None = None) -> MenuItem:
        def display(current: State) -> str:
            try:
                effective = ConfiguredSettings.get(current.project, parameter, model_id)
            except ValueError as exc:
                return f"{label} (invalid: {exc})"
            return make_menu_label(label, effective, parameter.default,
                                   num_decimals=0 if parameter.type == "int" else 2)

        def edit(current: State, _: MenuItem) -> None:
            try:
                effective = ConfiguredSettings.get(current.project, parameter, model_id)
            except ValueError as exc:
                ask.ask_error(str(exc))
                effective = parameter.default
            ceiling = parameter.max_request_value if parameter.max_request_value is not None else parameter.max
            reset = f"; {parameter.default_sentinel} resets to default {parameter.default}" if parameter.default_sentinel is not None else ""
            printt(f"Enter {label} (valid range: {parameter.min}-{ceiling}{reset}):")
            response = ask.ask_input(prefill=str(effective))
            if not response or response == str(effective):
                return
            try:
                value = float(response)
                if parameter.type == "int" and not value.is_integer():
                    raise ValueError("Expected an integer")
                if parameter.default_sentinel is not None and value == parameter.default_sentinel:
                    value_to_store = None
                else:
                    if parameter.max_request_value is not None and value > parameter.max_request_value:
                        raise ValueError(f"{label} must be between {parameter.min} and {parameter.max_request_value} for this server")
                    value_to_store = int(value) if parameter.type == "int" else value
                error = ConfiguredSettings.set(current.project, parameter, value_to_store, model_id)
                if error:
                    ask.ask_error(error)
                else:
                    print_feedback("Value set:", str(parameter.default if value_to_store is None else value_to_store))
            except ValueError as exc:
                ask.ask_error(str(exc))

        return MenuItem(display, edit)

    @staticmethod
    def menu(state: State, definition: SglOmniModelDefinition) -> None:
        VoiceMenuShared.menu_wrapper(state, lambda current: VoiceConfiguredSglOmniMenu.make_items(current, definition))
