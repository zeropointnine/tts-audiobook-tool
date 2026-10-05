"""Model settings rendered from the active validated SGL-Omni definition."""
from __future__ import annotations

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM
from tts_audiobook_tool.constants_hints import HINT_MOSS_TEMPERATURE
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredSettings
from tts_audiobook_tool.tts_models.sgl_omni_definition import MenuControl, NumericParameter, SglOmniModelDefinition
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.util import make_menu_label, print_feedback, printt


class ModelConfiguredSglOmniMenu:
    @staticmethod
    def make_items(state: State, definition: SglOmniModelDefinition) -> list[MenuItem]:
        items: list[MenuItem] = []
        for control in definition.menu:
            if control.target_menu == "model":
                items.extend(ModelConfiguredSglOmniMenu.make_control_items(state, definition, control))
        return items

    @staticmethod
    def make_control_items(
            state: State, definition: SglOmniModelDefinition, control: MenuControl,
    ) -> list[MenuItem]:
        """Expand a settings control identically in either destination menu."""
        if control.kind == "seed":
            return [ModelMenuShared.make_seed_item(
                state, SettingRef(definition.spec.id, "seed"), add_batch_warning=True)]
        parameter = definition.parameters[control.parameter]
        return [ModelConfiguredSglOmniMenu.make_parameter_item(
            state, parameter, control.label, definition.spec.id,
            standard_prompt=definition.language_policy == "moss" and parameter.request_key == "temperature",
        )]

    @staticmethod
    def make_parameter_item(state: State, parameter: NumericParameter, label: str, model_id: str | None = None,
                            *, standard_prompt: bool = False) -> MenuItem:
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
            suffix = f" {COL_DIM}{parameter.input_prompt_suffix}{COL_DEFAULT}" if parameter.input_prompt_suffix else ""
            # MOSS temperatures use the stock bounded-number editor. Keep other
            # controls' existing reset and server-specific validation behavior.
            if standard_prompt:
                hints.show_hint_if_necessary(current.prefs, HINT_MOSS_TEMPERATURE)
                ask.ask_number_and_save(
                    saveable=current.project,
                    target=SettingRef(model_id or parameter.model_id, parameter.name),
                    prompt=f"Enter {label}{suffix}",
                    min_value=parameter.min,
                    max_value=ceiling,
                    default_value=parameter.default,
                    success_prefix="Value set:",
                    is_int=parameter.type == "int",
                    prefill_value=effective,
                )
                return
            reset = f"; {parameter.default_sentinel} resets to default {parameter.default}" if parameter.default_sentinel is not None else ""
            printt(f"Enter {label} (valid range: {parameter.min}-{ceiling}{reset}){suffix}:")
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
    def menu(state: State) -> None:
        ModelMenuShared.menu_wrapper(state, ModelMenuShared.make_remote_items)
