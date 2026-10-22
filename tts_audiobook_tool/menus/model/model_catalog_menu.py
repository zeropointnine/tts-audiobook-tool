"""Model settings rendered from a catalog definition (SGL-Omni, audio.cpp or local).

Backend differences live in `CatalogMenuSource` adapters; this renderer only
expands declared menu controls into menu items. Hand-written local menus also
build individual numeric items here (`make_local_parameter_item`).
"""
from __future__ import annotations

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM
from tts_audiobook_tool.constants_hints import HINT_MOSS_TEMPERATURE
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.menus.model.catalog_menu_source import (
    CatalogDefinition,
    CatalogMenuControl,
    CatalogMenuSource,
    ChoiceSetting,
    NumericSetting,
    SglOmniMenuSource,
    catalog_menu_source,
)
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import make_menu_label, print_feedback, printt


class ModelCatalogMenu:
    @staticmethod
    def make_items(state: State, definition: CatalogDefinition) -> list[MenuItem]:
        source = catalog_menu_source(definition)
        items: list[MenuItem] = []
        for control in source.controls:
            if control.target_menu == "model":
                items.extend(ModelCatalogMenu.make_control_items(state, definition, control))
        return items

    @staticmethod
    def make_control_items(
            state: State, definition: CatalogDefinition, control: CatalogMenuControl,
    ) -> list[MenuItem]:
        """Expand a settings control identically in either destination menu."""
        source = catalog_menu_source(definition)
        if control.kind == "seed":
            return [ModelMenuShared.make_seed_item(
                state, SettingRef(source.model_id, "seed"), add_batch_warning=source.seed_batch_warning)]
        if control.kind == "choice":
            return [ModelCatalogMenu.make_choice_item(
                state, source.choice(control.parameter), control.label, source, subheading=control.subheading)]
        if control.kind == "rolling_continuation":
            return [ModelCatalogMenu.make_rolling_continuation_item(
                source.numeric(control.parameter), qualifier_line=control.subheading)]
        return [ModelCatalogMenu.make_parameter_item(
            state, source.numeric(control.parameter), control.label, source)]

    @staticmethod
    def make_local_parameter_item(state: State, model_id: str, name: str, label: str) -> MenuItem:
        """A catalog numeric item for a hand-written local menu (`[models.local.parameters]`)."""
        from tts_audiobook_tool.tts_models.local_definition import require_local_definition
        source = catalog_menu_source(require_local_definition(model_id))
        return ModelCatalogMenu.make_parameter_item(state, source.numeric(name), label, source)

    @staticmethod
    def make_parameter_item(
            state: State, parameter: NumericSetting, label: str, source: CatalogMenuSource,
    ) -> MenuItem:
        def display(current: State) -> str:
            try:
                effective = source.get_number(current.project, parameter)
            except ValueError as exc:
                return f"{label} (invalid: {exc})"
            return make_menu_label(label, effective, parameter.default,
                                   num_decimals=0 if parameter.type == "int" else 2)

        def edit(current: State, _: MenuItem) -> None:
            try:
                effective = source.get_number(current.project, parameter)
            except ValueError as exc:
                ask.ask_error(str(exc))
                effective = parameter.default
            style = source.edit_style(parameter)
            if style in ("standard", "standard_reset"):
                ModelCatalogMenu._edit_standard(current, parameter, label, source, effective,
                                                accepts_reset=style == "standard_reset")
            else:
                ModelCatalogMenu._edit_sentinel_reset(current, parameter, label, source, effective)

        return MenuItem(display, edit)

    @staticmethod
    def _ceiling(parameter: NumericSetting, source: CatalogMenuSource) -> int | float:
        cap = source.request_cap(parameter)
        return parameter.max if cap is None else cap

    @staticmethod
    def _edit_standard(
            state: State, parameter: NumericSetting, label: str, source: CatalogMenuSource,
            effective: int | float, *, accepts_reset: bool = False,
    ) -> None:
        """The stock bounded-number editor; a catalog proviso gets its own prompt line.

        With `accepts_reset`, entering -1 stores the sentinel ("use default")."""
        if source.shows_moss_hint(parameter):
            hints.show_hint_if_necessary(state.prefs, HINT_MOSS_TEMPERATURE)
        ask.ask_number_and_save(
            saveable=state.project,
            target=SettingRef(parameter.model_id, parameter.name),
            prompt=ModelMenuShared.with_prompt_suffix(f"Enter {label}", parameter.input_prompt_suffix),
            min_value=parameter.min,
            max_value=ModelCatalogMenu._ceiling(parameter, source),
            default_value=parameter.default,
            success_prefix="Value set:",
            is_int=parameter.type == "int",
            is_minus_one_default=accepts_reset,
            prefill_value=float(effective),
        )

    @staticmethod
    def _edit_sentinel_reset(
            state: State, parameter: NumericSetting, label: str, source: CatalogMenuSource,
            effective: int | float,
    ) -> None:
        """SGL-Omni's editor: entering the sentinel resets to the default, and
        the request ceiling may be below the UI max."""
        assert isinstance(source, SglOmniMenuSource), "only SGL-Omni uses the sentinel-reset editor"
        cap = source.request_cap(parameter)
        ceiling = ModelCatalogMenu._ceiling(parameter, source)
        suffix = f" {COL_DIM}{parameter.input_prompt_suffix}{COL_DEFAULT}" if parameter.input_prompt_suffix else ""
        sentinel = parameter.default_sentinel
        reset = f"; {sentinel} resets to default {parameter.default}" if sentinel is not None else ""
        printt(f"Enter {label} (valid range: {parameter.min}-{ceiling}{reset}){suffix}:")
        response = ask.ask_input(prefill=str(effective))
        if not response or response == str(effective):
            return
        try:
            value = float(response)
            if parameter.type == "int" and not value.is_integer():
                raise ValueError("Expected an integer")
            if sentinel is not None and value == sentinel:
                value_to_store = None
            else:
                if cap is not None and value > cap:
                    raise ValueError(f"{label} must be between {parameter.min} and {cap} for this server")
                value_to_store = int(value) if parameter.type == "int" else value
            error = source.set_number(state.project, parameter, value_to_store)
            if error:
                ask.ask_error(error)
            else:
                print_feedback("Value set:", str(parameter.default if value_to_store is None else value_to_store))
        except ValueError as exc:
            ask.ask_error(str(exc))

    @staticmethod
    def make_rolling_continuation_item(parameter: NumericSetting, *, qualifier_line: str = "") -> MenuItem:
        """The shared rolling-continuation editor; `parameter.max` is the maximum length."""
        target = SettingRef(parameter.model_id, parameter.name)

        def display(current: State) -> str:
            return ModelMenuShared.make_rolling_continuation_label(
                current.project.get_model_setting(parameter.model_id, parameter.name))

        def edit(current: State, _: MenuItem) -> None:
            ModelMenuShared.ask_rolling_continuation(
                state=current, target=target, max_value=int(parameter.max), qualifier_line=qualifier_line)

        return MenuItem(display, edit)

    @staticmethod
    def make_choice_item(
            state: State, parameter: ChoiceSetting, label: str, source: CatalogMenuSource, *, subheading: str = "",
    ) -> MenuItem:
        """A fixed-choice string setting, edited through a one-shot option submenu."""
        return ModelMenuShared.make_choice_item(
            state, parameter, label,
            subheading=subheading,
            get_value=lambda project: source.get_choice(project, parameter),
            set_value=lambda project, value: source.set_choice(project, parameter, value),
        )

    @staticmethod
    def menu(state: State) -> None:
        ModelMenuShared.menu_wrapper(state, ModelMenuShared.make_remote_items)
