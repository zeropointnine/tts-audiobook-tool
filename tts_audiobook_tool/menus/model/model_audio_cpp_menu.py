"""Model settings for the selected audio.cpp server family."""
from __future__ import annotations

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM, COL_ERROR
from tts_audiobook_tool.constants_hints import HINT_MOSS_TEMPERATURE
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.audio_cpp_behavior import get_audio_cpp_behavior
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppSettings
from tts_audiobook_tool.tts_models.audio_cpp_definition import (
    AudioCppMenuControl,
    AudioCppModelDefinition,
    AudioCppParameter,
    AudioCppTextParameter,
)
from tts_audiobook_tool.util import make_menu_label, print_feedback


class AudioCppMenuContext:
    """Per-redraw view of the model's behavior hooks, shared by both menus.

    Values are resolved once per redraw. When any stored value is invalid the
    hooks are skipped: every control shows (so a broken value can never hide
    the item that fixes it) and nothing is marked required.
    """

    def __init__(self, state: State, definition: AudioCppModelDefinition):
        self.behavior = get_audio_cpp_behavior(definition)
        try:
            self.values: dict[str, int | float | str] | None = {
                name: AudioCppSettings.get(state.project, parameter)
                for name, parameter in definition.parameters.items()
            }
        except ValueError:
            self.values = None

    def is_visible(self, control: AudioCppMenuControl) -> bool:
        if control.kind == "voice_samples" or self.values is None:
            return True
        return self.behavior.is_control_visible(control, self.values)

    def required_label(self, name: str) -> str | None:
        if self.values is None:
            return None
        return self.behavior.get_required_label(name, self.values)


class ModelAudioCppMenu:
    @staticmethod
    def make_items(state: State, definition: AudioCppModelDefinition) -> list[MenuItem]:
        context = AudioCppMenuContext(state, definition)
        items: list[MenuItem] = []
        for control in definition.menu:
            if control.target_menu == "model" and context.is_visible(control):
                items.extend(ModelAudioCppMenu.make_control_items(state, definition, control, context))
        return items

    @staticmethod
    def make_control_items(
            state: State, definition: AudioCppModelDefinition, control: AudioCppMenuControl,
            context: AudioCppMenuContext | None = None,
    ) -> list[MenuItem]:
        """Expand a settings control identically in either destination menu."""
        if control.kind == "seed":
            return [ModelMenuShared.make_seed_item(state, SettingRef(definition.spec.id, "seed"))]
        if control.kind == "voice_instructions":
            is_omnivoice = definition.family == "omnivoice"
            return ModelMenuShared.make_voice_instructions_item(
                state, definition.spec.id, control.parameter,
                label="Voice design instructions" if is_omnivoice else "Instructions",
                validate_omnivoice=is_omnivoice,
                required_label=context.required_label(control.parameter) if context is not None else None)
        parameter = definition.parameters[control.parameter]
        if control.kind == "choice":
            if not isinstance(parameter, AudioCppTextParameter) or not parameter.choices:
                raise ValueError(f"{definition.spec.id}.{control.parameter} is not a choice parameter")
            return [ModelAudioCppMenu.make_choice_item(state, parameter, control.label)]
        if isinstance(parameter, AudioCppTextParameter):
            # The catalog parser only routes string parameters to the
            # `voice_instructions` and `choice` controls above.
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
    def make_choice_item(state: State, parameter: AudioCppTextParameter, label: str) -> MenuItem:
        """A fixed-choice string setting, edited through a one-shot option submenu."""
        labels = {choice.value: choice.label for choice in parameter.choices}

        def valid_value(current: State) -> str | None:
            """The stored choice, or None when storage holds an invalid value."""
            try:
                return str(AudioCppSettings.get(current.project, parameter))
            except ValueError:
                return None

        def display(current: State) -> str:
            value = valid_value(current)
            if value is None:
                # Show the broken value rather than the default it would
                # otherwise masquerade as; readiness blocks on it.
                raw = current.project.get_model_setting(parameter.model_id, parameter.name)
                return f"{label} {COL_DIM}({COL_ERROR}invalid: {raw!r}{COL_DIM})"
            return make_menu_label(label, labels[value], labels[parameter.default])

        def on_select(current: State, value: str) -> None:
            err = AudioCppSettings.set(current.project, parameter, value)
            if err:
                ask.ask_error(err)
                return
            print_feedback(f"{label} set:", labels[value])

        def edit(current: State, _: MenuItem) -> None:
            MenuUtil.options_menu(
                state=current,
                heading_text=label,
                labels=[choice.label for choice in parameter.choices],
                values=[choice.value for choice in parameter.choices],
                # None when invalid: nothing shows as selected, so every
                # option (including the default) saves and repairs storage.
                current_value=valid_value(current),
                default_value=parameter.default,
                on_select=lambda value: on_select(current, value),
                sublabels=([choice.description for choice in parameter.choices]
                           if any(choice.description for choice in parameter.choices) else None),
                breadcrumb=label,
            )

        return MenuItem(display, edit)

    @staticmethod
    def menu(state: State) -> None:
        ModelMenuShared.menu_wrapper(state, ModelMenuShared.make_remote_items)
