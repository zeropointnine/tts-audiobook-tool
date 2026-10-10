"""Routing and shared controls for model settings menus."""
from typing import Callable

from tts_audiobook_tool import ask, target_util
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_types import Hint
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuItemListOrMaker, MenuUtil, StringOrMaker, get_string_from
from tts_audiobook_tool.menus.voice import voice_instruct_util
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *

class ModelMenuShared:
    @staticmethod
    def menu(state: State) -> None:
        """
        Delegate to the settings menu for the selected backend/model.

        In remote mode, a single menu hosts the model picker plus the
        selected model's controls, including when no model is selected yet.
        """
        model_type = state.project.get_tts_model_type()
        if (Tts.is_remote_mode()
                or Tts.get_audio_cpp_definition(model_type) is not None
                or Tts.get_configured_definition(model_type) is not None):
            ModelMenuShared.menu_wrapper(state, ModelMenuShared.make_remote_items)
            return
        # Only local models reach the legacy per-model menus; remote
        # variants are always handled by the remote menu above.
        match state.project.get_tts_model_type().id:
            case "chatterbox_local":
                from tts_audiobook_tool.menus.model.model_chatterbox_menu import ModelChatterboxMenu
                ModelChatterboxMenu.menu(state)
            case "dots_local":
                from tts_audiobook_tool.menus.model.model_dots_menu import ModelDotsMenu
                ModelDotsMenu.menu(state)
            case "fish_s1_local":
                from tts_audiobook_tool.menus.model.model_fish_s1_menu import ModelFishS1Menu
                ModelFishS1Menu.menu(state)
            case "fish_s2_local":
                from tts_audiobook_tool.menus.model.model_fish_s2_menu import ModelFishS2Menu
                ModelFishS2Menu.menu(state)
            case "glm_local":
                from tts_audiobook_tool.menus.model.model_glm_menu import ModelGlmMenu
                ModelGlmMenu.menu(state)
            case "higgs_v2_local":
                from tts_audiobook_tool.menus.model.model_higgs_v2_menu import ModelHiggsV2Menu
                ModelHiggsV2Menu.menu(state)
            case "indextts2_local":
                from tts_audiobook_tool.menus.model.model_indextts2_menu import ModelIndexTts2Menu
                ModelIndexTts2Menu.menu(state)
            case "mira_local":
                from tts_audiobook_tool.menus.model.model_mira_menu import ModelMiraMenu
                ModelMiraMenu.menu(state)
            case "moss_local":
                from tts_audiobook_tool.menus.model.model_moss_menu import ModelMossMenu
                ModelMossMenu.menu(state)
            case "omnivoice_local":
                from tts_audiobook_tool.menus.model.model_omnivoice_menu import ModelOmniVoiceMenu
                ModelOmniVoiceMenu.menu(state)
            case "pocket_local":
                from tts_audiobook_tool.menus.model.model_pocket_menu import ModelPocketMenu
                ModelPocketMenu.menu(state)

            case "qwen3tts_local":
                # Special case: Qwen model menu requires loaded model
                snapshot, _ = ModelWorker.get_model_state_blocking()
                already_loaded = (
                    snapshot is not None
                    and snapshot.tts_loaded
                    and snapshot.tts_type_id == state.project.get_tts_model_type().value.id
                )
                if not already_loaded:
                    printt(f"{COL_DIM_ITALICS}Initializing TTS model...")
                    printt()
                inspection, error = ModelWorker.inspect_tts_blocking(state)
                if error or inspection is None:
                    ask.ask_error(error or "Couldn't inspect Qwen3-TTS model")
                    return
                from tts_audiobook_tool.menus.model.model_qwen3_menu import ModelQwen3Menu
                ModelQwen3Menu.menu(state, inspection)

            case "vibevoice_local":
                from tts_audiobook_tool.menus.model.model_vibevoice_menu import ModelVibeVoiceMenu
                ModelVibeVoiceMenu.menu(state)
            case _:
                raise NotImplementedError(f"value: {state.project.get_tts_model_type()}")

    @staticmethod
    def make_remote_items(state: State) -> list[MenuItem]:
        """Resolve the current selection on every redraw, even across backends.

        Menu reconciliation (or the picker) can change the selection while this
        menu is open. Only the controls for this render retain a definition,
        never the menu factory itself. An unselected/unknown model offers no
        stale controls. In remote mode, the TTS model picker comes first.
        """
        items: list[MenuItem] = []
        if Tts.is_remote_mode():
            from tts_audiobook_tool.menus.model.model_select_menu import ModelSelectMenu
            items.append(ModelSelectMenu.make_item(state))
        model_type = state.project.get_tts_model_type()
        audio_definition = Tts.get_audio_cpp_definition(model_type)
        definition = Tts.get_configured_definition(model_type)
        if audio_definition is not None:
            from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu
            settings = ModelAudioCppMenu.make_items(state, audio_definition)
        elif definition is not None:
            from tts_audiobook_tool.menus.model.model_configured_sgl_omni_menu import ModelConfiguredSglOmniMenu
            settings = ModelConfiguredSglOmniMenu.make_items(state, definition)
        else:
            settings = []
        if items and settings:
            settings[0].blank_line_before = True
        return items + settings

    @staticmethod
    def menu_wrapper(
            state: State,
            items: MenuItemListOrMaker,
            subheading: StringOrMaker | None = None,
    ) -> None:
        """
        Simple wrapper with standardized heading, model note and exit callback
        """
        def make_subheading(current: State) -> str:
            note = current.project.get_tts_model_type().value.ui.get("settings_note", "").strip()
            existing = get_string_from(current, subheading) if subheading else ""
            return "\n\n".join(part for part in (note, existing) if part)

        MenuUtil.menu(
            state=state,
            heading="Model settings",
            items=items,
            subheading=make_subheading,
            on_exit=lambda: PlaySoundUtil.stop_sound_async(),
            breadcrumb="Model",
        )

    @staticmethod
    def make_voice_instructions_item(
            state: State,
            model_id: str,
            name: str = "instruct",
            label: str = "Voice design instructions",
            validate_omnivoice: bool = False,
            required_label: str | None = None,
    ) -> list[MenuItem]:
        """
        Makes the voice-design/edit item for a string instruction setting, plus
        its "Clear" item while a value is stored.

        Shared by local and audio.cpp models. OmniVoice callers opt into their
        tag-based validation; other models accept free-form instructions.
        `required_label` (eg "required for Instruct") replaces "(optional)"
        while the value is empty, in the error color; the corresponding
        readiness blocker is what actually stops generation.
        """
        def make_label(current: State) -> str:
            value = current.project.get_model_setting(model_id, name)
            if not value:
                suffix = (f"{COL_DIM}({COL_ERROR}{required_label}{COL_DIM})" if required_label
                          else f"{COL_DIM}(optional)")
            else:
                suffix = make_currently_string(truncate_pretty(value, 40, content_color=COL_ACCENT))
            return f"{label} {suffix}"

        def on_clear(current: State, _: MenuItem) -> None:
            current.project.set_model_setting(model_id, name, None, reset=True)
            current.project.save()
            print_feedback("Instructions cleared")

        def on_edit(current: State, _: MenuItem) -> None:
            no_voice_note = ""
            if validate_omnivoice and ProjectVoiceUtil.get_primary_voice_value(current.project, TtsModelType.require_by_id(model_id)):
                no_voice_note = "Note: When used alongside voice cloning, instructions may have minimal effect"
            ModelMenuShared.ask_instruct(
                current.project, model_id, name, no_voice_note,
                label=label, validate_omnivoice=validate_omnivoice)

        items = [MenuItem(make_label, on_edit)]
        if state.project.get_model_setting(model_id, name):
            items.append(MenuItem("Clear instructions", on_clear))
        return items

    @staticmethod
    def ask_instruct(
            project: Project, model_id: str, name: str = "instruct", no_voice_note: str = "",
            *, label: str = "Voice design instructions", validate_omnivoice: bool = False,
    ) -> None:
        """Edit instructions with the current value prefilled.

        Only OmniVoice callers use its best-effort tag validation. Free-form
        instructions must not be checked against OmniVoice's vocabulary.
        """

        def validator(value: str) -> str:
            error, _ = voice_instruct_util.validate_instruct(value)
            return error

        prompt = [f"Enter {label.lower()}"]
        if validate_omnivoice:
            prompt.append(f"{COL_DIM}Eg: \"male, british accent, low pitch\" / \"female, young adult, high pitch\"")
        else:
            prompt.append(f"{COL_DIM}Eg: \"Speak warmly and naturally, with calm pacing.\"")
        if no_voice_note:
            prompt.append(f"{COL_DIM}{no_voice_note}")
        ask.ask_string_and_save(
            project,
            "\n".join(prompt),
            SettingRef(model_id, name),
            "Set instructions:",
            validator=validator if validate_omnivoice else None,
        )

    @staticmethod
    def ask_temperature(
            state: State,
            target: str | SettingRef,
            prompt: str,
            min_value: float,
            max_value: float,
            default_value: float,
            hint: Hint | None = None
    ) -> None:
        if hint:
            hints.show_hint_if_necessary(state.prefs, hint)

        ask.ask_number_and_save(
            state.project,
            target,
            prompt,
            min_value,
            max_value,
            default_value,
            "Value set:",
            is_int=False,
            is_minus_one_default=True,
        )

    @staticmethod
    def make_temperature_item(
            state: State,
            target: str | SettingRef,
            default_value: float,
            min_value: float,
            max_value: float,
            base_label: str="Temperature",
            hint: Hint | None = None
    ) -> MenuItem:

        prompt = "Enter temperature"

        def on_item(_: State, __: MenuItem) -> None:
            ModelMenuShared.ask_temperature(
                state=state,
                target=target,
                prompt=prompt,
                min_value=min_value,
                max_value=max_value,
                default_value=default_value,
                hint=hint
            )

        label = MenuUtil.make_number_label(
            project=state.project,
            target=target,
            base_label=base_label,
            default_value=default_value,
            is_minus_one_default=True,
            num_decimals=2
        )

        return MenuItem(label, on_item)

    @staticmethod
    def make_top_k_item(
            state: State,
            target: str | SettingRef,
            default_value: int,
            min_value: int=TOP_K_MIN_DEFAULT,
            max_value: int=TOP_K_MAX_DEFAULT
    ) -> MenuItem:

        return MenuUtil.make_number_item(
            state=state,
            target=target,
            base_label="Top_K",
            default_value=default_value,
            is_minus_one_default=True,
            num_decimals=0,
            prompt=f"Enter Top-K {COL_DIM}({min_value} to {max_value}){COL_DEFAULT}:",
            min_value=min_value,
            max_value=max_value
        )

    @staticmethod
    def make_top_p_item(
            state: State,
            target: str | SettingRef,
            default_value: float
    ) -> MenuItem:

        min_value = TOP_P_MIN_DEFAULT
        max_value = TOP_P_MAX_DEFAULT

        return MenuUtil.make_number_item(
            state=state,
            target=target,
            base_label="Top-P",
            default_value=default_value,
            is_minus_one_default=True,
            num_decimals=2,
            prompt=f"Enter Top-P {COL_DIM}({min_value} to {max_value}){COL_DEFAULT}:",
            min_value=min_value,
            max_value=max_value
        )

    @staticmethod
    def make_repetition_penalty_item(
            state: State,
            target: str | SettingRef,
            default_value: float,
            min_value = REPETITION_PENALTY_MIN_DEFAULT,
            max_value = REPETITION_PENALTY_MAX_DEFAULT
    ) -> MenuItem:

        return MenuUtil.make_number_item(
            state=state,
            target=target,
            base_label="Repetition penalty",
            default_value=default_value,
            is_minus_one_default=True,
            num_decimals=2,
            prompt=f"Enter repetition penalty {COL_DIM}({min_value} to {max_value}){COL_DEFAULT}:",
            min_value=min_value,
            max_value=max_value
        )

    @staticmethod
    def make_seed_item(
            state: State,
            target: str | SettingRef,
            prompt_override: str="",
            add_batch_warning: bool=False
    ) -> MenuItem:
        """ Makes "self-contained" menu item for seed setting, including handler """

        if prompt_override:
            prompt = prompt_override
        else:
            prompt = f"Enter a static seed value {COL_DIM}(or -1 for random){COL_DEFAULT}"
            if not add_batch_warning:
                prompt += ": "

        if add_batch_warning:
            prompt += f"\n{COL_DIM}(Note, audio generations are not idempotent when using batch mode): "

        def on_item(_: State, __: MenuItem) -> None:
            ask.ask_number_and_save(
                saveable=state.project,
                target=target,
                prompt=prompt,
                min_value=-1,
                max_value=2**32-1,
                default_value=-1,
                success_prefix="Seed set:",
                is_int=True,
                print_range_info=False
            )

        seed_value: int | None = ask._get_saveable_attr(state.project, target)
        if seed_value is None:
            raise ValueError(f"Attribute doesn't exist: {target}")

        suffix = str(seed_value) if seed_value != -1 else "random"
        label = make_menu_label("Seed", suffix)
        return MenuItem(label, on_item)

    @staticmethod
    def ask_target(
            project: Project,
            prompt: str,
            current_target: str,
            callback: Callable[[Project, str], None]
    ) -> None:
        """ Gets line text input from user for a so-called "target" (ie, repo id or concrete file path) """
        printt(prompt)
        new_target = ask.ask_input(prefill=current_target, lower=False)
        if not new_target:
            return

        if target_util.is_same_target(current_target, new_target):
            print_feedback("Already set")
            return

        _, err = target_util.exist_test(new_target)
        if err:
            ask.ask_error(err)
            return

        callback(project, new_target)

    @staticmethod
    def make_target_label(
            label_prefix: str,
            target: str,
            default_target: str,
            remove_prefixes: list[str] | None = None,
            extra_suffix: str = "",
    ) -> str:
        value = target or default_target
        default_value = default_target

        for prefix in remove_prefixes or []:
            value = value.removeprefix(prefix)
            default_value = default_value.removeprefix(prefix)

        value = ellipsize_path_for_menu(value)
        default_value = ellipsize_path_for_menu(default_value)
        label = make_currently_string(value, default=default_value)
        return f"{label_prefix} {label}{extra_suffix}"

    @staticmethod
    def target_submenu(
            state: State,
            heading: str,
            preset_targets: list[str],
            current_target: str,
            default_target: str,
            ask_custom_target: Callable[[], None],
            apply_target: Callable[[str], None],
            sublabels: list[str] | None = None,
            custom_label: str = "Enter custom hf repo id or local path",
            breadcrumb: str | None = None,
    ) -> None:
        if sublabels and len(sublabels) != len(preset_targets):
            raise ValueError("sublabels and preset_targets lists must have same size")

        def make_preset_label(target: str) -> str:
            label = target
            if target == default_target:
                label += f" {COL_DIM}(default)"
            if target == current_target:
                label += f" {COL_ACCENT}(selected)"
            return label

        def make_custom_label() -> str:
            label = custom_label
            is_custom = bool(current_target) and not any(
                target_util.is_same_target(current_target, target) for target in preset_targets
            )
            if is_custom:
                value = ellipsize_path_for_menu(current_target)
                label += f" {COL_DIM}(currently: {COL_ACCENT}{value}{COL_DIM})"
            return label

        def apply_preset_if_changed(target: str) -> None:
            if target_util.is_same_target(current_target, target):
                return
            apply_target(target)

        items = []
        for i, target in enumerate(preset_targets):
            item = MenuItem(
                make_preset_label(target),
                lambda _, __, target=target: apply_preset_if_changed(target),
            )
            if sublabels:
                item.sublabel = sublabels[i]
            items.append(item)

        items.append(MenuItem(make_custom_label(), lambda _, __: ask_custom_target()))

        MenuUtil.menu(
            state=state,
            heading=heading,
            items=items,
            one_shot=True,
            breadcrumb=breadcrumb,
        )

    @staticmethod
    def make_rolling_continuation_label(value: int) -> str:
        if value > 0:
            val = f"enabled, length {value}"
        else:
            val = f"disabled {COL_DIM}default"
        return "Rolling continuation " + make_currently_string(val)

    @staticmethod
    def ask_rolling_continuation(state: State, target: str | SettingRef, max_value: int, qualifier_line: str="") -> None:
        """
        :param qualifier_line: Should describe any prereqs (eg, batch size 1)
        """
        subheading = ROLLING_CONTINUATION_DESC
        if qualifier_line:
            subheading += f"\n\n{qualifier_line}"

        MenuUtil.print_screen_heading(
            state,
            f"Rolling continuation {COL_DIM}(experimental)",
            subheading=subheading
        )
        ask.ask_number_and_save(
            state.project, target, "Enter value",
            min_value=0, max_value=max_value, default_value=0,
            success_prefix="Rolling continuation num segments set to", is_int=True
        )
