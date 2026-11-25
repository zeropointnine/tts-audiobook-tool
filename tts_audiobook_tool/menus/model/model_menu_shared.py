"""Routing and shared controls for model settings menus."""
from dataclasses import dataclass
import os
from typing import Callable

from tts_audiobook_tool import ask
from tts_audiobook_tool.app_types.model_target import ModelTarget, looks_like_local_path, parse_hf_input
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_types import Hint
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuItemListOrMaker, MenuUtil, StringOrMaker, get_string_from
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.menus.model.catalog_menu_source import ChoiceSetting
from tts_audiobook_tool.tts_models.local_definition import get_local_definition
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.util import *


@dataclass(frozen=True)
class TargetPromptSpec:
    """What `ModelMenuShared.ask_target` shows for one kind of target."""
    # Names the target in prompts, eg "OmniVoice model", "OmniVoice LoRA adapter"
    subject: str
    # An example repo id ("" for none)
    hf_example: str = ""
    # Shown only when entering a Hugging Face repo, eg a limitation of hf-hosted adapters
    hf_hint: str = ""
    # Shown only when selecting a local directory, eg the files it must contain
    local_hint: str = ""
    # Shown either way
    note: str = ""
    # Files that make a directory acceptable (`ModelTarget.exist_test`); None: the generic markers
    local_file_markers: list[str] | None = None


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
            initial_hotkey = ""
            if Tts.is_remote_mode():
                from tts_audiobook_tool.menus.model.model_select_menu import ModelSelectMenu
                if ModelSelectMenu.has_multiple_choices() and ModelSelectMenu.is_selection_required(state):
                    # Nothing else to do here until a model is chosen, so go
                    # straight to the picker (always item "1"; see make_remote_items).
                    initial_hotkey = "1"
            ModelMenuShared.menu_wrapper(
                state, ModelMenuShared.make_remote_items, initial_hotkey=initial_hotkey
            )
            return
        # Only local models reach here; remote variants are always handled
        # by the remote menu above. A local model whose catalog entry declares
        # `[[models.local.menu]]` uses the shared catalog renderer.
        local_definition = get_local_definition(model_type.id)
        if local_definition is not None and local_definition.has_menu:
            ModelMenuShared.menu_wrapper(state, ModelMenuShared.make_local_items)
            return
        match state.project.get_tts_model_type().id:
            case "chatterbox_local":
                from tts_audiobook_tool.menus.model.model_chatterbox_menu import ModelChatterboxMenu
                ModelChatterboxMenu.menu(state)
            case "dots_local":
                from tts_audiobook_tool.menus.model.model_dots_menu import ModelDotsMenu
                ModelDotsMenu.menu(state)
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
        stale controls. In remote mode, the TTS model picker comes first only
        when multiple selectable entries are available.
        """
        items: list[MenuItem] = []
        if Tts.is_remote_mode():
            from tts_audiobook_tool.menus.model.model_select_menu import ModelSelectMenu
            if ModelSelectMenu.has_multiple_choices():
                items.append(ModelSelectMenu.make_item(state))
        model_type = state.project.get_tts_model_type()
        if Tts.is_remote_mode() and model_type.value.backend_kind is TtsBackendKind.LOCAL:
            return items
        definition = Tts.get_audio_cpp_definition(model_type) or Tts.get_configured_definition(model_type)
        if definition is not None:
            from tts_audiobook_tool.menus.model.model_catalog_menu import ModelCatalogMenu
            settings = ModelCatalogMenu.make_items(state, definition)
        else:
            settings = []
        if items and settings:
            settings[0].blank_line_before = True
        return items + settings

    @staticmethod
    def make_local_items(state: State) -> list[MenuItem]:
        """Catalog-declared controls for a compatible local model, resolved per redraw."""
        if Tts.is_remote_mode():
            return []
        definition = get_local_definition(state.project.get_tts_model_type().id)
        if definition is None or not definition.has_menu:
            return []
        from tts_audiobook_tool.menus.model.model_catalog_menu import ModelCatalogMenu
        return ModelCatalogMenu.make_items(state, definition)

    @staticmethod
    def menu_wrapper(
            state: State,
            items: MenuItemListOrMaker,
            subheading: StringOrMaker | None = None,
            initial_hotkey: str = "",
    ) -> None:
        """
        Simple wrapper with standardized heading, model note and exit callback

        :param initial_hotkey: Item to activate on entry (see `MenuUtil.menu`)
        """
        def make_subheading(current: State) -> str:
            model_type = current.project.get_tts_model_type()
            incompatible = Tts.is_remote_mode() and model_type.value.backend_kind is TtsBackendKind.LOCAL
            note = "" if incompatible else model_type.value.ui.get("settings_note", "").strip()
            existing = get_string_from(current, subheading) if subheading and not incompatible else ""
            return "\n\n".join(part for part in (note, existing) if part)

        MenuUtil.menu(
            state=state,
            heading="Model settings",
            items=items,
            subheading=make_subheading,
            on_exit=lambda: PlaySoundUtil.stop_sound_async(),
            breadcrumb="Model",
            initial_hotkey=initial_hotkey,
        )

    @staticmethod
    def with_prompt_suffix(prompt: str, suffix: str) -> str | list[str]:
        """Give a dimmed proviso its own prompt line, matching catalog `input_prompt_suffix`."""
        return [prompt, f"{COL_DIM}{suffix}{COL_DEFAULT}"] if suffix else prompt

    @staticmethod
    def ask_temperature(
            state: State,
            target: str | SettingRef,
            prompt: str | list[str],
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
            hint: Hint | None = None,
            prompt_suffix: str = ""
    ) -> MenuItem:

        prompt = ModelMenuShared.with_prompt_suffix("Enter temperature", prompt_suffix)

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
            max_value: int=TOP_K_MAX_DEFAULT,
            prompt_suffix: str = ""
    ) -> MenuItem:

        return MenuUtil.make_number_item(
            state=state,
            target=target,
            base_label="Top_K",
            default_value=default_value,
            is_minus_one_default=True,
            num_decimals=0,
            prompt=ModelMenuShared.with_prompt_suffix("Enter Top-K", prompt_suffix),
            min_value=min_value,
            max_value=max_value
        )

    @staticmethod
    def make_top_p_item(
            state: State,
            target: str | SettingRef,
            default_value: float,
            prompt_suffix: str = ""
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
            prompt=ModelMenuShared.with_prompt_suffix("Enter Top-P", prompt_suffix),
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
            prompt=f"Enter repetition penalty",
            min_value=min_value,
            max_value=max_value
        )

    @staticmethod
    def make_choice_item(
            state: State,
            parameter: ChoiceSetting,
            label: str,
            get_value: Callable[[Project], object],
            set_value: Callable[[Project, str], str],
            *, subheading: str = "",
    ) -> MenuItem:
        """A fixed-choice string setting, edited through a one-shot option submenu.

        Shared by the server and local catalog menus. `get_value` returns the validated stored
        choice (raising ValueError when storage is invalid); `set_value` saves
        and returns an error message or "".
        """
        labels = {choice.value: choice.label for choice in parameter.choices}

        def valid_value(current: State) -> str | None:
            """The stored choice, or None when storage holds an invalid value."""
            try:
                return str(get_value(current.project))
            except ValueError:
                return None

        def display(current: State) -> str:
            value = valid_value(current)
            if value is None:
                # Show the broken value rather than the default it would
                # otherwise masquerade as. Server readiness blocks on it; local
                # choices are checked on load, and local inference falls back
                # to the default.
                raw = current.project.get_model_setting(parameter.model_id, parameter.name)
                return f"{label} {COL_DIM}({COL_ERROR}invalid: {raw!r}{COL_DIM})"
            return make_menu_label(label, labels[value], labels[parameter.default])

        def on_select(current: State, value: str) -> None:
            err = set_value(current.project, value)
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
                subheading=subheading,
            )

        return MenuItem(display, edit)

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
            spec: TargetPromptSpec,
            current_target: ModelTarget | None,
            callback: Callable[[Project, ModelTarget], None],
            on_clear: Callable[[], None] | None = None,
            preset_targets: list[ModelTarget] | None = None,
            show_subject: bool = True,
    ) -> None:
        """
        Asks the user for a so-called "target": first its kind (a hotkey: L for a local
        directory, H for a Hugging Face repo; C clears an existing target via `on_clear`
        when supplied; any other key aborts), then a directory
        (the app's directory request) or a repo id. Aborts on an empty entry or a
        cancelled dialog. Calls back only with a target that differs from
        `current_target` and passes its existence check (`ModelTarget.exist_test`,
        with `spec.local_file_markers`).

        A `current_target` among `preset_targets` isn't a custom one: the prompt then shows
        no current value, offers no clear and prefills nothing.

        `show_subject=False` omits the `spec.subject` heading, for callers whose menu
        heading already names it.
        """
        custom_target = ModelMenuShared.custom_target_of(current_target, preset_targets or [])
        if show_subject:
            printt(spec.subject)
        if custom_target is not None:
            printt(f"{COL_DIM}Currently: {custom_target.value} ({custom_target.kind.label}){COL_DEFAULT}")
        can_clear = custom_target is not None and on_clear is not None
        prompt = f"Press {make_hotkey_string('L')} to select a local directory, "
        if can_clear:
            prompt += (f"{make_hotkey_string('H')} to enter a Hugging Face repo id, "
                       f"or {make_hotkey_string('C')} to clear: ")
        else:
            prompt += f"or {make_hotkey_string('H')} to enter a Hugging Face repo id: "
        hotkey = ask.ask_hotkey(prompt)
        printt()
        if hotkey == "c" and can_clear:
            assert on_clear is not None
            on_clear()
            return
        if hotkey == "l":
            new_target = ModelMenuShared._ask_local_target(spec, custom_target)
        elif hotkey == "h":
            new_target = ModelMenuShared._ask_hf_target(spec, custom_target)
        else:
            return
        if new_target is None:
            return

        # Against the stored target, so re-entering a selected preset's id is still no change
        if new_target.is_same(current_target):
            print_feedback("Already set")
            return

        _, err = new_target.exist_test(spec.local_file_markers)
        if err:
            ask.ask_error(err)
            return

        callback(project, new_target)

    @staticmethod
    def custom_target_of(current_target: ModelTarget | None, preset_targets: list[ModelTarget]) -> ModelTarget | None:
        """`current_target`, unless it's one of `preset_targets` (then None)."""
        if current_target is None or any(current_target.is_same(target) for target in preset_targets):
            return None
        return current_target

    @staticmethod
    def _ask_local_target(spec: TargetPromptSpec, current_target: ModelTarget | None) -> ModelTarget | None:
        # The dialog can't show hints, so they precede it on the console
        for hint in (spec.local_hint, spec.note):
            if hint:
                printt(f"{COL_DIM}{hint}")
        # Start beside the current directory, where sibling checkpoints usually are
        initialdir = ""
        if current_target is not None and current_target.is_local:
            parent = os.path.dirname(os.path.normpath(current_target.value))
            initialdir = parent if os.path.isdir(parent) else ""
        path = ask.ask_dir_path(
            console_message=f"Enter local directory path of {spec.subject}:",
            dialog_title=f"Select {spec.subject} directory",
            initialdir=initialdir,
        )
        return ModelTarget.local(path) if path else None

    @staticmethod
    def _ask_hf_target(spec: TargetPromptSpec, current_target: ModelTarget | None) -> ModelTarget | None:
        # The subject is already named above (the `ask_target` heading, or the caller's menu)
        printt("Enter Hugging Face repo id")
        if spec.hf_example:
            printt(f"{COL_DIM}Eg, \"{spec.hf_example}\"")
        printt(f"{COL_DIM}For a subdirectory within the repo, append its path: \"owner/repo/sub/dir\"")
        for hint in (spec.hf_hint, spec.note):
            if hint:
                printt(f"{COL_DIM}{hint}")
        prefill = current_target.value if current_target is not None and current_target.is_hf else ""
        entered = ask.ask_input(prefill=prefill, lower=False)
        value, err = parse_hf_input(entered)
        if err:
            if looks_like_local_path(entered.strip().strip("\"'")):
                err += f"\nFor a local directory, press {make_hotkey_string('L')} instead"
            ask.ask_error(err)
            return None
        return ModelTarget.hf(value) if value else None

    @staticmethod
    def make_target_label(
            label_prefix: str,
            target: ModelTarget | None,
            default_target: ModelTarget,
            remove_prefixes: list[str] | None = None,
            extra_suffix: str = "",
    ) -> str:
        value = (target or default_target).value
        default_value = default_target.value

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
            preset_targets: list[ModelTarget],
            current_target: ModelTarget | None,
            default_target: ModelTarget,
            ask_custom_target: Callable[[], None],
            apply_target: Callable[[ModelTarget], None],
            sublabels: list[str] | None = None,
            custom_label: str = "Enter custom model",
            custom_hint: str = "(local directory or Hugging Face repo)",
            breadcrumb: str | None = None,
    ) -> None:
        if sublabels and len(sublabels) != len(preset_targets):
            raise ValueError("sublabels and preset_targets lists must have same size")

        # With no stored target, the default is what runs, so it shows as selected
        effective_target = current_target or default_target

        def make_preset_label(target: ModelTarget) -> str:
            label = target.value
            if target == default_target:
                label += f" {COL_DIM}(default)"
            if target == effective_target:
                label += f" {COL_ACCENT}(selected)"
            return label

        def make_custom_label() -> str:
            label = custom_label
            custom_target = ModelMenuShared.custom_target_of(current_target, preset_targets)
            if custom_target is not None:
                # The current value already shows what kind of entry is expected
                value = ellipsize_path_for_menu(custom_target.value)
                label += f" {COL_DIM}(currently: {COL_ACCENT}{value}{COL_DIM})"
            elif custom_hint:
                label += f" {custom_hint}"
            return label

        def apply_preset_if_changed(target: ModelTarget) -> None:
            if target.is_same(effective_target):
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
