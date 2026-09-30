import os
from typing import Callable

from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_types import Hint, SttVariant, VoiceSelectMode
from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuItemListOrMaker, MenuUtil, StringOrMaker, get_string_from
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    EditorSaveFailed,
    EditorSaved,
    run_content_textual_app,
)
from tts_audiobook_tool.textual.voice_line_editor import VoiceLineEditorTextualApp
from tts_audiobook_tool.menus.voice import voice_instruct_util
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY, SettingRef
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool import target_util
from tts_audiobook_tool.constants import VOICE_ADVANCED_SUPERLABEL
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.transcriber import Transcriber

# Maps a definition-driven menu control's declared group name to the superlabel
# rendered above that group. See VoiceMenuShared.apply_group_superlabels(...).
VOICE_GROUP_SUPERLABELS: dict[str, str] = {"advanced": VOICE_ADVANCED_SUPERLABEL}

class VoiceMenuShared:

    @staticmethod
    def menu(state: State) -> None:
        """
        Simply delegates to the correct model-specific voice menu
        """
        audio_definition = Tts.get_audio_cpp_definition(state.project.get_tts_model_type())
        if audio_definition is not None:
            from tts_audiobook_tool.menus.voice.voice_audio_cpp_menu import VoiceAudioCppMenu
            VoiceAudioCppMenu.menu(state)
            return
        definition = Tts.get_configured_definition(state.project.get_tts_model_type())
        if definition is not None:
            from tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu import VoiceConfiguredSglOmniMenu
            VoiceConfiguredSglOmniMenu.menu(state)
            return
        # Only local models reach the legacy per-model menus; SGL-Omni
        # variants are always handled by the configured menu above.
        match state.project.get_tts_model_type().id:
            case "chatterbox_local":
                from tts_audiobook_tool.menus.voice import VoiceChatterboxMenu
                VoiceChatterboxMenu.menu(state)
            case "dots_local":
                from tts_audiobook_tool.menus.voice import VoiceDotsMenu
                VoiceDotsMenu.menu(state)
            case "fish_s1_local":
                from tts_audiobook_tool.menus.voice import VoiceFishS1Menu
                VoiceFishS1Menu.menu(state)
            case "fish_s2_local":
                from tts_audiobook_tool.menus.voice import VoiceFishS2Menu
                VoiceFishS2Menu.menu(state)
            case "glm_local":
                from tts_audiobook_tool.menus.voice import VoiceGlmMenu
                VoiceGlmMenu.menu(state)
            case "higgs_v2_local":
                from tts_audiobook_tool.menus.voice import VoiceHiggsV2Menu
                VoiceHiggsV2Menu.menu(state)
            case "indextts2_local":
                from tts_audiobook_tool.menus.voice import VoiceIndexTts2Menu
                VoiceIndexTts2Menu.menu(state)
            case "mira_local":
                from tts_audiobook_tool.menus.voice import VoiceMiraMenu
                VoiceMiraMenu.menu(state)
            case "moss_local":
                from tts_audiobook_tool.menus.voice import VoiceMossMenu
                VoiceMossMenu.menu(state)
            case "omnivoice_local":
                from tts_audiobook_tool.menus.voice import VoiceOmniVoiceMenu
                VoiceOmniVoiceMenu.menu(state)
            case "pocket_local":
                from tts_audiobook_tool.menus.voice import VoicePocketMenu
                VoicePocketMenu.menu(state)

            case "qwen3tts_local":
                # Special case: Qwen voice menu requires loaded model
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
                from tts_audiobook_tool.menus.voice.voice_qwen3_menu import VoiceQwen3Menu
                VoiceQwen3Menu.menu(state, inspection)

            case "vibevoice_local":
                from tts_audiobook_tool.menus.voice import VoiceVibeVoiceMenu
                VoiceVibeVoiceMenu.menu(state)
            case _:
                raise NotImplementedError(f"value: {state.project.get_tts_model_type()}")

    @staticmethod
    def make_remote_items(state: State) -> list[MenuItem]:
        """Resolve the current selection on every redraw, even across backends.

        Menu reconciliation can change the selection while this menu is open.
        Only the controls for this render retain a definition, never the menu
        factory itself. An unselected/unknown model offers no stale controls.
        """
        model_type = state.project.get_tts_model_type()
        audio_definition = Tts.get_audio_cpp_definition(model_type)
        if audio_definition is not None:
            from tts_audiobook_tool.menus.voice.voice_audio_cpp_menu import VoiceAudioCppMenu
            return VoiceAudioCppMenu.make_items(state, audio_definition)
        definition = Tts.get_configured_definition(model_type)
        if definition is not None:
            from tts_audiobook_tool.menus.voice.voice_configured_sgl_omni_menu import VoiceConfiguredSglOmniMenu
            return VoiceConfiguredSglOmniMenu.make_items(state, definition)
        return []

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
            heading="Voice clone and model settings",
            items=items,
            subheading=make_subheading,
            on_exit=lambda: PlaySoundUtil.stop_sound_async(),
            breadcrumb="Voice",
        )

    @staticmethod
    def apply_group_superlabels(items: list[MenuItem], groups: list[str]) -> None:
        """
        Renders each declared group's superlabel once, above that group's first item.

        MenuItem.superlabel is a per-item decoration, not a section header: the menu
        renderer reprints it before every item that carries one. Menus built from a
        model definition therefore must not stamp every member of a group, or the
        heading reappears between members that are not adjacent. Later members render
        beneath the single heading, which is the intended grouping.

        :param items: menu items in display order
        :param groups: parallel list of group names ("" for ungrouped items)
        """
        if len(items) != len(groups):
            raise ValueError("items and groups must be parallel lists")
        labelled: set[str] = set()
        for item, group in zip(items, groups):
            if not group or group in labelled:
                continue
            label = VOICE_GROUP_SUPERLABELS.get(group)
            if label is None:
                raise ValueError(f"No superlabel mapped for menu group {group!r}")
            labelled.add(group)
            item.superlabel = label

    @staticmethod
    def make_resolved_voice_label(state: State) -> str:
        if state.project.get_tts_model_type().value.requires_voice and not ProjectVoiceUtil.has_voice(state.project):
            currently = make_currently_string("required", value_prefix="", color_code=COL_ERROR)
        elif not ProjectVoiceUtil.has_voice(state.project):
            currently = make_currently_string("none", color_code=COL_ERROR)
        else:
            currently = make_currently_string(ProjectVoiceUtil.get_voice_label(state.project))
        return f"Add voice sample {currently}"

    @staticmethod
    def ask_and_set_voice_file(
            state: State,
            tts_type: TtsModelType,
            is_secondary: bool=False,
            message_override: str="",
            append: bool=False,
    ) -> None:
        """
        Asks for voice sound file path.
        Transcribes text if necessary.
        Saves to project.
        Prints feedback on success or fail.
        """

        if REGISTRY.voice_binding(tts_type.id) is None:
            raise ValueError(f"Unsupported tts type for this operation {tts_type}")

        if REGISTRY.transcript_binding(tts_type.id) is not None:
            hints.show_hint_if_necessary(state.prefs, HINT_VOICE_TRANSCRIPT)

        if state.prefs.last_voice_dir and not os.path.exists(state.prefs.last_voice_dir):
            state.prefs.last_voice_dir = ""
            state.prefs.save()
        path = VoiceMenuShared.ask_voice_file(state.prefs.last_voice_dir, tts_type, message_override)
        if not path:
            return

        if not os.path.exists(path) or not os.path.isfile(path):
            ask.ask_error(f"File doesn't exist: {path}")
            return

        state.prefs.last_voice_dir = str(Path(path).parent)
        state.prefs.save()

        # Load sound
        sound_result = SoundFileUtil.load(path)
        if isinstance(sound_result, str):
            err = sound_result
            ask.ask_error(err)
            return
        sound = sound_result

        sound = SoundPipeline.apply_voice_clone_post_processing(sound)

        if sound.data.size == 0:
            ask.ask_error("Selected sound sample is entirely silence")
            return

        duration_s = len(sound.data) / sound.sr
        printt(f"{COL_DIM}Playing selected sound sample ({duration_s:.1f}s)...")
        printt()
        PlaySoundUtil.play_sound_async(sound)

        force_enter_prompt = False

        transcript = ""
        if REGISTRY.transcript_binding(tts_type.id) is not None:

            # [1] Get transcript from 'parallel text file' if possible
            transcript_path = Path(path).with_suffix(".txt")
            if transcript_path.exists():
                transcript = text_util.load_text_file(str(transcript_path), errors="replace").strip()
                if transcript:
                    printt(f"Loaded transcript text from")
                    printt(f"{transcript_path}:")
                    printt(f"{COL_DIM_ITALICS}{transcript}")
                    printt()

            if not transcript:
                # [2] Transcribe sound file using STT
                printt(f"Transcribing... {COL_DIM}(language code: {state.project.language_code or 'none'})")
                printt()

                if state.prefs.stt_variant == SttVariant.DISABLED:
                    stt_variant = SttVariant.LARGE_V3
                else:
                    stt_variant = state.prefs.stt_variant
                sound_result = Transcriber.transcribe_to_words(
                    sound,
                    state.project.language_code,
                    stt_variant,
                    state.prefs.stt_config,
                    state,
                )

                if isinstance(sound_result, str):
                    err = sound_result
                    ask.ask_error(err)
                    return

                words = sound_result
                transcript = Transcriber.get_flat_text_filtered_by_probability(words, VOICE_CLONE_TRANSCRIBE_MIN_PROBABILITY)
                print(f"Transcribed text {COL_DIM}(low probability words filtered out){COL_DEFAULT}:")
                printt(f"{COL_DIM_ITALICS}{transcript}")
                printt()

                force_enter_prompt = True

        file_stem = Path(path).stem
        err = ProjectVoiceUtil.set_voice_and_save(
            state.project,
            sound,
            file_stem,
            transcript,
            tts_type,
            is_secondary=is_secondary,
            append=append,
        )
        if err:
            ask.ask_error(err)
            return

        print_feedback("Voice file saved")

        hints.show_hint_if_necessary(state.prefs, HINT_TEST_REAL_TIME, and_prompt=not force_enter_prompt)

        if force_enter_prompt:
            ask.ask_enter_to_continue()

    @staticmethod
    def make_manage_voice_samples_item(
            state: State,
            tts_type: TtsModelType,
            no_samples_label: StringOrMaker | None = None,
            on_before_set_callback: Callable | None = None,
            on_set_callback: Callable | None = None,
            on_clear_callback: Callable | None = None,
    ) -> MenuItem:

        def make_label(s: State) -> str:
            voices = ProjectVoiceUtil.get_voice_values(s.project, tts_type)
            if not voices:
                if no_samples_label:
                    return get_string_from(s, no_samples_label)
                return VoiceMenuShared.make_resolved_voice_label(s)

            first_label = ProjectVoiceUtil.make_voice_sample_display_label(s.project, voices[0], tts_type.value)
            suffix = first_label
            if len(voices) > 1:
                suffix += f", +{len(voices) - 1} more"
            currently = make_currently_string(suffix)
            return f"Add/remove voice samples {currently}"

        def on_item(s: State, __: MenuItem) -> None:
            voices = ProjectVoiceUtil.get_voice_values(s.project, tts_type)
            if not voices:
                if on_before_set_callback:
                    on_before_set_callback()
                VoiceMenuShared.ask_and_set_voice_file(s, tts_type)
                if on_set_callback:
                    on_set_callback()
                return
            # Remote availability may change while the nested menu is open.
            # Local menus can also manage explicit secondary/model references.
            sample_type = tts_type if tts_type.value.backend_kind is TtsBackendKind.LOCAL else None
            VoiceMenuShared.manage_voice_samples_submenu(
                s, sample_type, on_before_set_callback, on_set_callback, on_clear_callback
            )

        return MenuItem(make_label, on_item)

    @staticmethod
    def make_voice_instructions_item(
            state: State,
            model_id: str,
            name: str = "instruct",
            label: str = "Voice design instructions",
            validate_omnivoice: bool = False,
    ) -> list[MenuItem]:
        """
        Makes the voice-design/edit item for a string instruction setting, plus
        its "Clear" item while a value is stored.

        Shared by local and audio.cpp models. OmniVoice callers opt into their
        tag-based validation; other models accept free-form instructions.
        """
        def make_label(current: State) -> str:
            value = current.project.get_model_setting(model_id, name)
            if not value:
                suffix = f"{COL_DIM}(optional)"
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
            VoiceMenuShared.ask_instruct(
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
    def make_voice_sample_items(
            state: State,
            tts_type: TtsModelType,
            no_samples_label: StringOrMaker | None = None,
            on_before_set_callback: Callable | None = None,
            on_set_callback: Callable | None = None,
            on_clear_callback: Callable | None = None,
    ) -> list[MenuItem]:
        """
        Makes the manage-samples item and, when applicable, its selection-mode item.
        """
        items = [
            VoiceMenuShared.make_manage_voice_samples_item(
                state,
                tts_type,
                no_samples_label,
                on_before_set_callback,
                on_set_callback,
                on_clear_callback,
            )
        ]
        voices = ProjectVoiceUtil.get_voice_values(state.project, tts_type)
        if len(voices) > 1:
            items.append(VoiceMenuShared.make_voice_sample_selection_mode_item())
        items.append(VoiceMenuShared.make_assign_voice_samples_to_text_lines_item(tts_type))
        return items

    @staticmethod
    def make_voice_sample_selection_mode_item() -> MenuItem:
        def make_label(state: State) -> str:
            return make_menu_label(
                "Voice selection mode",
                state.project.voice_select_mode.current_label,
            )

        return MenuItem(
            make_label,
            lambda state, _: VoiceMenuShared.voice_sample_selection_mode_submenu(state),
        )

    @staticmethod
    def make_assign_voice_samples_to_text_lines_item(tts_type: TtsModelType) -> MenuItem:
        def make_label(state: State) -> str:
            label = "Edit voice selections"
            voices = ProjectVoiceUtil.get_voice_values(state.project, tts_type)
            if len(voices) < 2:
                label += f" {COL_DIM}(optional; requires 2+ voice samples)"
            elif state.project.voice_select_mode is not VoiceSelectMode.USER_DEFINED:
                label += f" {COL_DIM}(optional; requires \"voice selection mode: user-defined\")"
            return label

        def on_item(state: State, _: MenuItem) -> None:
            voices = ProjectVoiceUtil.get_voice_values(state.project, tts_type)
            if len(voices) < 2:
                print_feedback("Requires 2+ voice samples")
                return
            VoiceMenuShared.assign_voice_samples_to_text_lines(state)

        return MenuItem(
            make_label,
            on_item,
        )

    @staticmethod
    def assign_voice_samples_to_text_lines(state: State) -> None:
        run_result = run_content_textual_app(
            VoiceLineEditorTextualApp(state.project)
        )
        if not isinstance(run_result, ContentAppCompleted):
            ask.ask_error(run_result.message)
            return
        if isinstance(run_result.result, EditorSaveFailed):
            ask.ask_error(run_result.result.error)
        elif isinstance(run_result.result, EditorSaved):
            print_feedback("Saved changes", long_pause=True)

    @staticmethod
    def voice_sample_selection_mode_submenu(state: State) -> None:
        modes = list(VoiceSelectMode)

        def on_select(value: VoiceSelectMode) -> None:
            state.project.voice_select_mode = value
            state.project.save()
            print_feedback("Voice selection mode set to:", value.id)

        MenuUtil.options_menu(
            state=state,
            heading_text="Voice selection mode",
            labels=[mode.label for mode in modes],
            values=modes,
            current_value=state.project.voice_select_mode,
            default_value=VoiceSelectMode.get_default(),
            on_select=on_select,
            sublabels=[mode.description for mode in modes],
        )

    @staticmethod
    def make_voice_samples_subheading(project: Project, tts_type: TtsModelType) -> str:
        voices = ProjectVoiceUtil.get_voice_values(project, tts_type)
        lines = []
        for i, voice in enumerate(voices, start=1):
            label = ProjectVoiceUtil.make_voice_sample_display_label(project, voice, tts_type.value)
            lines.append(f"{COL_DIM}- Voice sample {i}: {COL_DEFAULT}{label}")
        return "\n".join(lines)

    @staticmethod
    def manage_voice_samples_submenu(
            state: State,
            tts_type: TtsModelType | None,
            on_before_set_callback: Callable | None=None,
            on_set_callback: Callable | None=None,
            on_clear_callback: Callable | None=None,
    ) -> None:

        def selected_type(s: State) -> TtsModelType:
            return tts_type if tts_type is not None else s.project.get_tts_model_type()

        def add_voice(s: State) -> None:
            if on_before_set_callback:
                on_before_set_callback()
            VoiceMenuShared.ask_and_set_voice_file(s, selected_type(s), append=True)
            if on_set_callback:
                on_set_callback()

        def remove_voice(s: State) -> bool:
            is_empty = VoiceMenuShared.remove_voice_sample_from_menu(s, selected_type(s))
            if is_empty and on_clear_callback:
                on_clear_callback()
            return is_empty

        def make_items(s: State) -> list[MenuItem]:
            model_type = selected_type(s)
            if REGISTRY.voice_binding(model_type.id) is None:
                return []
            items = []
            voices = ProjectVoiceUtil.get_voice_values(s.project, model_type)
            if len(voices) < 9:
                items.append(MenuItem(
                    "Add voice sample",
                    lambda state, __: add_voice(state),
                ))
            items.append(MenuItem(
                "Remove voice sample",
                lambda state, __: remove_voice(state),
            ))
            return items

        MenuUtil.menu(
            state=state,
            heading="Add/remove voice sample",
            items=make_items,
            subheading=lambda s: VoiceMenuShared.make_voice_samples_subheading(s.project, selected_type(s)),
            breadcrumb="Voice samples",
        )

    @staticmethod
    def remove_voice_sample_from_menu(state: State, tts_type: TtsModelType) -> bool:
        voices = ProjectVoiceUtil.get_voice_values(state.project, tts_type)
        if not voices:
            return True

        index = 0
        clear_all = False
        label = ""
        if len(voices) > 1:
            printt(f"Enter voice sample number to remove {COL_DIM}(or \"all\" to clear all)")
            inp = ask.ask_input()
            if not inp:
                return False
            if inp.lower() in {"a", "all"}:
                ProjectVoiceUtil.clear_voice_and_save(state.project, tts_type)
                label = "all"
                clear_all = True
            else:
                try:
                    index = int(inp) - 1
                except ValueError:
                    ask.ask_error("Bad value")
                    return False
                if index < 0 or index >= len(voices):
                    ask.ask_error("Bad value")
                    return False

        if not clear_all:
            removed = ProjectVoiceUtil.remove_voice_at_index_and_save(state.project, tts_type, index)
            label = ProjectVoiceUtil.make_voice_sample_display_label(state.project, removed, tts_type.value)
        print_feedback(f"Removed {label}")
        return not ProjectVoiceUtil.get_voice_values(state.project, tts_type)

    @staticmethod
    def ask_voice_file(initial_dir: str, tts_type: TtsModelType, message_override: str="") -> str:
        """
        Asks for voice file path.
        Validates file and shows error prompt if necessary.
        Returns path or empty string.
        """

        if message_override:
            console_message = message_override
            requestor_title = message_override
        else:
            ui = tts_type.value.ui
            console_message = ui.get("voice_path_console", "")
            requestor_title = ui.get("voice_path_requestor", "")

        # We set "prefill" here to a directory, not a file path
        # The hope is that this is more of a convenience than a hinderance
        path = ask.ask_file_path(
             console_message=console_message,
             dialog_title=requestor_title,
             filetypes=FILE_REQUESTOR_SOUND_TYPES,
             initialdir=initial_dir,
             prefill=initial_dir
        )
        if not path:
            return ""

        if not os.path.exists(path):
            ask.ask_error(f"File not found: {path}")
            return ""

        err = SoundFileUtil.is_valid_sound_file(path)
        if err:
            ask.ask_error(err)
            return ""

        return path

    @staticmethod
    def make_clear_voice_item(state: State, info_item: TtsModelType, callback: Callable | None=None) -> MenuItem: # type: ignore

        def on_clear_voice(_: State, item: MenuItem) -> None:
            info_item: TtsModelType = item.data
            ProjectVoiceUtil.clear_voice_and_save(state.project, info_item, is_secondary=False)
            if callback:
                callback()
            print_feedback("Cleared")

        return MenuItem("Clear voice clone sample", on_clear_voice, data=info_item)

    # ---

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
            VoiceMenuShared.ask_temperature(
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
