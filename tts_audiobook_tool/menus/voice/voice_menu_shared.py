import os
from typing import TYPE_CHECKING, Callable

from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_types import SttVariant, VoiceSelectMode
from tts_audiobook_tool import ask

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuItemListOrMaker, MenuUtil, StringOrMaker, get_string_from
from tts_audiobook_tool.textual.content_textual_app import (
    ContentAppCompleted,
    EditorSaveFailed,
    EditorSaved,
    run_content_textual_app,
)
from tts_audiobook_tool.textual.voice_line_editor import VoiceLineEditorTextualApp
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.transcriber import Transcriber

# Menu item labels shared by per-model voice menus; tests reference these.
LABEL_ADD_VOICE_SAMPLE = "Add voice sample"
LABEL_REMOVE_VOICE_SAMPLE = "Remove voice sample"
LABEL_MOVE_VOICE_SAMPLE = "Move voice sample position"
LABEL_CROP_VOICE_SAMPLE = "Trim or play voice sample"
LABEL_EDIT_VOICE_TRANSCRIPTION = "Edit voice sample transcription"
LABEL_VOICE_SELECTION_MODE = "Voice selection mode"
LABEL_EDIT_VOICE_SELECTIONS = "Edit voice/line selections"

# Minimum duration of an imported voice sample, checked before and after silence trim
VOICE_SAMPLE_MIN_DURATION_S = 2.0


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
                from tts_audiobook_tool.menus.voice.voice_qwen3_menu import VoiceQwen3Menu
                VoiceQwen3Menu.menu(state)

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
        """Show sample management independently of model settings."""
        def make_subheading(current: State) -> str:
            samples = VoiceMenuShared.make_voice_samples_subheading(
                current.project, current.project.get_tts_model_type()
            )
            existing = get_string_from(current, subheading).rstrip("\r\n") if subheading else ""
            return "\n\n".join(part for part in (samples, existing) if part)

        MenuUtil.menu(
            state=state,
            heading="Voice clone",
            items=items,
            subheading=make_subheading,
            on_exit=lambda: PlaySoundUtil.stop_sound_async(),
            breadcrumb="Voice",
        )

    @staticmethod
    def make_resolved_voice_label(state: State) -> str:
        if state.project.get_tts_model_type().value.requires_voice and not ProjectVoiceUtil.has_voice(state.project):
            currently = make_currently_string("required", value_prefix="", color_code=COL_ERROR)
        elif not ProjectVoiceUtil.has_voice(state.project):
            currently = make_currently_string("none", color_code=COL_ERROR)
        else:
            currently = make_currently_string(ProjectVoiceUtil.get_voice_label(state.project))
        return f"{LABEL_ADD_VOICE_SAMPLE} {currently}"

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

        secondary = tts_type.id == "indextts2_local" and is_secondary
        # Transcripts are always collected for primary samples (sidecar text
        # or STT), so the shared list is complete for any model; whether a
        # model actually sends the transcript at generation time remains a
        # model-specific capability.
        requires_transcript = not secondary
        if requires_transcript:
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

        if len(sound.data) / sound.sr < VOICE_SAMPLE_MIN_DURATION_S:
            ask.ask_error("Sound file must be at least 2 seconds")
            return

        sound = SoundPipeline.apply_voice_clone_post_processing(sound)

        if sound.data.size == 0:
            ask.ask_error("Selected sound sample is entirely silence")
            return

        if len(sound.data) / sound.sr < VOICE_SAMPLE_MIN_DURATION_S:
            ask.ask_error("Sound file post silence trim must be at least 2 seconds")
            return

        duration_s = len(sound.data) / sound.sr
        printt(f"{COL_DIM}Playing selected sound sample ({duration_s:.1f}s)...")
        printt()
        PlaySoundUtil.play_sound_async(sound)

        force_enter_prompt = False

        transcript = ""
        if not secondary:

            # Retain sidecar text for every primary sample, so switching to a
            # transcript-requiring model does not lose available reference text.
            transcript_path = Path(path).with_suffix(".txt")
            if transcript_path.exists():
                transcript = text_util.load_text_file(str(transcript_path), errors="replace").strip()
                if transcript:
                    printt(f"Loaded transcript text from")
                    printt(f"{transcript_path}:")
                    printt(f"{COL_DIM_ITALICS}{transcript}")
                    printt()

            if not transcript:
                # [2] No sidecar text: transcribe every primary sample.
                transcript, err = VoiceMenuShared.transcribe_voice_sample_to_text(state, sound)
                if err:
                    ask.ask_error(err)
                    return

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
    def transcribe_voice_sample_to_text(state: State, sound: "Sound") -> tuple[str, str]:
        """Shared STT step for voice sample flows.

        Returns (transcript, error); exactly one is non-empty. A successful
        transcript can still be empty when every word was filtered out.
        """
        printt(f"Transcribing... {COL_DIM}(language code: {state.project.language_code or 'none'})")
        printt()

        if state.prefs.stt_variant == SttVariant.DISABLED:
            stt_variant = SttVariant.LARGE_V3
        else:
            stt_variant = state.prefs.stt_variant
        words = Transcriber.transcribe_to_words(
            sound,
            state.project.language_code,
            stt_variant,
            state.prefs.stt_config,
            state,
        )
        if isinstance(words, str):
            return "", words
        transcript = Transcriber.get_flat_text_filtered_by_probability(words, VOICE_CLONE_TRANSCRIBE_MIN_PROBABILITY)
        print(f"Transcribed text {COL_DIM}(low probability words filtered out){COL_DEFAULT}:")
        print_feedback(transcript, long_pause=True)
        return transcript, ""

    @staticmethod
    def _voice_check_scope(state: State) -> tuple[str, bool] | None:
        """Decide what the voice pre-flight must check for the active model.

        Returns None when the model (or checkpoint) uses no voice samples and
        nothing is to be checked. Otherwise ``(blocking_message, requires_transcript)``:
        a non-empty message means generation cannot proceed at all (a required
        voice is missing); else the project's voice references are to be
        checked, which is trivially fine when there are none.
        """
        project = state.project
        model_type = project.get_tts_model_type()
        if REGISTRY.voice_binding(model_type.id) is None:
            return None
        if model_type.id == "pocket_local" and project.get_model_setting(model_type.id, "predefined_voice"):
            return None
        # Qwen3 needs a clone sample only for its "base" checkpoint type;
        # custom_voice uses a speaker id and voice_design uses instructions,
        # so those model types launch without any voice samples. An unset
        # model_type defaults to "base".
        if model_type.id == "qwen3tts_local" and project.get_model_setting(model_type.id, "model_type") in (
            "custom_voice", "voice_design",
        ):
            return None
        if not project.voice_references:
            # Readiness no longer blocks a required voice; this pre-flight is
            # the single interactive place that does.
            if model_type.value.requires_voice:
                message = (
                    "A voice clone sample or predefined voice is required"
                    if model_type.id == "pocket_local"
                    else "A voice clone sample is required"
                )
                return message, False
            return "", False
        return "", REGISTRY.transcript_binding(model_type.id) is not None

    @staticmethod
    def _check_voice_sample(
            state: State,
            index: int,
            requires_transcript: bool,
            *,
            auto_transcribe: bool,
    ) -> str:
        """Return an active-sample problem, optionally repairing its missing transcript."""
        project = state.project
        entry = project.voice_references[index]
        file_name = entry["file_name"]
        cropped = ProjectVoiceUtil.get_crop_range(entry) is not None
        if cropped:
            # Validate the active crop, never fall back to the original sample.
            file_path = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry)
            label = f"Trimmed voice file for {file_name}"
        else:
            file_path = ProjectVoiceUtil.resolve_voice_file_path(project, file_name)
            label = f"Voice file {file_name}"
        if not os.path.exists(file_path):
            problem = f"{label} not found"
            if cropped:
                problem += f" (reset the trim or restore {os.path.basename(file_path)})"
            return problem
        sound_result = SoundFileUtil.load(file_path)
        if isinstance(sound_result, str):
            return f"{label} is invalid"
        transcript_key = "crop_transcript" if cropped else "transcript"
        if not requires_transcript or entry.get(transcript_key, "").strip():
            return ""
        if not auto_transcribe:
            return f"{label} has no transcript"

        transcript, err = VoiceMenuShared.transcribe_voice_sample_to_text(state, sound_result)
        if err or not transcript:
            return f"{label} could not be transcribed"
        if cropped:
            save_err = ProjectVoiceUtil.set_voice_crop_transcript_at_index_and_save(project, index, transcript)
        else:
            save_err = ProjectVoiceUtil.set_voice_transcript_at_index_and_save(project, index, transcript)
        if save_err:
            transcript_label = "trim transcript" if cropped else "transcript"
            return f"Could not save {transcript_label} for voice file {file_name}: {save_err}"
        return ""

    @staticmethod
    def get_voice_problems(state: State) -> list[str]:
        """Pure (no console output, no transcription) voice pre-flight.

        Returns every problem that would stop a TTS run: a missing required
        voice, missing or invalid sample files, and, for models that use
        transcripts, a sample whose transcript is still empty. For use by
        full-screen UI that cannot run the interactive ``validate_voices``.
        """
        project = state.project
        scope = VoiceMenuShared._voice_check_scope(state)
        if scope is None:
            return []
        blocking_message, requires_transcript = scope
        if blocking_message:
            return [blocking_message]

        problems: list[str] = []
        for index in range(len(project.voice_references)):
            problem = VoiceMenuShared._check_voice_sample(
                state, index, requires_transcript, auto_transcribe=False,
            )
            if problem:
                problems.append(problem)
        return problems

    @staticmethod
    def validate_voices(state: State) -> bool:
        """Pre-flight check before full-screen TTS features (main process only).

        Verifies every shared voice sample file and, when the active model
        uses transcripts, auto-transcribes entries whose transcript is empty
        (migrated projects may have none). Prints the collected problems,
        then returns False when the feature must not launch.
        """
        project = state.project
        scope = VoiceMenuShared._voice_check_scope(state)
        if scope is None:
            return True
        blocking_message, requires_transcript = scope
        if blocking_message:
            ask.ask_error(blocking_message)
            return False
        if not project.voice_references:
            return True

        errors: list[str] = []
        for index in range(len(project.voice_references)):
            problem = VoiceMenuShared._check_voice_sample(
                state, index, requires_transcript, auto_transcribe=True,
            )
            if problem:
                errors.append(problem)

        if errors:
            for error in errors:
                printt(f"{COL_ERROR}{error}{COL_DEFAULT}")
            printt()
            ask.ask_error("Replace problem voice clone file")
            return False
        return True

    @staticmethod
    def make_voice_sample_items(
            state: State,
            tts_type: TtsModelType,
            no_samples_label: StringOrMaker | None = None,
            on_before_set_callback: Callable | None = None,
            on_set_callback: Callable | None = None,
            on_clear_callback: Callable | None = None,
    ) -> list[MenuItem]:
        """Make direct sample-management and selection controls."""
        def make_add_label(s: State) -> str:
            if ProjectVoiceUtil.get_voice_values(s.project, tts_type):
                return LABEL_ADD_VOICE_SAMPLE
            if no_samples_label:
                return get_string_from(s, no_samples_label)
            return VoiceMenuShared.make_resolved_voice_label(s)

        def add_voice(s: State, _: MenuItem) -> None:
            if on_before_set_callback:
                on_before_set_callback()
            VoiceMenuShared.ask_and_set_voice_file(s, tts_type, append=True)
            if on_set_callback:
                on_set_callback()

        def remove_voice(s: State, _: MenuItem) -> None:
            had_samples = bool(ProjectVoiceUtil.get_voice_values(s.project, tts_type))
            is_empty = VoiceMenuShared.remove_voice_sample_from_menu(s, tts_type)
            if had_samples and is_empty and on_clear_callback:
                on_clear_callback()
            # The helper's True means empty, not that Voice clone should exit.

        voices = ProjectVoiceUtil.get_voice_values(state.project, tts_type)
        items: list[MenuItem] = []
        if len(voices) < 9:
            items.append(MenuItem(make_add_label, add_voice))
        items.append(MenuItem(LABEL_REMOVE_VOICE_SAMPLE, remove_voice))
        if len(voices) > 1:
            items.append(MenuItem(LABEL_MOVE_VOICE_SAMPLE, lambda s, _: VoiceMenuShared.move_voice_sample_from_menu(s)))
        items.extend([
            MenuItem(LABEL_CROP_VOICE_SAMPLE, lambda s, _: VoiceMenuShared.crop_voice_sample_from_menu(s)),
            MenuItem(LABEL_EDIT_VOICE_TRANSCRIPTION, lambda s, _: VoiceMenuShared.edit_voice_sample_transcript(s)),
        ])
        selection_items = []
        if len(voices) > 1:
            selection_items.append(VoiceMenuShared.make_voice_sample_selection_mode_item())
        selection_items.append(VoiceMenuShared.make_assign_voice_samples_to_text_lines_item(tts_type))
        selection_items[0].blank_line_before = True
        items.extend(selection_items)
        return items

    @staticmethod
    def make_voice_sample_selection_mode_item() -> MenuItem:
        def make_label(state: State) -> str:
            return make_menu_label(
                LABEL_VOICE_SELECTION_MODE,
                state.project.voice_select_mode.current_label,
            )

        return MenuItem(
            make_label,
            lambda state, _: VoiceMenuShared.voice_sample_selection_mode_submenu(state),
        )

    @staticmethod
    def make_assign_voice_samples_to_text_lines_item(tts_type: TtsModelType) -> MenuItem:
        def make_label(state: State) -> str:
            label = LABEL_EDIT_VOICE_SELECTIONS
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
        max_duration = tts_type.value.ui.get("voice_sample_max_duration_s")
        lines = []
        has_long_sample = False
        for i, voice in enumerate(voices, start=1):
            label = ProjectVoiceUtil.make_voice_sample_display_label(project, voice, tts_type.value)
            ref = project.voice_references[i - 1] if i - 1 < len(project.voice_references) else None
            crop_range = ProjectVoiceUtil.get_crop_range(ref) if ref is not None else None
            # The shown duration and the max-duration warning follow the sample
            # generation actually uses: the saved crop when one is in effect.
            path = ProjectVoiceUtil.effective_voice_file_path(project, ref or {"file_name": voice})
            duration = AudioMetaUtil.get_audio_duration(str(path))
            if duration is not None:
                duration_text = duration_string(duration, include_tenth=True)
                if crop_range is not None:
                    label += f"{COL_DIM} (trimmed, {duration_text}){COL_DEFAULT}"
                else:
                    label += f"{COL_DIM} ({duration_text}){COL_DEFAULT}"
                if max_duration is not None and duration > max_duration:
                    label += f"{COL_ERROR}*{COL_DEFAULT}"
                    has_long_sample = True
            elif crop_range is not None:
                label += f"{COL_DIM} (trimmed){COL_DEFAULT}"
            lines.append(f"{COL_DIM}- Voice sample {i}: {COL_DEFAULT}{label}")
        if has_long_sample:
            lines.append(
                f"  {COL_ERROR}* {COL_DEFAULT}{COL_DIM_ITALICS}Sample exceeds recommended duration for the current model "
                f"({max_duration:g}s){COL_DEFAULT}"
            )
        return "\n".join(lines)

    @staticmethod
    def ask_voice_sample_position(prompt: str, count: int) -> int | None:
        """Read a one-based position, returning a zero-based index or cancellation."""
        printt(prompt)
        value = ask.ask_input()
        if not value:
            return None
        try:
            index = int(value) - 1
        except ValueError:
            print_feedback("Bad value", is_error=True)
            return None
        if not 0 <= index < count:
            print_feedback("Out of range", is_error=True)
            return None
        return index

    @staticmethod
    def move_voice_sample_from_menu(state: State) -> None:
        count = len(state.project.voice_references)
        if not count:
            print_feedback("No voice samples")
            return
        index = VoiceMenuShared.ask_voice_sample_position("Enter voice sample number to move:", count)
        if index is None:
            return
        new_index = VoiceMenuShared.ask_voice_sample_position("Enter new position:", count)
        if new_index is None:
            return
        if index == new_index:
            print_feedback("Voice sample position unchanged")
            return
        error = ProjectVoiceUtil.move_voice_at_index_and_save(state.project, index, new_index)
        if error:
            ask.ask_error(error)
        else:
            print_feedback(f"Moved voice sample {index + 1} to position {new_index + 1}")

    @staticmethod
    def crop_voice_sample_from_menu(state: State) -> None:
        """Set or revert the active-range crop of one voice sample.

        Runs the full-screen interactive crop editor (Textual). On save, the
        cropped span is auto-transcribed; on failure the empty crop
        transcript remains and pre-flight validation (validate_voices)
        retries at the next generation launch.
        """
        from tts_audiobook_tool.textual.voice_crop_app import run_voice_crop_app

        entries = state.project.voice_references
        if not entries:
            print_feedback("No voice samples")
            return
        index = 0
        if len(entries) > 1:
            index = VoiceMenuShared.ask_voice_sample_position("Enter voice sample number to trim:", len(entries))
            if index is None:
                return

        entry = entries[index]
        crop_range = ProjectVoiceUtil.get_crop_range(entry)
        file_name = entry["file_name"]
        path = ProjectVoiceUtil.resolve_voice_file_path(state.project, file_name)
        sound = SoundFileUtil.load(path)
        if isinstance(sound, str):
            ask.ask_error(f"Couldn't load voice sample {file_name}: {sound}")
            return
        # Short samples can still be played; the editor clamps them to the
        # full range, so their cut points cannot be adjusted.

        title = Path(file_name).stem
        result = run_voice_crop_app(sound, title, crop_range)
        if isinstance(result, str):
            ask.ask_error(result)
            return

        if result.cleared:
            if crop_range is None:
                print_feedback("No trim to reset")
                return
            err = ProjectVoiceUtil.discard_voice_crop_and_save(state.project, index)
            if err:
                ask.ask_error(err)
            else:
                print_feedback(f"Trim reset; using original sample {file_name}")
            return
        if not result.saved:
            return

        err = ProjectVoiceUtil.apply_voice_crop_and_save(
            state.project, index, result.start_s, result.end_s, ""
        )
        if err:
            ask.ask_error(err)
            return
        print_feedback(f"Trim saved: {result.start_s:g}s - {result.end_s:g}s (used for generation)")

        # Auto-generate the cropped span's transcript now. On failure the
        # empty crop transcript remains; pre-flight validation retries at the
        # next generation launch.
        cropped_path = ProjectVoiceUtil.resolve_cropped_voice_file_path(state.project, state.project.voice_references[index])
        cropped_sound = SoundFileUtil.load(cropped_path)
        if isinstance(cropped_sound, str):
            print_feedback("Couldn't load trimmed sample to transcribe; will retry at next generation")
            return
        transcript, transcribe_err = VoiceMenuShared.transcribe_voice_sample_to_text(state, cropped_sound)
        if transcribe_err or not transcript:
            print_feedback("Couldn't transcribe trimmed span; will retry at next generation")
            return
        save_err = ProjectVoiceUtil.set_voice_crop_transcript_at_index_and_save(state.project, index, transcript)
        if save_err:
            ask.ask_error(save_err)

    @staticmethod
    def edit_voice_sample_transcript(state: State) -> None:
        entries = state.project.voice_references
        if not entries:
            print_feedback("No voice samples")
            return
        index = 0
        if len(entries) > 1:
            index = VoiceMenuShared.ask_voice_sample_position("Enter voice sample number to edit transcription:", len(entries))
            if index is None:
                return
        # Edit the transcript generation actually consumes: the cropped
        # span's transcript when a crop is active (and its file present),
        # otherwise the original sample's transcript. Decided through the
        # same funnel generation uses, so file and transcript stay in
        # lockstep.
        model_type = state.project.get_tts_model_type()
        voices = ProjectVoiceUtil.get_voice_values(state.project, model_type)
        if not voices:
            print_feedback("No voice samples")
            return
        effective_name, current = ProjectVoiceUtil.effective_voice_reference(state.project, model_type, index)
        is_cropped = index < len(voices) and effective_name != voices[index]
        scope_note = " (trimmed span)" if is_cropped else ""
        message = f'Enter voice sample transcript{scope_note} (or "/clear" to clear): '
        # ask_input (not raw AskAdvanced.ask): the app-standard helper adds
        # the input color/reset, clears stale buffered input, strips escape
        # sequences, and catches terminals prompt_toolkit cannot drive.
        transcript = ask.ask_input(message=message, prefill=current, lower=False)
        if not transcript or transcript == current:
            return
        if transcript == "/clear":
            transcript = ""
        if is_cropped:
            err = ProjectVoiceUtil.set_voice_crop_transcript_at_index_and_save(state.project, index, transcript)
        else:
            err = ProjectVoiceUtil.set_voice_transcript_at_index_and_save(state.project, index, transcript)
        if err:
            ask.ask_error(err)
        else:
            print_feedback("Transcript saved")

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
                    print_feedback("Bad value", is_error=True)
                    return False
                if index < 0 or index >= len(voices):
                    print_feedback("Out of range", is_error=True)
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
