from collections.abc import Iterable
import os
from typing import TYPE_CHECKING, Callable

from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_types import RoomReverb, SttVariant
from tts_audiobook_tool.app_types.voice import Voice, stereo_position_label, stereo_position_to_percent
from tts_audiobook_tool import ask

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuItemListOrMaker, MenuUtil, StringOrMaker
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
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.transcriber import Transcriber

# Menu item labels shared by per-model voice menus; tests reference these.
LABEL_VOICE_SAMPLE_ITEM = "Voice"  # "Voice N: <name> (<duration>); "<instructions>""
LABEL_VOICE_SAMPLES_SUPERLABEL = "Voice list"
LABEL_ADD_VOICE_SAMPLE = "Add"  # Follows the listed voices; adds a default (sample-less) voice
LABEL_EDIT_VOICE_SELECTIONS = "Edit voice/line selections"
LABEL_AUTO_ASSIGN_CHARACTER_VOICES = "Assign character voices using LLM"

# Per-voice submenu labels
LABEL_VOICE_NAME = "Name"
LABEL_CLEAR_VOICE_NAME = "Clear name"
LABEL_SET_VOICE_SAMPLE = "Voice sample"
LABEL_CLEAR_VOICE_SAMPLE = "Clear voice sample"
LABEL_REMOVE_VOICE_SAMPLE = "Remove"
LABEL_RESET_VOICE_SAMPLE = "Remove (reset to default)"  # The only voice
LABEL_MOVE_VOICE_SAMPLE = "Move voice sample position"
LABEL_CROP_VOICE_SAMPLE = "Trim or play"
LABEL_EDIT_VOICE_TRANSCRIPTION = "Transcription"
LABEL_SET_INSTRUCTIONS = "Instructions"
LABEL_CLEAR_INSTRUCTIONS = "Clear instructions"
SUBLABEL_INSTRUCTIONS_UNUSED_WITH_SAMPLE = "Not used while this voice has a voice sample"
# Per-voice preset (`Voice.presets`); the item label is the model's `preset_label`
DEFAULT_PRESET_LABEL = "Preset"
SUBLABEL_SAMPLE_UNUSED_WITH_PRESET = "Not used: {preset_label} takes precedence"
LABEL_EFFECTS_SUPERLABEL = "Effects"
LABEL_STEREO_POSITION = "Stereo position"
LABEL_ROOM_REVERB = "Room reverb"

# Max length of the transcript preview under the edit-transcription item
TRANSCRIPT_PREVIEW_LENGTH = 40
# UI input guard, not a model-specific instruction vocabulary/context limit
INSTRUCTIONS_MAX_LENGTH = 4000
DEFAULT_INSTRUCTIONS_MESSAGE = "Enter instructions for this voice:"


def format_instructions_message(message: str) -> str:
    """Dims the parenthesized hint lines of a catalog `instructions_message`."""
    return "\n".join(
        f"{COL_DIM}{line}{COL_DEFAULT}" if line.lstrip().startswith("(") else line
        for line in message.split("\n")
    )

# Minimum duration of an imported voice sample, checked before and after silence trim
VOICE_SAMPLE_MIN_DURATION_S = 2.0

# Supplies a model's preset choices for a voice's preset item (may inspect the
# model). None means unavailable; the provider has already told the user why.
PresetChoices = Callable[[State], "list[str] | None"]


def get_preset_label(tts_type: TtsModelType) -> str:
    """Eg "Predefined voice" (Pocket), "Speaker" (Qwen3-TTS)."""
    return tts_type.value.ui.get("preset_label", DEFAULT_PRESET_LABEL)


class VoiceMenuShared:

    @staticmethod
    def menu(state: State) -> None:
        """
        Delegates to the voice menu for the selected backend/model.

        Server models use the catalog menu. A local model gets the shared
        voice-sample menu unless it adds voice behavior of its own (extra
        items, callbacks or preset choices) in a `voice_*_menu.py` module.
        """
        model_type = state.project.get_tts_model_type()
        if (Tts.get_audio_cpp_definition(model_type) is not None
                or Tts.get_configured_definition(model_type) is not None):
            from tts_audiobook_tool.menus.voice.voice_catalog_menu import VoiceCatalogMenu
            VoiceCatalogMenu.menu(state)
            return
        # Only local models reach here; server variants are always handled by
        # the catalog menu above.
        match model_type.id:
            case "indextts2_local":
                from tts_audiobook_tool.menus.voice import VoiceIndexTts2Menu
                VoiceIndexTts2Menu.menu(state)
            case "mira_local":
                from tts_audiobook_tool.menus.voice import VoiceMiraMenu
                VoiceMiraMenu.menu(state)
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
                if model_type.value.backend_kind is not TtsBackendKind.LOCAL:
                    # eg the none placeholder: there is no model to manage voices for
                    raise NotImplementedError(f"value: {model_type}")
                VoiceMenuShared.menu_wrapper(state, VoiceMenuShared.make_local_items)

    @staticmethod
    def make_local_items(state: State) -> list[MenuItem]:
        """The shared voice-sample items, for a local model without its own voice menu."""
        return VoiceMenuShared.make_voice_sample_items(state, state.project.get_tts_model_type())

    @staticmethod
    def make_remote_items(state: State) -> list[MenuItem]:
        """Resolve the current selection on every redraw, even across backends.

        Menu reconciliation can change the selection while this menu is open.
        Only the controls for this render retain a definition, never the menu
        factory itself. An unselected/unknown model offers no stale controls.
        """
        model_type = state.project.get_tts_model_type()
        definition = Tts.get_audio_cpp_definition(model_type) or Tts.get_configured_definition(model_type)
        if definition is not None:
            from tts_audiobook_tool.menus.voice.voice_catalog_menu import VoiceCatalogMenu
            return VoiceCatalogMenu.make_items(state, definition)
        return []

    @staticmethod
    def with_voice_selection_items(items: MenuItemListOrMaker) -> Callable[[State], list[MenuItem]]:
        """Item maker for a Voices menu: `items`, then the selection controls.
        An empty menu (eg an unresolved remote model) stays empty."""
        def make(state: State) -> list[MenuItem]:
            base = items(state) if callable(items) else items
            if not base:
                return []
            return [*base, *VoiceMenuShared.make_voice_selection_items(
                state, state.project.get_tts_model_type())]
        return make

    @staticmethod
    def menu_wrapper(
            state: State,
            items: MenuItemListOrMaker,
            subheading: StringOrMaker | None = None,
    ) -> None:
        """Show sample management independently of model settings.

        `items` is the voice list plus any model-specific items (eg VibeVoice's
        LoRA); the shared selection controls are always appended last.
        """
        MenuUtil.menu(
            state=state,
            heading="Voices",
            items=VoiceMenuShared.with_voice_selection_items(items),
            subheading=subheading,
            on_exit=lambda: PlaySoundUtil.stop_sound_async(),
            breadcrumb="Voices",
        )

    @staticmethod
    def ask_and_load_voice_sound(
            state: State,
            tts_type: TtsModelType,
            message_override: str = "",
    ) -> tuple[str, "Sound"] | None:
        """
        Asks for a voice sound file path, loads and validates it (minimum
        duration before and after silence trim), applies voice clone
        post-processing, and starts playing it.
        Returns (path, sound), or None on cancel or error (after the error
        prompt). Saves nothing to the project.
        """
        if state.prefs.last_voice_dir and not os.path.exists(state.prefs.last_voice_dir):
            state.prefs.last_voice_dir = ""
            state.prefs.save()
        path = VoiceMenuShared.ask_voice_file(state.prefs.last_voice_dir, tts_type, message_override)
        if not path:
            return None

        if not os.path.exists(path) or not os.path.isfile(path):
            ask.ask_error(f"File doesn't exist: {path}")
            return None

        state.prefs.last_voice_dir = str(Path(path).parent)
        state.prefs.save()

        # Load sound
        sound_result = SoundFileUtil.load(path)
        if isinstance(sound_result, str):
            ask.ask_error(sound_result)
            return None
        sound = sound_result

        if len(sound.data) / sound.sr < VOICE_SAMPLE_MIN_DURATION_S:
            ask.ask_error("Sound file must be at least 2 seconds")
            return None

        # The result is what gets saved, so a re-import of the same source
        # under a different gap limit yields different audio, which import
        # treats as a new sample rather than sharing the existing file.
        sound = SoundPipeline.apply_voice_clone_post_processing(
            sound, max_silence_gap=state.prefs.import_max_silence_gap
        )

        if sound.data.size == 0:
            ask.ask_error("Selected sound sample is entirely silence")
            return None

        if len(sound.data) / sound.sr < VOICE_SAMPLE_MIN_DURATION_S:
            ask.ask_error("Sound file post silence trim must be at least 2 seconds")
            return None

        duration_s = len(sound.data) / sound.sr
        printt(f"{COL_DIM}Playing selected sound sample ({duration_s:.1f}s)...")
        printt()
        PlaySoundUtil.play_sound_async(sound)
        return path, sound

    @staticmethod
    def get_voice_sample_transcript(state: State, path: str, sound: "Sound") -> tuple[str, str]:
        """
        The transcript for a newly selected primary sample: its sidecar text
        file's, otherwise speech-to-text.
        Returns (transcript, error); error is non-empty only when
        speech-to-text failed.
        """
        # Retain sidecar text for every primary sample, so switching to a
        # transcript-requiring model does not lose available reference text.
        sidecar = VoiceMenuShared.load_transcript_sidecar(path)
        if sidecar:
            transcript_path, transcript = sidecar
            printt(f"Loaded transcript text from")
            printt(f"{transcript_path}:")
            printt(f"{COL_DIM_ITALICS}{transcript}")
            printt()
            return transcript, ""
        # No sidecar text: transcribe every primary sample.
        return VoiceMenuShared.transcribe_voice_sample_to_text(state, sound)

    @staticmethod
    def ask_and_set_voice_sample(state: State, tts_type: TtsModelType, index: int) -> None:
        """
        Asks for a voice sound file and sets it as the sample of the voice at
        `index`, replacing any current one. Validation happens first; then the
        sample is saved with its trim reset and its transcript cleared, and
        then the transcript is filled from the sidecar text file or
        speech-to-text. If speech-to-text fails, the sample stays with a
        missing transcript (the generation pre-flight retries).
        Prints feedback on success or fail.
        """
        if REGISTRY.voice_binding(tts_type.id) is None:
            raise ValueError(f"Unsupported tts type for this operation {tts_type}")
        if not 0 <= index < len(state.project.voice_references):
            print_feedback("No such voice")
            return

        # Transcripts are always collected for primary samples (sidecar text
        # or STT), so the shared list is complete for any model; whether a
        # model actually sends the transcript at generation time remains a
        # model-specific capability.
        loaded = VoiceMenuShared.ask_and_load_voice_sound(state, tts_type)
        if loaded is None:
            return
        path, sound = loaded

        err = ProjectVoiceUtil.set_voice_sample_at_index_and_save(
            state.project, index, sound, Path(path).stem, tts_type,
        )
        if err:
            ask.ask_error(err)
            return

        transcript, err = VoiceMenuShared.get_voice_sample_transcript(state, path, sound)
        if err:
            ask.ask_error(f"Voice file saved, but its transcript is missing: {err}")
            return
        err = ProjectVoiceUtil.set_voice_transcript_at_index_and_save(state.project, index, transcript)
        if err:
            ask.ask_error(f"Voice file saved, but couldn't save its transcript: {err}")
            return

        VoiceMenuShared._finish_voice_import(state)

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
        Saves to project (replacing the voice list, or appending to it; or,
        for IndexTTS2's secondary emotion clip, setting that).
        Prints feedback on success or fail.
        The voice menu sets per-voice samples with `ask_and_set_voice_sample`.
        """

        if REGISTRY.voice_binding(tts_type.id) is None:
            raise ValueError(f"Unsupported tts type for this operation {tts_type}")

        secondary = tts_type.id == "indextts2_local" and is_secondary

        loaded = VoiceMenuShared.ask_and_load_voice_sound(state, tts_type, message_override)
        if loaded is None:
            return
        path, sound = loaded

        transcript = ""
        if not secondary:
            transcript, err = VoiceMenuShared.get_voice_sample_transcript(state, path, sound)
            if err:
                ask.ask_error(err)
                return

        err = ProjectVoiceUtil.set_voice_and_save(
            state.project,
            sound,
            Path(path).stem,
            transcript,
            tts_type,
            is_secondary=is_secondary,
            append=append,
        )
        if err:
            ask.ask_error(err)
            return

        VoiceMenuShared._finish_voice_import(state)

    @staticmethod
    def _finish_voice_import(state: State) -> None:
        """
        Ends a successful voice sample import, the same way for every source.
        Always waits for Enter, so that the import's console output (the
        sidecar or speech-to-text transcript, plus the one-time realtime
        playback tip) stays readable before the menu redraws.
        """
        print_feedback("Voice file saved", skip_pause=True)
        hints.show_hints_if_necessary(state.prefs, [HINT_TEST_REAL_TIME])
        ask.ask_enter_to_continue()

    @staticmethod
    def load_transcript_sidecar(path: str) -> tuple[Path, str] | None:
        """
        Returns the transcript text file accompanying a voice sample and its
        stripped text, if any.
        Checks "voice.txt" first, then "voice.flac.txt"; a blank or unreadable
        candidate is skipped in favor of the next one.
        """
        audio_path = Path(path)
        candidates = [
            audio_path.with_suffix(".txt"),
            audio_path.with_name(audio_path.name + ".txt"),
        ]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            text = text_util.load_text_file(str(candidate), errors="replace").strip()
            if text:
                return candidate, text
        return None

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
    def _voice_check_scope_for_project(project: Project) -> bool | None:
        """Return the settings-aware transcript requirement, or None to skip samples.

        This describes the model/checkpoint, not the project's sample inventory.
        Missing samples are checked per selected voice, never project-wide.
        """
        model_type = project.get_tts_model_type()
        if REGISTRY.voice_binding(model_type.id) is None:
            return None
        # Samples are irrelevant when the model can't clone under the current
        # settings (eg Qwen3-TTS CustomVoice/VoiceDesign checkpoints).
        if not Tts.get_model_support(project).can_voice_clone(project):
            return None
        # Runtime requirement (settings-aware), not mere transcript storage.
        return Tts.requires_reference_transcript(project)

    @staticmethod
    def _check_voice_sample(
            state: State,
            index: int,
            requires_transcript: bool,
            *,
            auto_transcribe: bool,
    ) -> str:
        """Return a problem with the voice's (existing) sample, optionally
        repairing its missing transcript."""
        project = state.project
        voice = project.voice_references[index]
        file_name = voice.file_name
        cropped = voice.crop is not None
        if cropped:
            # Validate the active crop, never fall back to the original sample.
            file_path = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, voice)
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
        active_transcript = voice.crop.transcript if voice.crop is not None else voice.transcript
        if not requires_transcript or active_transcript.strip():
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
    def _collect_voice_problems(
            state: State,
            voice_indices: Iterable[int] | None,
            *,
            auto_transcribe: bool,
    ) -> list[str]:
        project = state.project
        model_type = project.get_tts_model_type()
        support = Tts.get_model_support(project)
        uses_presets = REGISTRY.voice_binding(model_type.id) is not None and support.can_voice_presets(project)
        requires_preset = uses_presets and support.requires_voice_preset(project)
        requires_transcript = VoiceMenuShared._voice_check_scope_for_project(project)
        if requires_transcript is None and not requires_preset:
            return []
        indices = (
            range(len(project.voice_references))
            if voice_indices is None else sorted(set(voice_indices))
        )
        problems: list[str] = []
        for index in indices:
            source = ProjectVoiceUtil.resolve_voice_source(project, project.voice_references[index])
            if source.kind == "missing":
                if requires_preset:
                    preset_label = get_preset_label(model_type).lower()
                    remedy = f"select a {preset_label}"
                    what = preset_label
                else:
                    remedy = (
                        "set a sample or select a predefined voice"
                        if model_type.id == "pocket_local"
                        else "set a sample or remove the voice"
                    )
                    what = "voice sample"
                problems.append(f"{LABEL_VOICE_SAMPLE_ITEM} {index + 1} has no {what} ({remedy})")
                continue
            # Only a voice that generates from its sample has a sample to check;
            # otherwise it uses a preset, its instructions, or the built-in voice.
            if source.kind != "sample" or requires_transcript is None:
                continue
            problem = VoiceMenuShared._check_voice_sample(
                state, index, requires_transcript, auto_transcribe=auto_transcribe,
            )
            if problem:
                problems.append(problem)
        return problems

    @staticmethod
    def get_voice_problems(state: State, *, voice_indices: Iterable[int] | None = None) -> list[str]:
        """Pure (no console output, no transcription) voice pre-flight.

        Check the job's selected voice slots for missing required samples,
        invalid active files, and missing required transcripts. None checks
        all voices; an explicit empty iterable checks none. For full-screen
        UI that cannot run the interactive ``validate_voices``.
        """
        return VoiceMenuShared._collect_voice_problems(state, voice_indices, auto_transcribe=False)

    @staticmethod
    def validate_voices(state: State, *, voice_indices: Iterable[int] | None = None) -> bool:
        """Pre-flight check before full-screen TTS features (main process only).

        Check only the job's selected voice slots; None checks all voices.
        Auto-transcribe missing required transcripts only within that scope.
        Print collected problems and wait for Enter before returning False.
        """
        errors = VoiceMenuShared._collect_voice_problems(state, voice_indices, auto_transcribe=True)
        if errors:
            for error in errors:
                printt(f"{COL_ERROR}{error}{COL_DEFAULT}")
            printt()
            ask.ask_enter_to_continue()
            return False
        return True

    @staticmethod
    def make_voice_sample_items(
            state: State,
            tts_type: TtsModelType,
            on_before_set_callback: Callable | None = None,
            on_set_callback: Callable | None = None,
            on_clear_callback: Callable | None = None,
            preset_choices: PresetChoices | None = None,
            allow_custom_preset: bool = False,
    ) -> list[MenuItem]:
        """Make the voice list (each opening its own submenu) and Add.
        The selection controls close the menu; `menu_wrapper` adds them.

        Built on every redraw, so labels and indices are those of the current
        list. A project always has at least one voice. The set callbacks run
        around a voice's "Set voice sample"; `on_clear_callback` runs when a
        removal leaves no voice with a sample. `preset_choices` enables each
        voice's preset item (while the model uses presets, `can_voice_presets`).
        `allow_custom_preset` additionally offers free-form preset names.
        """
        def open_submenu(s: State, index: int) -> None:
            VoiceMenuShared.voice_sample_submenu(
                s, tts_type, index, on_clear_callback,
                on_before_set_callback=on_before_set_callback,
                on_set_callback=on_set_callback,
                preset_choices=preset_choices,
                allow_custom_preset=allow_custom_preset,
            )

        def add_voice(s: State, _: MenuItem) -> None:
            index, err = ProjectVoiceUtil.add_voice_and_save(s.project)
            if err:
                ask.ask_error(err)
                return
            open_submenu(s, index)

        voices = ProjectVoiceUtil.get_voice_values(state.project, tts_type)
        items: list[MenuItem] = []
        for index in range(len(voices)):
            label, _ = VoiceMenuShared.make_voice_sample_item_label(state.project, tts_type, index)
            items.append(MenuItem(
                label,
                lambda s, item: open_submenu(s, item.data),
                data=index,
                superlabel=LABEL_VOICE_SAMPLES_SUPERLABEL if index == 0 else "",
                superlabel_no_blank_line=True,
            ))
        if len(voices) < 9:
            items.append(MenuItem(LABEL_ADD_VOICE_SAMPLE, add_voice))
        return items

    @staticmethod
    def make_voice_selection_items(state: State, tts_type: TtsModelType) -> list[MenuItem]:
        """The closing group of every Voices menu: Edit voice/line selections
        and Auto-assign character voices. Added by `menu_wrapper`."""
        items: list[MenuItem] = []
        items.append(VoiceMenuShared.make_assign_voice_samples_to_text_lines_item(tts_type))
        items.append(VoiceMenuShared.make_auto_assign_character_voices_item())
        items[0].blank_line_before = True
        return items

    @staticmethod
    def make_auto_assign_character_voices_item() -> MenuItem:
        def on_item(state: State, _: MenuItem) -> None:
            # Deferred: the feature's modules aren't needed until it's opened
            from tts_audiobook_tool.menus.speaker_assign.speaker_assign_menu import SpeakerAssignMenu
            SpeakerAssignMenu.menu(state)

        def make_label(state: State) -> str:
            from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
            label = LABEL_AUTO_ASSIGN_CHARACTER_VOICES
            if not SpeakerAssignFileUtil.has_any_data(state.project.dir_path):
                label += f" {COL_DIM}(optional)"
            return label

        return MenuItem(make_label, on_item)

    @staticmethod
    def make_assign_voice_samples_to_text_lines_item(tts_type: TtsModelType) -> MenuItem:
        def make_label(state: State) -> str:
            label = LABEL_EDIT_VOICE_SELECTIONS
            voices = ProjectVoiceUtil.get_voice_values(state.project, tts_type)
            if len(voices) < 2:
                label += f" {COL_DIM}(optional; requires 2+ voice samples)"
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
    def get_voice_sample_duration(project: Project, voice: Voice) -> float | None:
        """Duration of the sample generation actually uses: the saved crop when in effect."""
        path = ProjectVoiceUtil.effective_voice_file_path(project, voice)
        return AudioMetaUtil.get_audio_duration(str(path))

    @staticmethod
    def make_voice_sample_duration_text(voice: Voice, duration: float | None) -> str:
        """Eg "(12.3s)", "(trimmed, 12.3s)", "(trimmed)", or "" when unknown and untrimmed."""
        if duration is not None:
            duration_text = duration_string(duration, include_tenth=True)
            return f"(trimmed, {duration_text})" if voice.crop is not None else f"({duration_text})"
        return "(trimmed)" if voice.crop is not None else ""

    @staticmethod
    def get_voice_sample_required_label(project: Project, index: int) -> str:
        """For a voice without a sample: why one is required, else "".

        A sample is required only when the model requires a voice clone
        sample: then every voice needs one (including when another voice has
        one). Otherwise a voice without a sample uses the model's built-in
        voice. Never for a voice that has a sample or an active preset (eg a
        Pocket predefined voice), or when the model or its settings use no
        voice samples.
        """
        model_type = project.get_tts_model_type()
        if not model_type.value.requires_voice:
            return ""  # A missing required preset is labeled separately
        source = ProjectVoiceUtil.resolve_voice_source(project, project.voice_references[index])
        if source.kind != "missing" or VoiceMenuShared._voice_check_scope_for_project(project) is None:
            return ""
        if model_type.id == "pocket_local":
            # This voice has neither a sample nor a predefined voice
            return "voice sample or predefined voice required"
        return "required"

    @staticmethod
    def is_voice_transcript_missing(project: Project, tts_type: TtsModelType, index: int) -> bool:
        """Whether the voice has a sample but its effective transcript (the trim's when trimmed) is empty."""
        if not project.voice_references[index].has_sample:
            return False
        _, transcript = ProjectVoiceUtil.effective_voice_reference(project, tts_type, index)
        return not transcript.strip()

    @staticmethod
    def get_missing_transcript_color(project: Project) -> str:
        """Error-colored when the current model uses transcripts, else dim."""
        return COL_ERROR if Tts.requires_reference_transcript(project) else COL_DIM

    @staticmethod
    def make_voice_sample_item_label(project: Project, tts_type: TtsModelType, index: int) -> tuple[str, bool]:
        """
        Returns the list item label for one voice, and whether its sample exceeds the model's recommended duration.
        Eg: 'Voice 1: some_voice.flac (15.3s*); "some instructions"',
        'Voice 1: some_voice.flac (12.3s); missing transcript', 'Voice 1: "some instructions"'
        when the voice has instructions (for the current model) but no sample file, or
        'Voice 1: required' (error-colored) when the model requires a sample, or
        'Voice 1: no instructions set' (non-blocking hint) when the model expects them, or
        'Voice 1: anna' for an active preset (eg a Pocket predefined voice), which
        takes precedence over (and so replaces) any sample details.
        Sample details are omitted when the model can't clone under the current
        settings (`can_voice_clone`).
        """
        voice = project.voice_references[index]
        parts: list[str] = []
        is_long = False
        support = Tts.get_model_support(project)
        # Sample details are hidden (not cleared) when the model can't clone
        # under the current settings, eg Qwen3-TTS VoiceDesign.
        can_clone = support.can_voice_clone(project)
        required_label = VoiceMenuShared.get_voice_sample_required_label(project, index)
        # Only the preset comes from the resolved source; samples and
        # instructions keep their fuller list details below.
        source = ProjectVoiceUtil.resolve_voice_source(project, voice)
        preset = source.label if source.kind == "preset" else ""
        preset_label = get_preset_label(tts_type).lower()
        if preset:
            # The preset is the voice's value; any sample it overrides is unused
            parts.append(preset)
        elif voice.has_sample and can_clone:
            part = ProjectVoiceUtil.make_voice_sample_display_label(project, voice.file_name, tts_type.value)
            duration = VoiceMenuShared.get_voice_sample_duration(project, voice)
            duration_text = VoiceMenuShared.make_voice_sample_duration_text(voice, duration)
            max_duration = tts_type.value.ui.get("voice_sample_max_duration_s")
            is_long = duration is not None and max_duration is not None and duration > max_duration
            if is_long:
                # Mark inside the closing paren, eg "(15.3s*)"
                duration_text = f"{duration_text[:-1]}{COL_ERROR}*{COL_DIM})"
            if duration_text:
                part += f"{COL_DIM} {duration_text}{COL_DEFAULT}"
            parts.append(part)
            # Only when the current model (under its settings) requires a transcript
            if (Tts.requires_reference_transcript(project)
                    and VoiceMenuShared.is_voice_transcript_missing(project, tts_type, index)):
                parts.append(f"{COL_ERROR}missing transcript{COL_DEFAULT}")
        elif required_label:
            parts.append(f"{COL_ERROR}{required_label}{COL_DEFAULT}")
        if not preset and source.kind == "missing" and support.requires_voice_preset(project):
            parts.append(f"no {preset_label} {COL_DIM}({COL_ERROR}required{COL_DIM}){COL_DEFAULT}")
        # Same runtime gate as the submenu's Instructions item (eg CosyVoice3's Mode)
        instructions = voice.instructions.get(tts_type.value.id, "")
        can_instructions = support.can_instructions(project)
        # Eg AuK: a voice sample takes precedence, so its instructions are unused
        instructions_unused = (voice.has_sample and can_clone
                               and not support.uses_instructions_with_sample(project))
        if can_instructions and not instructions_unused:
            if instructions.strip():
                parts.append(ProjectVoiceUtil.make_instructions_preview(instructions))
            elif support.expects_voice_instructions(project):
                # Non-blocking hint (so not error-colored), eg Qwen3-TTS VoiceDesign
                # would get an arbitrary voice
                parts.append("no instructions set")
        if not parts:
            # Eg Qwen3-TTS CustomVoice without instructions: the speaker's default delivery
            parts.append(ProjectVoiceUtil.make_no_voice_value_text(project))
        title = VoiceMenuShared.make_voice_title(voice, index)
        label = f"{title}: " + f"{COL_DIM};{COL_DEFAULT} ".join(parts)
        return label, is_long

    @staticmethod
    def make_voice_title(voice: Voice | None, index: int) -> str:
        """Eg "Voice 1", or "Voice 1 MyName" when the voice has a name."""
        title = f"{LABEL_VOICE_SAMPLE_ITEM} {index + 1}"
        return f"{title} {voice.name}" if voice is not None and voice.name else title

    @staticmethod
    def make_transcript_preview(project: Project, tts_type: TtsModelType, index: int) -> str:
        """The effective transcript (the crop's when in effect), flattened, ellipsized,
        and quoted; "missing" when the voice has a sample but no transcript."""
        _, transcript = ProjectVoiceUtil.effective_voice_reference(project, tts_type, index)
        transcript = " ".join(transcript.split())
        if not transcript:
            return "missing" if project.voice_references[index].has_sample else "none"
        return f"\"{ellipsize(transcript, TRANSCRIPT_PREVIEW_LENGTH)}\""

    @staticmethod
    def make_set_voice_sample_label(project: Project, tts_type: TtsModelType, index: int) -> str:
        """Eg 'Voice sample (currently: some_voice)' or 'Voice sample (currently: none)'."""
        voice = project.voice_references[index]
        if voice.has_sample:
            value = ProjectVoiceUtil.make_voice_sample_display_label(project, voice.file_name, tts_type.value)
            return make_menu_label(LABEL_SET_VOICE_SAMPLE, value)
        color = COL_ERROR if VoiceMenuShared.get_voice_sample_required_label(project, index) else COL_ACCENT
        return make_menu_label(LABEL_SET_VOICE_SAMPLE, "none", color_code=color)

    @staticmethod
    def voice_sample_submenu(
            state: State,
            tts_type: TtsModelType,
            index: int,
            on_clear_callback: Callable | None = None,
            on_before_set_callback: Callable | None = None,
            on_set_callback: Callable | None = None,
            preset_choices: PresetChoices | None = None,
            allow_custom_preset: bool = False,
    ) -> None:
        """Options for one voice. Follows the voice when it is moved; exits when it is removed.
        `preset_choices` enables the voice's preset item (see make_voice_sample_items)."""
        current = [index]  # Updated on move

        def get_voice(s: State) -> Voice | None:
            entries = s.project.voice_references
            return entries[current[0]] if 0 <= current[0] < len(entries) else None

        def make_heading(s: State) -> str:
            return VoiceMenuShared.make_voice_title(get_voice(s), current[0])

        def on_set_sample(s: State, _: MenuItem) -> None:
            if on_before_set_callback:
                on_before_set_callback()
            VoiceMenuShared.ask_and_set_voice_sample(s, tts_type, current[0])
            if on_set_callback:
                on_set_callback()

        def on_clear_sample(s: State, _: MenuItem) -> None:
            voice = get_voice(s)
            if voice is None or not voice.has_sample:
                return
            error = ProjectVoiceUtil.clear_voice_sample_at_index_and_save(s.project, current[0])
            if error:
                ask.ask_error(error)
                return
            PlaySoundUtil.stop_sound_async()
            if on_clear_callback and not ProjectVoiceUtil.has_any_voice_sample(s.project):
                on_clear_callback()
            print_feedback("Voice sample cleared")

        def on_set_name(s: State, _: MenuItem) -> None:
            VoiceMenuShared.ask_voice_name(s, current[0])

        def on_clear_name(s: State, _: MenuItem) -> None:
            error = ProjectVoiceUtil.set_voice_name_at_index_and_save(s.project, current[0], "")
            if error:
                ask.ask_error(error)
            else:
                print_feedback("Name cleared")

        def on_set_instructions(s: State, _: MenuItem) -> None:
            VoiceMenuShared.ask_voice_instructions(s, tts_type, current[0])

        def on_clear_instructions(s: State, _: MenuItem) -> None:
            error = ProjectVoiceUtil.set_voice_instructions_at_index_and_save(
                s.project, current[0], tts_type.id, "")
            if error:
                ask.ask_error(error)
            else:
                print_feedback("Instructions cleared")

        def on_set_preset(s: State, _: MenuItem) -> None:
            if preset_choices is not None:
                VoiceMenuShared.ask_voice_preset(
                    s, tts_type, current[0], preset_choices, allow_custom_preset=allow_custom_preset,
                )

        def on_clear_preset(s: State, _: MenuItem) -> None:
            error = ProjectVoiceUtil.set_voice_preset_at_index_and_save(s.project, current[0], tts_type.id, "")
            if error:
                ask.ask_error(error)
            else:
                print_feedback(f"{get_preset_label(tts_type)} cleared")

        def on_remove(s: State, _: MenuItem) -> bool:
            voice = get_voice(s)
            had_sample = voice is not None and voice.has_sample
            removed = VoiceMenuShared.remove_voice_sample_from_menu(s, tts_type, current[0])
            if not removed:
                return False
            if on_clear_callback and had_sample and not ProjectVoiceUtil.has_any_voice_sample(s.project):
                on_clear_callback()
            return True

        def on_move(s: State, _: MenuItem) -> None:
            current[0] = VoiceMenuShared.move_voice_sample_from_menu(s, current[0])

        def make_items(s: State) -> list[MenuItem]:
            voice = get_voice(s)
            if voice is None:
                return []
            i = current[0]
            support = Tts.get_model_support(s.project)
            items: list[MenuItem] = []
            # Sample items are hidden (the sample is kept) when the model can't
            # clone under the current settings, eg Qwen3-TTS VoiceDesign.
            can_clone = support.can_voice_clone(s.project)
            uses_presets = preset_choices is not None and support.can_voice_presets(s.project)
            preset = voice.preset_for(tts_type.id) if uses_presets else ""
            preset_label = get_preset_label(tts_type)
            items.append(MenuItem(VoiceMenuShared.make_voice_name_menu_label(voice), on_set_name))
            if voice.name:
                items.append(MenuItem(LABEL_CLEAR_VOICE_NAME, on_clear_name))
            if can_clone:
                items.append(MenuItem(
                    VoiceMenuShared.make_set_voice_sample_label(s.project, tts_type, i),
                    on_set_sample,
                    sublabel=(
                        SUBLABEL_SAMPLE_UNUSED_WITH_PRESET.format(preset_label=preset_label.lower())
                        if preset and voice.has_sample else None
                    ),
                ))
            if can_clone and voice.has_sample:
                items.append(MenuItem(LABEL_CLEAR_VOICE_SAMPLE, on_clear_sample))
                duration = VoiceMenuShared.get_voice_sample_duration(s.project, voice)
                duration_text = VoiceMenuShared.make_voice_sample_duration_text(voice, duration)
                crop_label = (
                    make_menu_label(LABEL_CROP_VOICE_SAMPLE, duration_text[1:-1])
                    if duration_text else LABEL_CROP_VOICE_SAMPLE
                )
                max_duration = tts_type.value.ui.get("voice_sample_max_duration_s")
                if duration is not None and max_duration is not None and duration > max_duration:
                    crop_label += (
                        f" {COL_ERROR}*{COL_DIM_ITALICS}exceeds current model's "
                        f"recommended limit of {max_duration:g}s{COL_DEFAULT}"
                    )
                items.append(MenuItem(crop_label, lambda s2, _: VoiceMenuShared.crop_voice_sample_from_menu(s2, i)))
                transcript_preview = VoiceMenuShared.make_transcript_preview(s.project, tts_type, i)
                transcript_color = (
                    VoiceMenuShared.get_missing_transcript_color(s.project)
                    if VoiceMenuShared.is_voice_transcript_missing(s.project, tts_type, i) else COL_ACCENT
                )
                items.append(MenuItem(
                    make_menu_label(LABEL_EDIT_VOICE_TRANSCRIPTION, transcript_preview, color_code=transcript_color),
                    lambda s2, _: VoiceMenuShared.edit_voice_sample_transcript(s2, i),
                ))
            if uses_presets:
                required = support.requires_voice_preset(s.project)
                items.append(MenuItem(
                    make_menu_label(
                        preset_label, preset or ("required" if required else "none"),
                        color_code=COL_ERROR if required and not preset else COL_ACCENT,
                    ),
                    on_set_preset,
                ))
                if preset:
                    items.append(MenuItem(f"Clear {preset_label.lower()}", on_clear_preset))
            if support.can_instructions(s.project):
                instructions = voice.instructions.get(tts_type.id, "")
                preview = " ".join(instructions.split())
                value = ProjectVoiceUtil.make_instructions_preview(preview) if preview else "none"
                label_text = tts_type.value.ui.get("instructions_label", LABEL_SET_INSTRUCTIONS)
                label = f"{label_text}/{tts_type.value.ui['short_name']}"
                # Still editable (eg to prepare a voice before clearing its sample);
                # the proviso only matters once there is a value to ignore.
                unused = (bool(preview) and can_clone and voice.has_sample
                          and not support.uses_instructions_with_sample(s.project))
                items.append(MenuItem(
                    make_menu_label(label, value), on_set_instructions,
                    sublabel=SUBLABEL_INSTRUCTIONS_UNUSED_WITH_SAMPLE if unused else None,
                ))
                if instructions:
                    items.append(MenuItem(LABEL_CLEAR_INSTRUCTIONS, on_clear_instructions))
            if len(s.project.voice_references) > 1:
                items.append(MenuItem(LABEL_MOVE_VOICE_SAMPLE, on_move))
                items.append(MenuItem(LABEL_REMOVE_VOICE_SAMPLE, on_remove))
            else:
                # Removing the only voice leaves a fresh default voice.
                items.append(MenuItem(LABEL_RESET_VOICE_SAMPLE, on_remove))
            items.append(MenuItem(
                make_menu_label(
                    LABEL_STEREO_POSITION, stereo_position_label(voice.pan), stereo_position_label(0.0)
                ),
                lambda s2, _: VoiceMenuShared.ask_stereo_position(s2, i),
                superlabel=LABEL_EFFECTS_SUPERLABEL,
            ))
            items.append(MenuItem(
                make_menu_label(LABEL_ROOM_REVERB, voice.room_reverb.label, RoomReverb.get_default().label),
                lambda s2, _: VoiceMenuShared.room_reverb_submenu(s2, i),
            ))
            return items

        MenuUtil.menu(
            state=state,
            heading=make_heading,
            items=make_items,
            on_exit=lambda: PlaySoundUtil.stop_sound_async(),
            breadcrumb=lambda _: f"{LABEL_VOICE_SAMPLE_ITEM} {current[0] + 1}",
        )

    @staticmethod
    def ask_voice_preset(
            state: State, tts_type: TtsModelType, index: int, preset_choices: PresetChoices,
            allow_custom_preset: bool = False,
    ) -> None:
        """Choose this voice's preset, optionally accepting a custom speaker id.
        Without custom input, a single choice is saved directly (so readiness,
        which can't see the model's inventory, finds it set)."""
        choices = preset_choices(state)
        if choices is None:
            return
        preset_label = get_preset_label(tts_type)

        def save(value: str) -> None:
            error = ProjectVoiceUtil.set_voice_preset_at_index_and_save(state.project, index, tts_type.id, value)
            if error:
                ask.ask_error(error)

        if not choices and not allow_custom_preset:
            ask.ask_error(f"Model has no {preset_label.lower()} choices")
            return
        if len(choices) == 1 and not allow_custom_preset:
            save(choices[0])
            print_feedback(f"Model has only one {preset_label.lower()} ({choices[0]})")
            return
        current = state.project.voice_references[index].preset_for(tts_type.id)

        def on_select(value: str) -> None:
            if allow_custom_preset and not value:
                printt(f"Enter a custom {preset_label.lower()} name:")
                value = ask.ask_input(lower=False, prefill=current).strip()
                if not value:
                    return  # Blank/cancel preserves the current preset.
            save(value)

        MenuUtil.options_menu(
            state=state,
            heading_text=f"{preset_label} for {LABEL_VOICE_SAMPLE_ITEM} {index + 1}",
            labels=[*choices, f"Custom {preset_label.lower()}..."] if allow_custom_preset else choices,
            values=[*choices, ""] if allow_custom_preset else choices,
            current_value=current or None,
            default_value=None,
            on_select=on_select,
            breadcrumb=preset_label,
        )

    @staticmethod
    def make_voice_name_menu_label(voice: Voice) -> str:
        """Eg 'Name (currently: Narrator)', or 'Name (optional)' (dimmed) when unset."""
        if not voice.name:
            return make_menu_label_optional(LABEL_VOICE_NAME)
        return f"{LABEL_VOICE_NAME} {make_currently_string(voice.name)}"

    @staticmethod
    def ask_voice_name(state: State, index: int) -> None:
        """Edit this voice's user-facing name; blank/cancel leaves it unchanged.
        An invalid name (not one word, or not unique) is reported and not saved."""
        current = state.project.voice_references[index].name
        printt("Enter a name for this voice:")
        printt(f"{COL_DIM}(One word, no spaces)")
        value = ask.ask_input(lower=False, prefill=current).strip()
        if not value or value == current:
            return
        error = ProjectVoiceUtil.get_voice_name_error(state.project, index, value)
        if error:
            print_feedback(error, is_error=True)
            return
        error = ProjectVoiceUtil.set_voice_name_at_index_and_save(state.project, index, value)
        if error:
            ask.ask_error(error)
        else:
            print_feedback(f"{LABEL_VOICE_NAME} set to: {value}")

    @staticmethod
    def ask_voice_instructions(state: State, tts_type: TtsModelType, index: int) -> None:
        """Edit this voice's instructions; blank/cancel leaves them unchanged."""
        current = state.project.voice_references[index].instructions.get(tts_type.id, "")
        printt(format_instructions_message(tts_type.value.ui.get("instructions_message", DEFAULT_INSTRUCTIONS_MESSAGE)))
        value = ask.ask_input(lower=False, prefill=current).strip()
        if not value or value == current:
            return
        if len(value) > INSTRUCTIONS_MAX_LENGTH:
            ask.ask_error(f"Instructions must be at most {INSTRUCTIONS_MAX_LENGTH} characters")
            return
        # Validation only: the text is saved as entered
        error = Tts.get_model_support_for_type(tts_type).validate_instructions(value)
        if error:
            ask.ask_error(error)
            return
        error = ProjectVoiceUtil.set_voice_instructions_at_index_and_save(
            state.project, index, tts_type.id, value)
        if error:
            ask.ask_error(error)
        else:
            print_feedback("Instructions set")

    @staticmethod
    def ask_stereo_position(state: State, index: int) -> None:
        """Prompt for the voice's stereo position, -100 (full left) to 100 (full right)."""
        voice = state.project.voice_references[index]
        current = stereo_position_to_percent(voice.pan)
        printt("Enter stereo position, from -100 (full left) to 100 (full right), or 0 for center:")
        printt(f"{COL_DIM}(Subtle values like -30 or 30 work well for separating voices.)")
        value = ask.ask_input(prefill=str(current))
        if not value:
            return
        try:
            percent = int(round(float(value)))
        except (ValueError, OverflowError):
            ask.ask_error("Bad value")
            return
        if not -100 <= percent <= 100:
            ask.ask_error("Out of range")
            return
        if percent == current:
            return
        pan = percent / 100
        err = ProjectVoiceUtil.set_voice_pan_at_index_and_save(state.project, index, pan)
        if err:
            ask.ask_error(err)
        else:
            print_feedback(f"{LABEL_STEREO_POSITION} set to: {stereo_position_label(pan)}")

    @staticmethod
    def room_reverb_submenu(state: State, index: int) -> None:

        voice = state.project.voice_references[index]

        def on_select(value: RoomReverb) -> None:
            err = ProjectVoiceUtil.set_voice_room_reverb_at_index_and_save(state.project, index, value)
            if err:
                ask.ask_error(err)
            else:
                print_feedback(f"{LABEL_ROOM_REVERB} set to: {value.label}")

        MenuUtil.options_menu(
            state=state,
            heading_text=LABEL_ROOM_REVERB,
            labels=[item.label for item in RoomReverb],
            values=list(RoomReverb),
            current_value=voice.room_reverb,
            default_value=RoomReverb.get_default(),
            on_select=on_select
        )

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
    def move_voice_sample_from_menu(state: State, index: int) -> int:
        """Asks for the new position of the sample at `index`. Returns its index afterward."""
        count = len(state.project.voice_references)
        if count < 2 or not 0 <= index < count:
            return index
        new_index = VoiceMenuShared.ask_voice_sample_position(f"Enter new position {COL_DIM}(1-{count}):", count)
        if new_index is None:
            return index
        if index == new_index:
            print_feedback("Voice sample position unchanged")
            return index
        error = ProjectVoiceUtil.move_voice_at_index_and_save(state.project, index, new_index)
        if error:
            ask.ask_error(error)
            return index
        print_feedback(f"Moved voice sample {index + 1} to position {new_index + 1}")
        return new_index

    @staticmethod
    def crop_voice_sample_from_menu(state: State, index: int) -> None:
        """Set or revert the active-range crop of the voice sample at `index`.

        Runs the full-screen interactive crop editor (Textual). On save, the
        cropped span is auto-transcribed; on failure the empty crop
        transcript remains and pre-flight validation (validate_voices)
        retries at the next generation launch.
        """
        from tts_audiobook_tool.textual.voice_crop_app import run_voice_crop_app

        entries = state.project.voice_references
        if not 0 <= index < len(entries):
            print_feedback("No such voice sample")
            return

        entry = entries[index]
        if not entry.has_sample:
            print_feedback("No voice sample")
            return
        crop_range = entry.crop_range
        file_name = entry.file_name
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
    def edit_voice_sample_transcript(state: State, index: int) -> None:
        if not 0 <= index < len(state.project.voice_references):
            print_feedback("No such voice sample")
            return
        # Edit the transcript generation actually consumes: the cropped
        # span's transcript when a crop is active (and its file present),
        # otherwise the original sample's transcript. Decided through the
        # same funnel generation uses, so file and transcript stay in
        # lockstep.
        model_type = state.project.get_tts_model_type()
        voices = ProjectVoiceUtil.get_voice_values(state.project, model_type)
        if not voices or not voices[index]:
            print_feedback("No voice sample")
            return
        effective_name, current = ProjectVoiceUtil.effective_voice_reference(state.project, model_type, index)
        is_cropped = index < len(voices) and effective_name != voices[index]
        scope_note = " (trimmed span)" if is_cropped else ""
        printt(f'Enter voice sample transcript{scope_note} (or "/clear" to clear):')
        if not Tts.requires_reference_transcript(state.project):
            printt(f"{COL_DIM}(Note that the current TTS model does not utilize the transcript value)")
        # ask_input (not raw AskAdvanced.ask): the app-standard helper adds
        # the input color/reset, clears stale buffered input, strips escape
        # sequences, and catches terminals prompt_toolkit cannot drive.
        transcript = ask.ask_input(prefill=current, lower=False)
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
    def remove_voice_sample_from_menu(state: State, tts_type: TtsModelType, index: int) -> bool:
        """Removes the voice at `index` after confirmation. Returns True if it was removed.

        Removing the only voice resets it to a default voice; when it is
        already at defaults that happens without confirmation.
        """
        voices = ProjectVoiceUtil.get_voice_values(state.project, tts_type)
        if not 0 <= index < len(voices):
            return False
        label = f"{LABEL_VOICE_SAMPLE_ITEM} {index + 1}"
        if voices[index]:
            name = ProjectVoiceUtil.make_voice_sample_display_label(state.project, voices[index], tts_type.value)
            label += f" ({name})"
        is_only = len(voices) == 1
        if not (is_only and state.project.voice_references[index].is_default):
            printt(f"Reset {label} to default?" if is_only else f"Remove {label}?")
            if not ask.ask_confirm():
                return False
        ProjectVoiceUtil.remove_voice_at_index_and_save(state.project, tts_type, index)
        print_feedback(f"Reset {label} to default" if is_only else f"Removed {label}")
        return True

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
