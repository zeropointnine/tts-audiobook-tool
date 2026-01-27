import os
from typing import Callable

from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuItemListOrMaker, MenuUtil, StringOrMaker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.target_util import TargetUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.whisper_util import WhisperUtil

class VoiceMenuShared:

    @staticmethod
    def show_voice_menu(
            state: State, 
            items: MenuItemListOrMaker,
            subheading: StringOrMaker | None = None,
    ) -> None:
        MenuUtil.menu(
            state=state,
            heading="Voice clone and model settings",
            items=items,
            subheading=subheading,
            on_exit=lambda: SoundFileUtil.stop_sound_async()
        )

    @staticmethod
    def make_voice_label(state: State) -> str:
        if Tts.get_type().value.requires_voice and not state.project.has_voice:
            currently = make_currently_string("required", value_prefix="", color_code=COL_ERROR)
        elif not state.project.has_voice:
            currently = make_currently_string("none", color_code=COL_ERROR)
        else:
            currently = make_currently_string(state.project.get_voice_label())
        return f"Select voice clone sample {currently}"

    @staticmethod
    def ask_and_set_voice_file(
            state: State,
            tts_type: TtsModelInfos,
            is_secondary: bool=False,
            message_override: str=""
    ) -> None:
        """
        Asks for voice sound file path.
        Transcribes text if necessary.
        Saves to project.
        Prints feedback on success or fail.
        """

        # Rem, we do not save raw voice sound file for Oute
        if not tts_type.value.uses_voice_sound_file:
            raise ValueError(f"Unsupported tts type for this operation {tts_type}")

        if tts_type.value.requires_voice_transcript:
            Hint.show_hint_if_necessary(state.prefs, HINT_VOICE_TRANSCRIPT)

        if state.prefs.last_voice_dir and not os.path.exists(state.prefs.last_voice_dir):
            state.prefs.last_voice_dir = ""
        path = VoiceMenuShared.ask_voice_file(state.prefs.last_voice_dir, tts_type, message_override)
        if not path:
            return
        
        if not os.path.exists(path) or not os.path.isfile(path):
            AskUtil.ask_error(f"File doesn't exist: {path}")
            return
        
        state.prefs.last_voice_dir = str(Path(path).parent)

        # Load sound
        result = SoundFileUtil.load(path)
        if isinstance(result, str):
            err = result
            AskUtil.ask_error(err)
            return
        sound = result

        printt(f"{COL_DIM}Playing selected sound sample...")
        printt()
        SoundFileUtil.play_sound_async(sound)

        transcript = ""
        if tts_type.value.requires_voice_transcript:

            # [1] Get transcript from 'parallel text file' if possible
            transcript_path = Path(path).with_suffix(".txt")
            if transcript_path.exists():
                transcript = load_text_file(str(transcript_path), errors="replace").strip()
                if transcript:
                    printt(f"Loaded transcript text from")
                    printt(f"{transcript_path}:")
                    printt(f"{COL_DIM}{Ansi.ITALICS}{transcript}")
                    printt()

            if not transcript:
                # [2] Transcribe sound file using STT
                printt(f"Transcribing... {COL_DIM}(language code: {state.project.language_code or 'none'})")
                printt()

                if state.prefs.stt_variant == SttVariant.DISABLED:
                    stt_variant = SttVariant.LARGE_V3
                else:
                    stt_variant = state.prefs.stt_variant
                result = WhisperUtil.transcribe_to_words(
                    sound, state.project.language_code, stt_variant, state.prefs.stt_config
                )
                if state.prefs.stt_variant == SttVariant.DISABLED:
                    Stt.clear_stt_model()

                if isinstance(result, str):
                    err = result
                    AskUtil.ask_error(err)
                    return

                words = result
                transcript = WhisperUtil.get_flat_text_filtered_by_probability(words, VOICE_TRANSCRIBE_MIN_PROBABILITY)
                print(f"Transcribed text {COL_DIM}(low probability words filtered out){COL_DEFAULT}:")
                printt(f"{COL_DIM}{Ansi.ITALICS}{transcript}")
                printt()

        file_stem = Path(path).stem
        err = state.project.set_voice_and_save(sound, file_stem, transcript, tts_type, is_secondary=is_secondary)
        if err:
            AskUtil.ask_error(err)
            return

        print_feedback("Voice file saved")

        Hint.show_hint_if_necessary(state.prefs, HINT_TEST_REAL_TIME)

    @staticmethod
    def ask_voice_file(default_dir_path: str, tts_type: TtsModelInfos, message_override: str="") -> str:
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

        path = AskUtil.ask_file_path(
             console_message=console_message,
             requestor_title=requestor_title,
             filetypes=FILE_REQUESTOR_SOUND_TYPES,
             initialdir=default_dir_path
        )
        if not path:
            return ""

        if not os.path.exists(path):
            AskUtil.ask_error(f"File not found: {path}")
            return ""

        err = SoundFileUtil.is_valid_sound_file(path)
        if err:
            AskUtil.ask_error(err)
            return ""

        return path

    @staticmethod
    def make_clear_voice_item(state: State, info_item: TtsModelInfos, callback: Callable | None=None) -> MenuItem: # type: ignore

        def on_clear_voice(_: State, item: MenuItem) -> None:
            info_item: TtsModelInfos = item.data
            state.project.clear_voice_and_save(info_item, is_secondary=False)
            if callback:
                callback()
            print_feedback("Cleared")

        return MenuItem("Clear voice clone sample", on_clear_voice, data=info_item)

    @staticmethod
    def ask_number(
        project: Project,
        prompt: str,
        lb: float,
        ub: float,
        project_attr_name: str,
        success_prefix: str,
        is_int: bool=False
    ) -> None:

        if not hasattr(project, project_attr_name):
            raise ValueError(f"No such attribute {project_attr_name}")

        if is_int:
            lb = int(lb)
            ub = int(ub)

        value = AskUtil.ask(prompt.strip() + " ")
        if not value:
            return
        try:
            # fyi, always cast to float bc "int(5.1)"" throws exception in 3.11 seems like
            value = float(value)
        except Exception as e:
            print_feedback("Bad value", is_error=True)
            return
        if is_int:
            value = int(value)
        if not (lb <= value <= ub):
            print_feedback("Out of range", is_error=True)
            return

        setattr(project, project_attr_name, value)
        project.save()
        print_feedback(success_prefix, str(value))

    @staticmethod
    def ask_string_and_save(
        project: Project,
        prompt: str,
        project_attr_name: str,
        success_prefix: str,
        validator: Callable[[str], str] | None = None 
    ) -> None:
        """
        Helper to ask for a string value and save it to the project.
        :param validator: Takes in the user input string and returns error string if invalid (optional)
        """
        if not hasattr(project, project_attr_name):
            raise ValueError(f"No such attribute {project_attr_name}")

        if prompt:
            prompt = prompt.strip() + " "
        value = AskUtil.ask(prompt)
        if not value:
            return
        
        if validator:
            err = validator(value)
            if err:
                print_feedback(err, is_error=True)
                return

        setattr(project, project_attr_name, value)
        project.save()
        print_feedback(success_prefix, value)

    @staticmethod    
    def ask_seed_and_save(state: State, seed_attr_name: str) -> None:
        
        Hint.show_hint_if_necessary(state.prefs, HINT_SEED)

        prompt = f"Enter a static seed value {COL_DIM}(-1 for random){COL_DEFAULT}: "
        value = AskUtil.ask(prompt)
        if not value:
            return
        
        try:
            # fyi, always cast to float bc "int(5.1)"" throws exception in 3.11 seems like
            value = float(value)
        except Exception as e:
            print_feedback("Bad value", is_error=True)
            return
        
        value = int(value)
        if value < -1 or value > 2**32 - 1:
            print_feedback("Out of range", is_error=True)
            return

        project = state.project
        setattr(project, seed_attr_name, value)
        project.save()
        print_feedback("Seed set to:", value if value > -1 else "random")

    @staticmethod
    def ask_target(
            project: Project, 
            prompt: str,
            current_target: str, 
            callback: Callable[[Project, str], None]
    ) -> None: 

        printt(prompt)
        new_target = AskUtil.ask(lower=False)
        if not new_target:
            return

        if TargetUtil.is_same_target(current_target, new_target):
            print_feedback("Already set")
            return

        _, err = TargetUtil.exist_test(new_target)
        if err:
            print_feedback(err, is_error=True)
            return
        
        callback(project, new_target)
