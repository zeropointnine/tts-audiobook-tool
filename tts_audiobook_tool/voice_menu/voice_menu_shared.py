import os
from typing import Callable

from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuItemListOrMaker, MenuUtil, StringOrMaker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.target_util import TargetUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.whisper_util import WhisperUtil

class VoiceMenuShared:

    @staticmethod
    def menu(state: State) -> None:
        """
        Simply delegates to the correct model-specific voice menu
        """
        match Tts.get_type():
            case TtsModelInfos.OUTE:
                from tts_audiobook_tool.voice_menu import VoiceOuteMenu
                VoiceOuteMenu.menu(state)
            case TtsModelInfos.CHATTERBOX:
                from tts_audiobook_tool.voice_menu import VoiceChatterboxMenu
                VoiceChatterboxMenu.menu(state)
            case TtsModelInfos.FISH_S1:
                from tts_audiobook_tool.voice_menu import VoiceFishS1Menu
                VoiceFishS1Menu.menu(state)
            case TtsModelInfos.FISH_S2:
                from tts_audiobook_tool.voice_menu import VoiceFishS2Menu
                VoiceFishS2Menu.menu(state)
            case TtsModelInfos.HIGGS:
                from tts_audiobook_tool.voice_menu import VoiceHiggsMenu
                VoiceHiggsMenu.menu(state)
            case TtsModelInfos.VIBEVOICE:
                from tts_audiobook_tool.voice_menu import VoiceVibeVoiceMenu
                VoiceVibeVoiceMenu.menu(state)
            case TtsModelInfos.INDEXTTS2:
                from tts_audiobook_tool.voice_menu import VoiceIndexTts2Menu
                VoiceIndexTts2Menu.menu(state)
            case TtsModelInfos.GLM:
                from tts_audiobook_tool.voice_menu import VoiceGlmMenu
                VoiceGlmMenu.menu(state)
            case TtsModelInfos.MIRA:
                from tts_audiobook_tool.voice_menu import VoiceMiraMenu
                VoiceMiraMenu.menu(state)
            case TtsModelInfos.QWEN3TTS:
                _ = Tts.get_instance() # Pre-emptively instantiate model (special case for qwen)
                from tts_audiobook_tool.voice_menu.voice_qwen3_menu import VoiceQwen3Menu
                VoiceQwen3Menu.menu(state)
            case _:
                ...

    @staticmethod
    def menu_wrapper(
            state: State, 
            items: MenuItemListOrMaker,
            subheading: StringOrMaker | None = None,
    ) -> None:
        """
        Simple wrapper with standardized heading and exit callback
        """
        MenuUtil.menu(
            state=state,
            heading="Voice clone and model settings",
            items=items,
            subheading=subheading,
            on_exit=lambda: SoundFileUtil.stop_sound_async()
        )

    @staticmethod
    def make_resolved_voice_label(state: State) -> str:
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
        if not tts_type.value.voice_file_name_attr:
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
        sound_result = SoundFileUtil.load(path)
        if isinstance(sound_result, str):
            err = sound_result
            AskUtil.ask_error(err)
            return
        sound = sound_result

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
                sound_result = WhisperUtil.transcribe_to_words(
                    sound, state.project.language_code, stt_variant, state.prefs.stt_config
                )
                if state.prefs.stt_variant == SttVariant.DISABLED:
                    Stt.clear_stt_model()

                if isinstance(sound_result, str):
                    err = sound_result
                    AskUtil.ask_error(err)
                    return

                words = sound_result
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
             dialog_title=requestor_title,
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
    def make_temperature_item(
            state: State,
            attr: str,
            default_value: float,
            min_value: float,
            max_value: float
    ) -> MenuItem:

        return MenuUtil.make_number_item(
            state=state,
            attr=attr,
            base_label="Temperature", 
            default_value=default_value,
            is_minus_one_default=True,
            num_decimals=2,
            prompt=f"Enter temperature {COL_DIM}({min_value} to {max_value}){COL_DEFAULT}:",
            min_value=min_value,
            max_value=max_value
        )

    @staticmethod
    def make_top_k_item(
            state: State,
            attr: str,
            default_value: int,
            min_value: int=TOP_K_MIN_DEFAULT,
            max_value: int=TOP_K_MAX_DEFAULT
    ) -> MenuItem:

        return MenuUtil.make_number_item(
            state=state,
            attr=attr,
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
            attr: str,
            default_value: float
    ) -> MenuItem:

        min_value = TOP_P_MIN_DEFAULT
        max_value = TOP_P_MAX_DEFAULT

        return MenuUtil.make_number_item(
            state=state,
            attr=attr,
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
            attr: str,
            default_value: float,
            min_value = REPETITION_PENALTY_MIN_DEFAULT,
            max_value = REPETITION_PENALTY_MAX_DEFAULT
    ) -> MenuItem:

        return MenuUtil.make_number_item(
            state=state,
            attr=attr,
            base_label="Repetition penalty", 
            default_value=default_value,
            is_minus_one_default=True,
            num_decimals=2,
            prompt=f"Enter repetition penalty {COL_DIM}({min_value} to {max_value}){COL_DEFAULT}:",
            min_value=min_value,
            max_value=max_value
        )

    @staticmethod
    def make_seed_item(state: State, attr: str) -> MenuItem:
        """ Makes "self-contained" menu item for seed setting, including handler """

        def on_item(_: State, __: MenuItem) -> None:
            AskUtil.ask_number(
                saveable=state.project,
                attr=attr,
                prompt=f"Enter a static seed value {COL_DIM}(or -1 for random){COL_DEFAULT}: ",
                lb=-1, 
                ub=2**32-1,
                success_prefix="Seed set:",
                is_int=True
            )

        seed_value: int | None = getattr(state.project, attr, None)
        if seed_value is None:
            raise ValueError(f"Attribute doesn't exist: {attr}")
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
