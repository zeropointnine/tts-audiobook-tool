import os
from typing import Callable

from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_types import Hint, SttVariant
from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuItemListOrMaker, MenuUtil, StringOrMaker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool import target_util
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.transcriber import Transcriber

class VoiceMenuShared:

    @staticmethod
    def menu(state: State) -> None:
        """
        Simply delegates to the correct model-specific voice menu
        """
        match Tts.get_type():
            case TtsModelType.CHATTERBOX:
                from tts_audiobook_tool.menus.voice import VoiceChatterboxMenu
                VoiceChatterboxMenu.menu(state)
            case TtsModelType.FISH_S1:
                from tts_audiobook_tool.menus.voice import VoiceFishS1Menu
                VoiceFishS1Menu.menu(state)
            case TtsModelType.FISH_S2:
                from tts_audiobook_tool.menus.voice import VoiceFishS2Menu
                VoiceFishS2Menu.menu(state)
            case TtsModelType.FISH_S2_SERVER:
                from tts_audiobook_tool.menus.voice import VoiceFishS2ServerMenu
                VoiceFishS2ServerMenu.menu(state)
            case TtsModelType.GLM:
                from tts_audiobook_tool.menus.voice import VoiceGlmMenu
                VoiceGlmMenu.menu(state)
            case TtsModelType.HIGGS_V2:
                from tts_audiobook_tool.menus.voice import VoiceHiggsV2Menu
                VoiceHiggsV2Menu.menu(state)
            case TtsModelType.HIGGS_V3_SERVER:
                from tts_audiobook_tool.menus.voice import VoiceHiggsV3Menu
                VoiceHiggsV3Menu.menu(state)
            case TtsModelType.INDEXTTS2:
                from tts_audiobook_tool.menus.voice import VoiceIndexTts2Menu
                VoiceIndexTts2Menu.menu(state)
            case TtsModelType.MIRA:
                from tts_audiobook_tool.menus.voice import VoiceMiraMenu
                VoiceMiraMenu.menu(state)
            case TtsModelType.MOSS:
                from tts_audiobook_tool.menus.voice import VoiceMossMenu
                VoiceMossMenu.menu(state)
            case TtsModelType.MOSS_SERVER:
                from tts_audiobook_tool.menus.voice import VoiceMossServerMenu
                VoiceMossServerMenu.menu(state) 
            case TtsModelType.OMNIVOICE:
                from tts_audiobook_tool.menus.voice import VoiceOmniVoiceMenu
                VoiceOmniVoiceMenu.menu(state)
            case TtsModelType.OUTE:
                from tts_audiobook_tool.menus.voice import VoiceOuteMenu
                VoiceOuteMenu.menu(state)
            case TtsModelType.POCKET:
                from tts_audiobook_tool.menus.voice import VoicePocketMenu
                VoicePocketMenu.menu(state)
            case TtsModelType.QWEN3TTS:
                # Special case: Pre-emptively instantiate model 
                has_instance = bool( Tts.get_instance_if_exists() )
                if not has_instance:
                    _ = Tts.get_instance() 
                    print_feedback("Model loaded")
                from tts_audiobook_tool.menus.voice.voice_qwen3_menu import VoiceQwen3Menu
                VoiceQwen3Menu.menu(state)
            case TtsModelType.VIBEVOICE:
                from tts_audiobook_tool.menus.voice import VoiceVibeVoiceMenu
                VoiceVibeVoiceMenu.menu(state)
            case _:
                raise NotImplementedError(f"value: {Tts.get_type()}")

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
            on_exit=lambda: SoundFileUtil.stop_sound_async(),
            breadcrumb="Voice",
        )

    @staticmethod
    def make_resolved_voice_label(state: State) -> str:
        if Tts.get_type().value.requires_voice and not ProjectVoiceUtil.has_voice(state.project):
            currently = make_currently_string("required", value_prefix="", color_code=COL_ERROR)
        elif not ProjectVoiceUtil.has_voice(state.project):
            currently = make_currently_string("none", color_code=COL_ERROR)
        else:
            currently = make_currently_string(ProjectVoiceUtil.get_voice_label(state.project))
        return f"Select voice clone sample {currently}"

    @staticmethod
    def ask_and_set_voice_file(
            state: State,
            tts_type: TtsModelType,
            is_secondary: bool=False,
            message_override: str=""
    ) -> None:
        """
        Asks for voice sound file path.
        Transcribes text if necessary.
        Saves to project.
        Prints feedback on success or fail.
        """

        if not tts_type.value.voice_target_attr:
            # Rem, we do not save raw voice sound file for Oute
            raise ValueError(f"Unsupported tts type for this operation {tts_type}")

        if tts_type.value.voice_transcript_attr:
            hints.show_hint_if_necessary(state.prefs, HINT_VOICE_TRANSCRIPT)

        if state.prefs.last_voice_dir and not os.path.exists(state.prefs.last_voice_dir):
            state.prefs.last_voice_dir = ""
        path = VoiceMenuShared.ask_voice_file(state.prefs.last_voice_dir, tts_type, message_override)
        if not path:
            return
        
        if not os.path.exists(path) or not os.path.isfile(path):
            ask.ask_error(f"File doesn't exist: {path}")
            return
        
        state.prefs.last_voice_dir = str(Path(path).parent)

        # Load sound
        sound_result = SoundFileUtil.load(path)
        if isinstance(sound_result, str):
            err = sound_result
            ask.ask_error(err)
            return
        sound = sound_result

        sound = SoundPipeline.apply_voice_clone_post_processing(
            sound, tts_type.value.sample_rate,
        )

        duration_s = len(sound.data) / sound.sr
        printt(f"{COL_DIM}Playing selected sound sample ({duration_s:.1f}s)...")
        printt()
        SoundFileUtil.play_sound_async(sound)

        force_enter_prompt = False

        transcript = ""
        if tts_type.value.voice_transcript_attr:

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
                    sound, state.project.language_code, stt_variant, state.prefs.stt_config
                )
                if state.prefs.stt_variant == SttVariant.DISABLED:
                    Stt.clear_stt_model()

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
        )
        if err:
            ask.ask_error(err)
            return

        print_feedback("Voice file saved")

        hints.show_hint_if_necessary(state.prefs, HINT_TEST_REAL_TIME)

        if force_enter_prompt:
            ask.ask_enter_to_continue()

    @staticmethod
    def ask_voice_file(default_dir_path: str, tts_type: TtsModelType, message_override: str="") -> str:
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

        path = ask.ask_file_path(
             console_message=console_message,
             dialog_title=requestor_title,
             filetypes=FILE_REQUESTOR_SOUND_TYPES,
             initialdir=default_dir_path
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


    @staticmethod
    def ask_temperature(
            state: State,
            attr: str,
            prompt: str,
            min_value: float,
            max_value: float,
            default_value: float,
            hint: Hint | None = None
    ) -> None:
        if hint:
            hints.show_hint_if_necessary(state.prefs, hint)

        ask.ask_number(
            state.project,
            attr,
            prompt,
            min_value,
            max_value,
            default_value,
            "Value set:",
            is_int=False
        )

    @staticmethod
    def make_temperature_item(
            state: State,
            attr: str,
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
                attr=attr,
                prompt=prompt,
                min_value=min_value,
                max_value=max_value,
                default_value=default_value,
                hint=hint
            )

        label = MenuUtil.make_number_label(
            project=state.project,
            attr=attr,
            base_label=base_label,
            default_value=default_value,
            is_minus_one_default=True,
            num_decimals=2
        )

        return MenuItem(label, on_item)

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
    def make_seed_item(
            state: State,
            attr: str,
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
            ask.ask_number(
                saveable=state.project,
                attr=attr,
                prompt=prompt,
                min_value=-1,
                max_value=2**32-1,
                default_value=-1,
                success_prefix="Seed set:",
                is_int=True,
                print_range_info=False
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
        new_target = ask.ask(lower=False)
        if not new_target:
            return

        if target_util.is_same_target(current_target, new_target):
            print_feedback("Already set")
            return

        _, err = target_util.exist_test(new_target)
        if err:
            print_feedback(err, is_error=True)
            return
        
        callback(project, new_target)

    @staticmethod
    def make_rolling_continuation_label(value: int) -> str:
        if value > 0:
            val = f"enabled, length {value}"
        else:
            val = f"disabled {COL_DIM}default"
        return "Rolling continuation " + make_currently_string(val)


    @staticmethod
    def ask_rolling_continuation(state: State, attribute_name: str, max_value: int, qualifier_line: str="") -> None:
        """
        :param qualifier_line: Should describe any prereqs (eg, batch size 1)
        """
        subheading = ROLLING_CONTINUATION_DESC
        if qualifier_line:
            subheading += f"\n{qualifier_line}\n"
        
        MenuUtil.print_screen_heading(
            state, 
            f"Rolling continuation {COL_DIM}(experimental)",
            subheading=subheading
        )
        ask.ask_number(
            state.project, attribute_name, "Enter value", 
            min_value=0, max_value=max_value, default_value=0, 
            success_prefix="Rolling continuation num segments set to", is_int=True
        )

    @staticmethod
    def make_manual_voice_menu_items(
        state: State, 
        path_attribute: str, 
        transcript_attribute: str,
        is_required: bool=False
    ) -> tuple[MenuItem, MenuItem]:
        """
        Creates pair of MenuItems for voice path and voice transcript for "SGL-Omni mode"
        (specifically for cases where the server api for the given TTS model does NOT support handling data uri)
        """

        def make_path_label(_) -> str:
            
            prefix = "Enter voice clone sample filepath"
            value = getattr(state.project, path_attribute)

            if value:
                value = ellipsize_path_for_menu(value)
                value_prefix = "currently: "
                color = COL_ACCENT
            else:
                if is_required:
                    value = "required"
                    value_prefix = ""
                    color = COL_ERROR
                else:
                    value = "none"
                    value_prefix = "currently: "
                    color = COL_ERROR

            return make_menu_label(prefix, value, value_prefix=value_prefix, color_code=color)

        def ask_path() -> None:
            s = (
                "Enter voice clone reference audio path:\n"
                f"{COL_DIM}This must be either a file path accessible from the\n"
                f"running server environment or a URL"
            )    
            ask.ask_string_and_save(state.project, s, path_attribute, "Voice clone sample path set:")

        path_item = MenuItem(make_path_label, lambda _, __: ask_path())
        
        # ---

        def make_transcript_label(_) -> str:

            prefix = "Enter voice clone sample transcript"
            value = getattr(state.project, transcript_attribute)
            has_path = bool( getattr(state.project, path_attribute) )

            if value:
                value = truncate_pretty(value, 40, content_color=COL_ACCENT)
                value_prefix = "currently: "
                color = COL_ACCENT
            else:
                if is_required or has_path:
                    value = "required"
                    value_prefix = ""
                    color = COL_ERROR
                else:
                    value = "none"
                    value_prefix = "currently: "
                    color = COL_ERROR

            return make_menu_label(prefix, value, value_prefix=value_prefix, color_code=color)


            # prefix = "Enter voice clone sample transcript"
            # value = getattr(state.project, transcript_attribute)
            # has_path = bool( getattr(state.project, path_attribute) )
            
            # if not value and has_path:
            #     return f"{prefix} {COL_DIM}({COL_ERROR}required{COL_DIM})"

            # label_value = truncate_pretty(value, 40) if value else "none"
            # return make_menu_label(prefix, label_value)

        def ask_transcript() -> None:
            ask.ask_string_and_save(
                state.project,
                "Enter voice clone sample transcript:",
                transcript_attribute,
                "Voice clone sample transcript set:",
            )

        transcript_item = MenuItem(make_transcript_label, lambda _, __: ask_transcript())

        return path_item, transcript_item
