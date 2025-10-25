import os

from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.whisper_util import WhisperUtil

class VoiceSubmenuShared:

    MENU_HEADING = "Voice clone and model settings"

    @staticmethod
    def make_select_voice_label(state: State) -> str:
        voice_label = state.project.get_voice_label()
        color_code = COL_ERROR if voice_label == "none" else COL_ACCENT
        currently = make_currently_string(voice_label, color_code=color_code)
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
        if not tts_type in [
            TtsModelInfos.CHATTERBOX,
            TtsModelInfos.FISH,
            TtsModelInfos.HIGGS,
            TtsModelInfos.VIBEVOICE,
            TtsModelInfos.INDEXTTS2
        ]:
            raise ValueError(f"Unsupported tts type {tts_type}")

        path = VoiceSubmenuShared.ask_voice_file(state.project.dir_path, tts_type, message_override)
        if not path:
            return

        # Load sound
        result = SoundFileUtil.load(path)
        if isinstance(result, str):
            err = result
            ask_error(err)
            return
        sound = result

        needs_transcript = tts_type in [TtsModelInfos.FISH, TtsModelInfos.HIGGS]
        if needs_transcript:

            printt("Transcribing...")
            printt()

            if state.prefs.stt_variant == SttVariant.DISABLED:
                stt_variant = SttVariant.LARGE_V3
            else:
                stt_variant = state.prefs.stt_variant

            result = WhisperUtil.transcribe_to_segments(sound, stt_variant)

            if state.prefs.stt_variant == SttVariant.DISABLED:
                Stt.clear_stt_model()

            if isinstance(result, str):
                err = result
                ask_error(err)
                return

            words = WhisperUtil.get_words_from_segments(result)
            transcript = WhisperUtil.get_flat_text_filtered_by_probability(words, VOICE_TRANSCRIBE_MIN_PROBABILITY)

        else:
            transcript = ""

        file_stem = Path(path).stem
        err = state.project.set_voice_and_save(sound, file_stem, transcript, tts_type, is_secondary=is_secondary)
        if err:
            ask_error(err)
            return

        printt_set("Voice file saved")

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

        path = ask_file_path(
             console_message=console_message,
             requestor_title=requestor_title,
             filetypes=[("WAV", "*.wav"), ("FLAC", "*.flac"), ("MP3", "*.mp3")], # TODO: on windows file requestor, this results in file type dropdown, may be more harm than good
             initialdir=default_dir_path
        )
        if not path:
            return ""

        if not os.path.exists(path):
            ask_error(f"File not found: {path}")
            return ""

        err = SoundFileUtil.is_valid_sound_file(path)
        if err:
            ask_error(err)
            return ""

        return path

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

        value = ask(prompt.strip() + " ")
        if not value:
            return
        try:
            # fyi, always cast to float bc "int(5.1)"" throws exception in 3.11 seems like
            value = float(value)
        except Exception as e:
            printt_set("Bad value", color_code=COL_ERROR)
            return
        if is_int:
            value = int(value)
        if not (lb <= value <= ub):
            printt_set("Out of range", color_code=COL_ERROR)
            return

        setattr(project, project_attr_name, value)
        project.save()
        printt_set(f"{success_prefix.strip()} {value}")

    @staticmethod
    def make_parameter_value_string(
        value: float | int | bool,
        default_value: float | int | bool,
        num_decimals: int=0
    ) -> str:

        DEFAULT_LABEL = " (default)"

        if isinstance(value, bool):
            s = str(value)
            if value == default_value:
                s += DEFAULT_LABEL
            return s

        if value == -1:
            value = default_value
        if num_decimals == 0:
            s = str(int(value))
        else:
            s = f"{value:.{num_decimals}f}"
        if value == default_value:
            s += DEFAULT_LABEL
        return s
