import os

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.whisper_util import WhisperUtil

class VoiceSubmenuShared:

    @staticmethod
    def ask_and_set_voice_file(
            project: Project,
            tts_type: TtsModelInfos,
            is_secondary: bool=False
    ) -> None:
        """
        Asks for voice sound file path.
        Transcribes text if necessary.
        Saves to project.
        Prints feedback on success or fail.
        """

        if not tts_type in [TtsModelInfos.CHATTERBOX, TtsModelInfos.FISH, TtsModelInfos.HIGGS,
                TtsModelInfos.VIBEVOICE, TtsModelInfos.INDEXTTS2]:
            # Rem, we do not save raw voice sound file for Oute
            raise ValueError(f"Unsupported tts type {tts_type}")

        path = VoiceSubmenuShared.ask_voice_file(project.dir_path)
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
            # Transcribe
            printt("Transcribing...")
            printt()
            result = WhisperUtil.transcribe_to_words(sound)
            if isinstance(result, str):
                err = result
                ask_error(err)
                return
            transcript = WhisperUtil.get_flat_text_filtered_by_probability(result, VOICE_TRANSCRIBE_MIN_PROBABILITY)
        else:
            transcript = ""

        file_stem = Path(path).stem
        err = project.set_voice_and_save(sound, file_stem, transcript, tts_type, is_secondary=is_secondary)
        if err:
            ask_error(err)
            return

        printt_set("Voice file saved")

    @staticmethod
    def ask_voice_file(default_dir_path: str) -> str:
        """
        Asks for voice file path.
        Validates file and shows error prompt if necessary.
        Returns path or empty string.
        """

        ui = Tts.get_type().value.ui
        console_message = ui.get("voice_path_console", "")
        requestor_title = ui.get("voice_path_requestor", "")

        path = ask_file_path(
             console_message=console_message,
             requestor_title=requestor_title,
             filetypes=[("WAV", "*.wav"), ("FLAC", "*.flac"), ("MP3", "*.mp3")], #z verify this works out
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
    def make_parameter_value_string(value: float | int | bool, default_value: float | int | bool, num_decimals: int=0) -> str:

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
