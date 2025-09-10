import os

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_info import TtsType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.whisper_util import WhisperUtil

class VoiceSubmenuShared:

    @staticmethod
    def ask_and_set_voice_file(project: Project, tts_type: TtsType) -> None:
        """
        Asks for voice sound file path.
        Transcribes text if necessary.
        Saves to project.
        Prints feedback on success or fail.
        """

        if not tts_type in [TtsType.CHATTERBOX, TtsType.FISH, TtsType.HIGGS, TtsType.VIBEVOICE]:
            # Rem, we do not save raw voice sound file for Oute
            raise ValueError("Unsupported tts type")

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

        needs_transcript = tts_type in [TtsType.FISH, TtsType.HIGGS]
        if needs_transcript: #z
            # Transcribe
            result = WhisperUtil.transcribe_to_words(sound)
            if isinstance(result, str):
                err = result
                ask_error(err)
                return
            transcript = WhisperUtil.get_flat_text_filtered_by_probability(result, VOICE_TRANSCRIBE_MIN_PROBABILITY)
        else:
            transcript = ""

        file_stem = Path(path).stem
        err = project.set_voice_and_save(sound, file_stem, transcript, tts_type)
        if err:
            ask_error(err)
            return

        printt_set("Voice file saved.")

    @staticmethod
    def ask_voice_file(default_dir_path) -> str:
        """
        Asks for voice file path.
        Shows error prompt if necessary.
        Returns path or empty string.
        """

        ui = Tts.get_type().value.ui
        valid_suffixes = ui["voice_path_suffixes"]

        path = ask_file_path(
             console_message=ui.get("voice_path_console", ""),
             requestor_title=ui.get("voice_path_requestor", ""),
             initialdir=default_dir_path
        )
        if not path:
            return ""

        if not os.path.exists(path):
            ask_error(f"File not found: {path}")
            return ""

        suffix = Path(path).suffix
        if not suffix in valid_suffixes:
            ask_error(f"Invalid suffix: {suffix}. Must be {valid_suffixes}")
            return ""

        err = SoundFileUtil.is_valid_sound_file(path)
        if err:
            ask_error(err)
            return ""

        return path