import json
import os
from pathlib import Path

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class VoiceUtil:

    @staticmethod
    def voice_submenu(state: State, num_audio_files: int) -> None:
        """
        Gets/creates voice json, re-saves it to the project dir, and updates State.
        Or prints error message.

        Deletes any extant audio files in project dir.
        """

        def confirm() -> bool:
            printt(f"Replacing voice will invalidate {num_audio_files} previously generated audio file fragments for this project.")
            return ask_confirm("Are you sure? ")

        voice = None

        word = "Set" if not state.project.voice else "Replace"
        print_heading(f"{word} voice:")
        printt(f"[{COL_ACCENT}1{Ansi.RESET}] Create voice file using reference WAV file (15s or less)")
        printt(f"[{COL_ACCENT}2{Ansi.RESET}] Use pre-existing voice json file")
        printt(f"[{COL_ACCENT}3{Ansi.RESET}] Use oute-tts default voice")
        printt()
        inp = ask()

        # Get voice
        match inp:
            case "1":
                if not confirm():
                    return
                result = VoiceUtil.ask_create_voice()
            case "2":
                if not confirm():
                    return
                result = VoiceUtil.ask_load_voice()
            case "3":
                if not confirm():
                    return
                result = VoiceUtil.load_voice(DEFAULT_VOICE_FILE_PATH)
                if isinstance(result, dict):
                    result["identifier"] = "default" # special case
            case _:
                return

        if isinstance(result, str):
            if result == "":
                return # signifies cancel
            else:
                printt(f"Error: {result}", "error")
                return

        voice = result

        # Save voice to project dir and set state
        VoiceUtil.save_to_project_dir_and_set_state(voice, state)

        # Delete any existing audio files in project dir, which are now 'invalid'
        err = AppUtil.delete_project_audio_files(state.prefs.project_dir)
        if err:
            printt(err, "error")

    @staticmethod
    def save_to_project_dir_and_set_state(voice: dict, state: State):  # TODO: should probably live in State, as as "setter"
        """Prints error on fail"""
        dest_path = os.path.join(state.prefs.project_dir, PROJECT_VOICE_FILE_NAME)
        assert isinstance(voice, dict)
        err = AppUtil.save_json(voice, dest_path)
        if err:
            printt(err, "error")
            return
        state.project.voice = voice

    @staticmethod
    def ask_load_voice() -> dict | str:
        """ Returns voice dict or error string or empty string for cancel """
        path = ask("Enter file path of voice json file:\n")
        if not path:
            return ""
        return VoiceUtil.load_voice(path)

    @staticmethod
    def ask_create_voice() -> dict | str:
        """ Returns voice dict or error string or empty string for cancel """
        source_path = ask("Enter file path of source audio (up to 15s) for voice clone:\n")
        if not source_path:
            return ""
        if not os.path.exists(source_path):
            return f"File not found: {source_path}"

        result = VoiceUtil.create_voice(source_path)
        printt()

        return result

    @staticmethod
    def create_voice(path: str) -> dict | str:
        """ Returns voice dict or error string """

        interface = Shared.get_oute_interface()

        try:
            voice = interface.create_speaker(path)
            VoiceUtil._add_special_properties(voice, path)
        except Exception as e:
            return f"Error creating voice: {e}"

        # Outte created a whisper instance, which will stick around in VRAM
        # (in addition to app's own whisper instance)
        # if we don't explicitly flush stuffs
        AppUtil.gc_ram_vram()

        return voice

    @staticmethod
    def load_voice(path: str) -> dict | str:
        """
        Returns voice dict or error string
        Note how we are intentionally not using oute interface `load_speaker()`
        """
        if not os.path.exists(path):
            return f"Doesn't exist: {path}"
        try:
            with open(path, 'r', encoding='utf-8') as file:
                voice = json.load(file)
        except Exception as e:
            return f"Error loading voice json file: {e}"
        if not isinstance(voice, dict):
            return f"Voice json object unexpected type: {type(voice)}"
        # TODO: do some extra validation here since we are avoiding using "interface.load_speaker()"

        VoiceUtil._add_special_properties(voice, path)
        return voice

    @staticmethod
    def _add_special_properties(voice: dict, path: str) -> None:
        from tts_audiobook_tool.hash_file_util import HashFileUtil
        # Hash value is cached in the dict itself (perversely; easier this way)
        if not "hash" in voice:
            voice["hash"] = HashFileUtil.get_voice_hash(voice)
        # Identifier
        if not "identifier" in voice:
            s = sanitize_for_filename(Path(path).stem[:20])
            voice["identifier"] = s
