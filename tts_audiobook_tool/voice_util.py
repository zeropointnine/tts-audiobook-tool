import os
from pathlib import Path
from outetts.version.interface import InterfaceHF

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.state import State
from .util import *
from .constants import *

class VoiceUtil:

    @staticmethod
    def ask_voice_and_set(state: State) -> None:
        """
        Gets/creates voice json, re-saves it to the project dir, and updates State.
        Or prints error message.

        Deletes any extant audio files in project dir.
        """
        voice = None

        word = "Set" if not state.voice else "Replace"
        printt(f"{COL_ACCENT}{word} voice:\n")
        printt(f"[{COL_ACCENT}1{Ansi.RESET}] Create voice file using reference WAV file (15s or less)")
        printt(f"[{COL_ACCENT}2{Ansi.RESET}] Use pre-existing voice json file")
        printt(f"[{COL_ACCENT}3{Ansi.RESET}] Use oute-tts default voice")
        printt()
        inp = ask()
        printt()

        # Get voice
        match inp:
            case "1":
                result = VoiceUtil.ask_generate_voice(state.interface)
            case "2":
                result = VoiceUtil.ask_load_voice(state.interface)
            case "3":
                result = VoiceUtil.load_voice(state.interface, DEFAULT_VOICE_FILE_PATH)
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
        err = AppUtil.delete_project_audio_files(state.project_dir)
        if err:
            printt(err, "error")

    @staticmethod
    def save_to_project_dir_and_set_state(voice: dict, state: State):  # TODO: should probably live in State, as as "setter"
        """Prints error on fail"""
        dest_path = os.path.join(state.project_dir, PROJECT_VOICE_FILE_NAME)
        assert isinstance(voice, dict)
        err = AppUtil.save_json(voice, dest_path)
        if err:
            printt(err, "error")
            return
        state.voice = voice

    @staticmethod
    def ask_load_voice(interface: InterfaceHF) -> dict | str:
        """ Returns voice dict or error string or empty string for cancel """
        path = ask("Enter file path of voice json file:\n")
        if not path:
            return ""
        return VoiceUtil.load_voice(interface, path)

    @staticmethod
    def ask_generate_voice(interface: InterfaceHF) -> dict | str:
        """ Returns voice dict or error string or empty string for cancel """
        source_path = ask("Enter file path of source audio (up to 15s) for voice clone:\n")
        if not source_path:
            return ""
        if not os.path.exists(source_path):
            return f"File not found: {source_path}"

        printt("Please wait...\n")

        result = VoiceUtil.generate_voice(interface, source_path)
        printt()

        return result

    @staticmethod
    def generate_voice(interface: InterfaceHF, path: str) -> dict | str:
        """ Returns voice dict or error string """
        try:
            voice = interface.create_speaker(path)
            VoiceUtil._add_special_properties(voice, path)
            return voice
        except Exception as e:
            return f"Error creating voice: {e}"

    @staticmethod
    def load_voice(interface: InterfaceHF, path: str) -> dict | str:
        """ Returns voice dict or error string """
        if not os.path.exists(path):
            return f"Doesn't exist: {path}"
        try:
            voice = interface.load_speaker(path)
            VoiceUtil._add_special_properties(voice, path)
            return voice
        except Exception as e:
            return f"Error loading voice file: {e}"

    @staticmethod
    def _add_special_properties(voice: dict, path: str) -> None:
        # Hash value is cached in the dict itself (perversely; easier this way)
        if not "hash" in voice:
            voice["hash"] = HashFileUtil.get_voice_hash(voice)
        # Identifier
        if not "identifier" in voice:
            s = AppUtil.sanitize_for_filename(Path(path).stem[:20])
            voice["identifier"] = s
