import json
import os
from pathlib import Path

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class VoiceUtil:

    @staticmethod
    def voice_submenu(state: State) -> None:
        """
        Gets/creates voice json, re-saves it to the project dir, and updates State.
        Or prints error message.

        Deletes any extant audio files in project dir.
        """

        if Shared.is_oute():

            word = "Set" if not state.project.has_voice else "Replace"
            print_heading(f"{word} voice:")
            printt(f"{make_hotkey_string('1')} Create voice file using reference WAV file (15s or less)")
            printt(f"{make_hotkey_string('2')} Use pre-existing voice json file")
            if Shared.is_oute():
                printt(f"{make_hotkey_string('3')} Use oute-tts default voice")
            printt()
            inp = ask()

            match inp:
                case "1":
                    VoiceUtil.ask_create_oute_voice(state)
                case "2":
                    VoiceUtil.ask_load_oute_voice(state)
                case "3":
                    result = VoiceUtil.load_voice_json(DEFAULT_VOICE_JSON_FILE_PATH)
                    if isinstance(result, dict):
                        result["identifier"] = "default" # special case
                case _:
                    return

        elif Shared.is_chatterbox():

            path = ask("Enter file path of source audio for voice clone:\n")
            if not path:
                return
            if not os.path.exists(path):
                ask_continue(f"File not found: {path}")
                return
            from mutagen._file import File
            try:
                File(path)
            except Exception as e:
                printt("Not a valid audio file: {e}", "error")
                return
            err = state.project.set_chatterbox_voice_and_save(path)
            if err:
                printt(err, "error")
            elif MENU_CLEARS_SCREEN:
                    ask_continue("Saved.")


    @staticmethod
    def ask_create_oute_voice(state: State) -> None:

        path = ask("Enter file path of source audio (up to 15s) for voice clone:\n")
        if not path:
            return
        if not os.path.exists(path):
            ask_continue(f"File not found: {path}")
            return

        interface = Shared.get_oute()
        try:
            voice_json = interface.create_speaker(path)
            printt()
        except Exception as e:
            printt(f"Error creating voice: {e}", "error")
            return

        state.project.set_oute_voice_and_save(voice_json, Path(path).stem)

        # Outte created a whisper instance, which will stick around in VRAM
        # (in addition to app's own whisper instance)
        # if we don't explicitly flush stuffs
        AppUtil.gc_ram_vram()

    @staticmethod
    def ask_load_oute_voice(state: State):

        path = ask("Enter file path of voice json file:\n")
        if not path:
            return

        result = VoiceUtil.load_voice_json(path)
        if isinstance(result, str):
            printt(result)
            ask_confirm()
            return

        state.project.set_oute_voice_and_save(result, Path(path).stem)

    @staticmethod
    def load_voice_json(path: str) -> dict | str:
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

        return voice
