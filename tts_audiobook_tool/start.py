"""
Application entry-point

- Sets global tts type based on current environment introspection
- Does other dependency checks and blocks or shows warning if necessary
- Shows other one-time warnings as needed
- Launches either `App` or `Server`

Imports must be staged carefully due to dependency checks, etc.
"""

# --------------------------------------------------------------------------------------------------
# Must be imported as early as possible or else HF_HUB_CACHE can result in returning a relative path
# due to unknown import side-effect (possibly from a specific model library?, not sure)
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
from huggingface_hub import constants # type: ignore
# --------------------------------------------------------------------------------------------------

import sys
from tts_audiobook_tool.util import *
from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_types import Hint
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

# This pulls in some dependencies we would ideally first like to test for the existence of,
# but can't be helped
from tts_audiobook_tool.tts import Tts


class Start:

    def __init__(self) -> None:

        import argparse
        _parser = argparse.ArgumentParser()
        _parser.add_argument("--server", action="store_true")
        _parser.add_argument("--host", type=str, default="127.0.0.1")
        _parser.add_argument("--port", type=int, default=5001)
        _args = _parser.parse_args()

        self.is_server: bool = _args.server
        self.server_host: str = _args.host
        self.server_port: int = _args.port

    def start(self) -> None:
        """ App entrypoint """
        
        print()
        self.init_tts_or_exit(self.is_server)
        self.exit_on_wrong_torch_flavor_windows()
        if not self.is_server:
            self.exit_on_missing_ffmpeg_exe()
        self.exit_on_missing_ffmpeg_libs()
        self.exit_on_chatterbox_python_version()
        self.exit_on_missing_new_packages()
        
        self.show_startup_hints()
        self.init_logging()
        self.start_app_or_server()

    # ---

    def init_tts_or_exit(self, is_server: bool) -> None:
        """ Inits TTS else prompt to continue anyway """

        tts_model_type, num_matches = Tts.init_local_model_type()

        if tts_model_type != TtsModelType.NONE:
            return

        if num_matches > 1: # Rly shouldn't happen
            error = "\nMore than one of the supported TTS models' core libraries is currently installed.\n"
            error += "This is not recommended. Please re-install your virtual environment, \n"
            error += "following the instructions in the project's README."
            printt(COL_ERROR + error)
            exit(1)

    def exit_on_wrong_torch_flavor_windows(self) -> None:
        if not _is_cpu_only_torch_on_windows_nvidia_system():
            return
        if Tts.get_type().value.is_sgl_omni:
            return

        printt(f"{COL_ERROR}An NVIDIA GPU was detected, but the installed PyTorch build does not include CUDA support.")
        printt(f"{COL_DEFAULT}Reinstall torch using the CUDA wheel appropriate for your model requirements.")
        printt()

        prompt = f"Or press {make_hotkey_string('Y')} to run anyway without CUDA support: "
        from tts_audiobook_tool import ask
        hotkey = ask.ask_hotkey(prompt)
        if hotkey != "y":
            exit(1)

    def exit_on_missing_ffmpeg_exe(self) -> None:        
        from tts_audiobook_tool.sound.ffmpeg_util import FfmpegUtil
        if not FfmpegUtil.is_ffmpeg_available():
            printt(f"{COL_ERROR}The command 'ffmpeg' must exist on the system path.")
            printt(f"{COL_ERROR}Please install it first:")
            printt("https://ffmpeg.org/download.html")
            exit(1)

    def exit_on_missing_ffmpeg_libs(self) -> None:
        if Tts.get_type().value.requires_ffmpeg_libs:
            from tts_audiobook_tool.sound.ffmpeg_util import FfmpegUtil
            if not FfmpegUtil.are_ffmpeg_libraries_available():
                FfmpegUtil.attempt_add_ffmpeg_dll_windows()

            if not FfmpegUtil.are_ffmpeg_libraries_available():
                printt(COL_ERROR + FfmpegUtil.SHARED_LIBS_MISSING_MESSAGE)
                exit(1)

    def exit_on_chatterbox_python_version(self) -> None:
        # Chatterbox special case, Python v3.11, legacy guard
        if Tts.get_type() == TtsModelType.CHATTERBOX:
            if sys.version_info.major > 3 or (sys.version_info.major == 3 and sys.version_info.minor > 11):
                hints.show_hint(HINT_CHATTERBOX_PYTHON_DOWNGRADE)
                exit(1)

    def exit_on_missing_new_packages(self) -> None:

        new_packages = self.get_new_packages()

        from importlib import util

        missing_packages = [package for package in new_packages if not util.find_spec(package)]
        if not missing_packages:
            return
        
        hint = Hint(
            "none",
            "The app's dependencies have changed",
            f"The following packages were not found: {COL_ERROR}{', '.join(missing_packages)}{COL_DEFAULT}\n"
            "You may have updated the app from the repository without updating its dependencies.\n"
            "Update your virtual environment by re-running:\n"
            f"`{Ansi.BOLD}pip install -r {Tts.get_type().value.requirements_file_name}{Ansi.RESET}`."
        )
        hints.print_hint(hint)
        exit(1)

    def get_new_packages(self) -> list[str]:
        """
        Packages that have been added to the requirements files since first release.
        """
        
        new_packages = [
            "audiotsm", "psutil", "num2words", "chardet", "metaphone", "whisper_normalizer", 
            "pydantic", "requests", "text_to_num", "ebooklib", "bs4", "httpx"
        ]

        # win32 + linux
        if sys.platform in ("win32", "linux"):
            new_packages.append("sidon")

        # apple silicon vs not
        is_apple_silicon = ("darwin" and platform.machine() == "arm64")
        if is_apple_silicon:
            new_packages.append("mlx_whisper")
        else:
            new_packages.append("faster_whisper")

        # win32
        if sys.platform == "win32":
            new_packages.append("win32api") # ie, pywin32

        # chatterbox 
        if Tts.get_type() == TtsModelType.CHATTERBOX:
            new_packages.append("chatterbox.tts_turbo")

        # vibevoice
        if Tts.get_type() in [TtsModelType.VIBEVOICE]:
            new_packages.append("peft")

        return new_packages

    def show_startup_hints(self) -> None:
        """ Shows various one-time startup messages (which are not blockers) """

        # TODO: compare hash of current requirements file with saved hash, and if different, message user
        #   and reconcile this addition with 'hard requirement' messaging below etc

        from tts_audiobook_tool.prefs import Prefs
        temp_prefs = Prefs.load(save_if_dirty=False)

        # Tkinter (must do concrete import to test for tkinter functionality)
        if not self.is_server and not does_import_test_pass("tkinter"): 
            hints.show_hint_if_necessary(temp_prefs, HINT_TKINTER, and_prompt=True)

        # Long paths on Windows
        if not self.is_server and not is_long_path_enabled():
            hints.show_hint_if_necessary(temp_prefs, HINT_LONG_PATHS, and_prompt=True)

        # Oute
        if Tts.get_type() == TtsModelType.OUTE:
            hints.show_hint_if_necessary(temp_prefs, HINT_OUTE_CONFIG, and_prompt=True)

        # Updated UI
        # 
        # If user prefs does not have relatively new attribute "llm_url" and they're on the "old" menu system,
        # force the setting on and show FYI.
        b = not "llm_url" in temp_prefs.source_dict_keys and not temp_prefs.menu_clears_screen
        if b:
            temp_prefs.menu_clears_screen = True
            temp_prefs.save()
            hints.show_hint(HINT_UPDATED_UI, and_prompt=True)

    def init_logging(self) -> None:
        app_support.init_logging()
        printt()
        if DEV:
            printt(f"{Ansi.CLEAR_SCREEN_AND_SCROLLBACK}### DEV ###")

    def start_app_or_server(self) -> None:
        # Start
        printt()
        if self.is_server:
            from tts_audiobook_tool.server.server import Server
            Server().run(host=self.server_host, port=self.server_port)
        else:
            from tts_audiobook_tool.app import App
            _ = App()


def main() -> None:
    Start().start()

# ---

def _is_cpu_only_torch_on_windows_nvidia_system() -> bool:
    """ Infers whether vanilla Torch is installed on CUDA-capable Windows system """
    if sys.platform != "win32":
        return False

    try:
        import torch
    except Exception:
        return False

    return (
        not torch.cuda.is_available()
        and torch.version.cuda is None
        and _has_nvidia_gpu_windows()
    )

def _has_nvidia_gpu_windows() -> bool:
    """ Infers whether Nvidia CUDA exists on Windows """

    if sys.platform != "win32":
        return False

    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return False

    return result.returncode == 0 and bool(result.stdout.strip())
