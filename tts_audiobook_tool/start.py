"""
Application entry-point

- Sets global tts type based on current environment introspection
- Does other dependency checks and blocks or shows warning if necessary
- Shows other one-time warnings as needed
- Launches either `App` or `Server`

Imports must be staged carefully due to dependency checks, etc.
"""

# Must be imported as early as possible (see that module's docstring)
# The import is for side effects only (see hf_bootstrap.py)
import tts_audiobook_tool.hf_bootstrap  # pyright: ignore[reportUnusedImport]

import os
import sys
from typing import Callable
from tts_audiobook_tool.util import *
from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_types import Hint
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType

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
        _parser.add_argument("--project", type=str, default="")
        _args = _parser.parse_args()

        self.is_server: bool = _args.server
        self.server_host: str = _args.host
        self.server_port: int = _args.port
        self.project_path: str = _args.project

    def apply_project_override(self) -> None:
        """
        If a `--project` path was given on the command line, validates and applies it.

        Server mode: a valid path is used for this run only (prefs are not updated,
        and no confirmation prompt is shown); an invalid path prints an error and
        exits with a non-zero code.

        App mode: a valid path is persisted to prefs (so `State`, which re-loads
        prefs from disk, picks it up). On an invalid path, prints "Bad project
        path."; if prefs has a stored project_dir the user is asked whether to keep
        it (yes keeps the stored path; no exits). If there is no stored path, the
        user is prompted to press enter, and startup continues with no project.
        """
        if not self.project_path:
            return

        err_message = COL_ERROR + f"{self.project_path} does not appear to be a project directory"

        from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil

        if self.is_server:
            if ProjectLoadUtil.is_valid_project_dir(self.project_path):
                printt(err_message)
                exit(1)
            return

        from tts_audiobook_tool import ask
        from tts_audiobook_tool.prefs import Prefs

        prefs = Prefs.load()
        resolved, should_continue = resolve_project_override(
            self.project_path,
            prefs.project_dir,
            ProjectLoadUtil.is_valid_project_dir,
            lambda: printt(err_message),
            ask.ask_confirm,
            ask.ask_enter_to_continue,
        )

        if not should_continue:
            exit(1)

        if resolved != prefs.project_dir:
            prefs.project_dir = resolved
            prefs.save()

    def start(self) -> None:
        """ App entrypoint """

        print()
        if not self.is_server:
            self.exit_on_unsupported_terminal()
        self.init_tts_or_exit(self.is_server)
        self.exit_on_wrong_torch_flavor_windows()
        if not self.is_server:
            self.exit_on_missing_ffmpeg_exe()
        self.exit_on_missing_ffmpeg_libs()
        self.exit_on_chatterbox_python_version()
        self.exit_on_missing_new_packages()

        self.show_startup_hints()
        self.apply_project_override()
        self.init_logging()
        self.start_app_or_server()

    # ---

    def exit_on_unsupported_terminal(self) -> None:
        from tts_audiobook_tool.system_support.terminal import can_use_full_screen_terminal

        if can_use_full_screen_terminal():
            return

        print(
            f"{APP_NAME} requires a full-featured interactive terminal.\n"
            "The current terminal is unsupported. Exiting.",
            flush=True,
        )
        exit(1)

    def init_tts_or_exit(self, is_server: bool) -> None:
        """ Inits TTS else prompt to continue anyway.

        Also fixes the process-level backend mode (probed from the
        SGL-Omni sentinel package) before any further checks run.
        """

        tts_model_type, num_matches = Tts.init_local_model_type()

        # dots.tts win32 special treatment:
        #
        # On Windows, PyTorch 2.8's internal static CUDA launcher passes a 64-bit
        # CUDA stream handle through a 32-bit C long, causing dots.tts compile
        # warm-up to fail with "Python int too large to convert to C long".
        # Disable only that launcher so Inductor falls back to Triton's normal
        # launcher; torch.compile and dots.tts's reduce-overhead/fullgraph
        # optimizations remain enabled. This must run before importing torch or
        # dots_tts because TorchInductor reads the setting at import.
        if tts_model_type == TtsModelType.DOTS and sys.platform == "win32":
            if "torch" in sys.modules:
                raise RuntimeError(
                    "Cannot configure the dots.tts Windows CUDA compile workaround "
                    "because torch was already imported"
                )
            os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

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
        if Tts.is_sgl_mode():
            # SGL-Omni backend: local torch is not used at all
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
            f"`{Ansi.BOLD}pip install -r {Tts.get_requirements_file_name()}{Ansi.RESET}`."
        )
        hints.print_hint(hint)
        exit(1)

    def get_new_packages(self) -> list[str]:
        """
        Packages that have been added to the requirements files since first release.
        """

        new_packages = [
            "audiotsm", "psutil", "num2words", "chardet", "metaphone", "whisper_normalizer",
            "pydantic", "requests", "text_to_num", "ebooklib", "bs4", "httpx", "textual",
            "LavaSR"
        ]

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

        # SGL-Omni is a venv-level capability: in a local-mode venv without a
        # TTS model, saved SGL-Omni settings cannot be used here
        if (
            Tts.get_backend_mode() == TtsBackendKind.LOCAL
            and Tts.get_type() == TtsModelType.NONE
            and (
                temp_prefs.sgl_omni_type is not None
                or temp_prefs.sgl_omni_url != ""
            )
        ):
            hints.show_hint_if_necessary(temp_prefs, HINT_SGL_OMNI_DORMANT, and_prompt=True)

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
            Server(project_dir=self.project_path).run(host=self.server_host, port=self.server_port)
        else:
            from tts_audiobook_tool.app import App
            from tts_audiobook_tool.model_runtime import mark_interactive_main

            mark_interactive_main()
            _ = App()


def main() -> None:
    Start().start()

# ---

def resolve_project_override(
    override_path: str,
    current_prefs_project_dir: str,
    is_valid_project_dir: Callable[[str], str],
    report_invalid: Callable[[], None],
    confirm: Callable[[str], bool],
    enter_to_continue: Callable[[], None],
) -> tuple[str, bool]:
    """
    Pure resolution logic for the `--project` CLI override (app mode only;
    server mode validates directly and exits on failure).

    :param override_path: Path given via `--project`
    :param current_prefs_project_dir: The `project_dir` currently stored in prefs
    :param is_valid_project_dir: Validator that returns an error string ("" if valid)
    :param report_invalid: Called once if the override path is invalid, before any
                           further user interaction (e.g. to print an error line)
    :param confirm: Confirmation prompt used to ask whether to keep a stored path
    :param enter_to_continue: Called when the path is invalid and no stored path
                              exists, so the user is aware before continuing
                              without a project
    :return: `(project_dir_to_persist, should_continue)` — `should_continue` is
             False when the caller should abort (stored path was declined)
    """
    if not is_valid_project_dir(override_path):
        return (override_path, True)

    report_invalid()

    if not current_prefs_project_dir:
        enter_to_continue()
        return ("", True)

    if confirm(f"Do you want to load last used project path ({current_prefs_project_dir})?"):
        return (current_prefs_project_dir, True)
    return ("", False)

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
