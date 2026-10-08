import os
from pathlib import Path
import platform

from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_support.hints import show_hint_if_necessary
from tts_audiobook_tool.app_types import Hint, SttVariant
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.system_support.gpu_caps_util import GpuCapsUtil
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs

"""
Hints related to 'high level app flow'
"""

def show_pre_inference_hints(prefs: Prefs, project: Project) -> bool:
    """
    Shows one-time hints/warnings related to doing inference.
    Returns False if inference should *not* continue.
    """
    can_continue = True

    # TTS-model hint/warning
    match project.get_tts_model_type().id:
        case "fish_s1_local":
            if project.get_model_setting('fish_s1_local', 'compile_enabled'):
                hints.show_hint_if_necessary(prefs, HINT_FISH_S1_FIRST_COMPILE, and_prompt=True)
        case "fish_s2_local":
            if project.get_model_setting('fish_s2_local', 'compile_enabled'):
                hints.show_hint_if_necessary(prefs, HINT_FISH_S2_FIRST_COMPILE, and_prompt=True)
        case "dots_local":
            if project.get_model_setting('dots_local', 'compile'):
                hints.show_hint_if_necessary(prefs, HINT_DOTS_FIRST_COMPILE, and_prompt=True)
        case "moss_local":
            target = project.get_model_setting('moss_local', 'target') or MossConfigs.get_default().value.repo_id
            target = "huggingface repo id: " + target
            hint = Hint.make_using(HINT_MOSS_REMOTE_CODE, target)
            can_continue = show_hint_if_necessary(prefs, hint, and_confirm=True)

    # cuDNN compatibility hint/warning
    if platform.system() == "Linux":
        if prefs.stt_variant != SttVariant.DISABLED and prefs.stt_config.device == "cuda":
            version = GpuCapsUtil.cudnn_version()
            if version and version > CTRANSLATE_REQUIRED_CUDNN_VERSION:
                hints.show_hint(HINT_LINUX_CUDNN_VERSION, and_prompt=True)

    return can_continue

def get_startup_version_messages(
        stored_code: int | None,
        current_code: int,
        messages: list[tuple[int, str]],
) -> list[str]:
    """
    Returns the messages with a version code greater than stored_code and
    not greater than current_code, in ascending version-code order.
    Returns an empty list if there is no stored code (no history).
    """
    if stored_code is None:
        return []
    qualifying = [
        (code, message) for code, message in messages
        if stored_code < code <= current_code
    ]
    qualifying.sort(key=lambda item: item[0])
    return [message for _, message in qualifying]

def show_startup_version_messages(prefs: Prefs) -> None:
    """
    Shows what has changed since the last time the app ran, then records the
    current startup version code in prefs.

    - No stored code (new user, or first run with this feature): stores the
      current code silently.
    - Stored code newer than current (downgrade): leaves the stored code alone,
      so that re-upgrading doesn't repeat messages.
    The code is stored only after any messages have been shown.
    """
    from tts_audiobook_tool.constants_startup_messaging import (
        STARTUP_VERSION_CODE, STARTUP_VERSION_MESSAGES
    )

    stored = prefs.startup_version_code
    if stored is not None and stored >= STARTUP_VERSION_CODE:
        return

    messages = get_startup_version_messages(stored, STARTUP_VERSION_CODE, STARTUP_VERSION_MESSAGES)
    if messages:
        hint = Hint(
            "",
            f"New features since the last time you ran {APP_NAME}:",
            "\n".join(f"- {message}" for message in messages),
        )
        hints.show_hint(hint)

    prefs.startup_version_code = STARTUP_VERSION_CODE
    prefs.save()

def show_shared_startup_hints(prefs: Prefs, is_server: bool) -> None:
    """
    Shows one-time informational startup messages which are not blockers
    that are relevant to both app mode and server mode.
    """

    from tts_audiobook_tool.tts import Tts
    from tts_audiobook_tool.util import does_import_test_pass, is_long_path_enabled

    # Tkinter (must do concrete import to test for tkinter functionality)
    if not is_server and not does_import_test_pass("tkinter"):
        hints.show_hint_if_necessary(prefs, HINT_TKINTER, and_prompt=is_server)

    # Long paths on Windows
    if not is_server and not is_long_path_enabled():
        hints.show_hint_if_necessary(prefs, HINT_LONG_PATHS, and_prompt=is_server)

    # SGL-Omni is a venv-level capability: in a local-mode venv without a
    # TTS model, saved SGL-Omni settings cannot be used here
    if (
        not Tts.is_remote_mode()
        and not Tts.get_available_tts_models()
        and prefs.remote_tts_url != ""
    ):
        hints.show_hint_if_necessary(prefs, HINT_SGL_OMNI_DORMANT, and_prompt=is_server)

def make_player_hint() -> Hint:
    from tts_audiobook_tool.util import get_package_dir

    s = "You can open audio files with the interactive player/reader here:\n"
    package_dir = get_package_dir()
    if package_dir:
        browser_path = str(Path(package_dir).parent / "browser_player" / "index.html")
    else:
        browser_path = "browser_player" + os.path.sep + "index.html"
    s += text_util.make_terminal_hyperlink(browser_path, is_file=True) + "\n"
    s += "or on the web:" + "\n"
    s += text_util.make_terminal_hyperlink(PLAYER_URL)

    return Hint(key="player", heading="Reminder", text=s)

def show_player_hint(prefs: Prefs, and_prompt: bool=False) -> None:
    hint = make_player_hint()
    show_hint_if_necessary(prefs, hint, and_prompt=and_prompt)
