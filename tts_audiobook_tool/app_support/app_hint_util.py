import os
from pathlib import Path
import platform

from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_support.hints import show_hint_if_necessary
from tts_audiobook_tool.app_types import Hint, SttVariant
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

"""
Hints related to 'high level app flow'
"""

def show_pre_inference_hints(prefs: Prefs, project: Project) -> bool:
    """ 
    Shows one-time hints/warnings related to doing inference 
    """
    from tts_audiobook_tool.tts import Tts

    can_continue = True

    # TTS-model hint/warning
    if Tts.get_type() == TtsModelType.FISH_S1 and project.fish_s1_compile_enabled:
        hints.show_hint_if_necessary(prefs, HINT_FISH_S1_FIRST_COMPILE)
    elif Tts.get_type() == TtsModelType.FISH_S2 and project.fish_s2_compile_enabled:
        hints.show_hint_if_necessary(prefs, HINT_FISH_S2_FIRST_COMPILE)
    elif Tts.get_type() == TtsModelType.MOSS:
        target = project.moss_target or MossConfigs.get_default().value.repo_id
        hint = Hint.make_using(HINT_MOSS_REMOTE_CODE, target)
        can_continue = show_hint_if_necessary(prefs, hint, and_confirm=True)

    # CUDNN compatibility hint/warning
    import torch
    if platform.system() == "Linux" and torch.cuda.is_available():
        if prefs.stt_variant != SttVariant.DISABLED and prefs.stt_config.device == "cuda":
            version = torch.backends.cudnn.version()
            if version and version > CTRANSLATE_REQUIRED_CUDNN_VERSION:
                hints.show_hint(HINT_LINUX_CUDNN_VERSION, and_prompt=True)

    return can_continue


def show_player_hint(prefs: Prefs) -> None:

    from tts_audiobook_tool.util import get_package_dir

    s = "You can open audio files with the interactive player/reader here:\n"
    package_dir = get_package_dir()
    if package_dir:
        browser_path = str(Path(package_dir).parent / "browser_player" / "index.html")
    else:
        browser_path = "browser_player" + os.path.sep + "index.html"
    s += browser_path + "\n"
    s += "or on the web here:" + "\n"
    s += PLAYER_URL

    hint = Hint(key="player", heading="Reminder", text=s)
    show_hint_if_necessary(prefs, hint)
