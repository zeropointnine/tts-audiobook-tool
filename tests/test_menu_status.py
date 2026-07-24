from tts_audiobook_tool.menus.menu_status import MenuStatus
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_DIM, COL_MEDIUM


def make_state() -> State:
    state = object.__new__(State)
    state._prefs = Prefs()
    state._project = Project(dir_path="")
    return state


def preserve_tts_state():
    return {
        "had_tts_type": hasattr(Tts, "_type"),
        "tts_type": getattr(Tts, "_type", None),
    }


def restore_tts_state(saved) -> None:
    if saved["had_tts_type"]:
        Tts._type = saved["tts_type"]
    else:
        delattr(Tts, "_type")


def test_menu_status_print_block_supports_voice_display_info(capsys):
    saved = preserve_tts_state()
    try:
        Tts._type = TtsModelType.OMNIVOICE
        state = make_state()
        state.project.omnivoice_voice_file_name = [
            "zzz belle 24a 19s_omnivoice.flac",
            "zzz belle 24b 20s_omnivoice.flac",
        ]

        MenuStatus.print_block(state)

        output = capsys.readouterr().out
        assert f"{COL_DIM}Voice clone: {COL_MEDIUM}zzz belle 24a 19s, +1 more" in output
    finally:
        restore_tts_state(saved)
