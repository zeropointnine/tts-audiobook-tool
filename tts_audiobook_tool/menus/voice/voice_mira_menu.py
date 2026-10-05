from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared

class VoiceMiraMenu:

    @staticmethod
    def menu(state: State) -> None:
        def on_clear_voice() -> None:
            _ = ModelWorker.clear_models_if_running_blocking()

        def make_items(_: State) -> list[MenuItem]:
            return VoiceMenuShared.make_voice_sample_items(
                state,
                TtsModelType.require_by_id("mira_local"),
                on_clear_callback=on_clear_voice,
            )

        VoiceMenuShared.menu_wrapper(state, make_items)
