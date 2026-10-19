from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared

class VoiceQwen3Menu:

    @staticmethod
    def menu(state: State) -> None:
        def make_items(_: State) -> list[MenuItem]:
            # Each voice's Speaker item (CustomVoice checkpoints) is shared
            # voice-submenu code; this supplies its choices.
            return VoiceMenuShared.make_voice_sample_items(
                state,
                TtsModelType.require_by_id("qwen3tts_local"),
                preset_choices=get_speaker_choices,
            )

        VoiceMenuShared.menu_wrapper(state, make_items)


def get_speaker_choices(state: State) -> list[str] | None:
    """The checkpoint's speaker ids, or None (with an error shown).

    Inspects the model only when asked: opening sample management must not
    load a checkpoint just to discover its speaker inventory.
    """
    inspection, error = ModelWorker.inspect_tts_blocking(state)
    if error or inspection is None:
        ask.ask_error(error or "Couldn't inspect Qwen3-TTS model")
        return None
    metadata = inspection.metadata or {}
    model_type = str(metadata.get("model_type", state.project.get_model_setting('qwen3tts_local', 'model_type')))
    if model_type != "custom_voice":
        ask.ask_error("Speaker selection requires a CustomVoice checkpoint")
        return None
    value = metadata.get("supported_speakers")
    return [str(speaker) for speaker in value] if isinstance(value, (list, tuple)) else []
