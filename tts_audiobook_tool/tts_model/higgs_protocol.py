from typing import Protocol

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_model import TtsModel


class HiggsProtocol(Protocol):

    # from higgs project README example code
    DEFAULT_TEMPERATURE = 0.3

class HiggsModelProtocol(TtsModel, HiggsProtocol):
    ...


