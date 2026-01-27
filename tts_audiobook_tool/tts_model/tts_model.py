from __future__ import annotations
from abc import ABC, abstractmethod

from tts_audiobook_tool.tts_model_info import TtsModelInfo
from tts_audiobook_tool.util import *


class TtsModel(ABC):
    """
    Base class for a TTS model

    Note: generate() method 
        Must be implementation-specific b/c variable params. 
        Must return Sound with np.dtype("float32")
    """

    def __init__(self, info: TtsModelInfo):
        self.info = info

    @abstractmethod
    def kill(self) -> None:
        """
        Performs any clean-up (nulling out local member variables, etc)
        that might help with garbage collection.
        """
        ...

    def massage_for_inference(self, text: str) -> str:
        """
        Applies text transformations from `self.info.substitutions` (usually single character punctuation)
        to address any model-specific idiosyncrasies, etc.
        Concrete class may want to override-and-super with extra logic.
        """
        for before, after in self.info.substitutions:
            text = text.replace(before, after)
        return text
