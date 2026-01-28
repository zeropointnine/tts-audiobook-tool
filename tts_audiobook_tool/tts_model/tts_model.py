from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model import TtsModelInfo
from tts_audiobook_tool.util import *

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


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

    def generate_using_project(
            self, project: Project, prompts: list[str], force_random_seed: bool=False
    ) -> list[Sound] | str:
        """
        Generates Sound/s using the relevant TTS model attributes in the `project` object.
        Typically delegates-to/wraps a more "parameter-specific" concrete method.

        :param prompts:
            List of one or more prompts to process
        :param force_random_seed:
            Is ignored if implementation does not support seed

        Returns:
            A list of Sounds (one per prompt), or error string
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
