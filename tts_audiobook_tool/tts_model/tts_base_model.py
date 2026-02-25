from __future__ import annotations
from abc import ABC, abstractmethod

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfo
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class TtsBaseModel(ABC):
    """
    Base class for a TTS model
    
    Rem, @abstractmethods are instance methods that must be implemented by concrete model classes.
    And @classmethods are static methods that can be used without instances.

    Implementations must subclass in two steps:

        MyBaseModel(TtsBaseModel) 
            Must not contain any model library imports
            Implements the @classmethods and holds any other static functionality
            Gets called for any non-instance-related functionality

        MyModel(MyBaseModel)
            The concrete class, with generate() function and any other required abstractmethod implementations.
            Model library imports live here, and here alone.

    """

    # Gets assigned by concrete class
    INFO: TtsModelInfo

    def __init_subclass__(cls, **kwargs):
        # Called whenever a new subclass is created
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "INFO"):
            raise TypeError(f"Class {cls.__name__} must define 'INFO'")
            
    @abstractmethod
    def kill(self) -> None:
        """
        Performs any clean-up (nulling out local member variables, etc)
        that might help with garbage collection.
        """
        ...

    @abstractmethod
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
            Sound.data must be of np.dtype("float32")

        """
        ...

    def massage_for_inference(self, text: str) -> str:
        """
        Applies text transformations from `TtsModelInfo.substitutions` (usually single character punctuation)
        to address any model-specific idiosyncrasies, etc.

        Concrete class may want to override-and-super with extra logic, as needed.
        """

        for before, after in self.INFO.substitutions:
            text = text.replace(before, after)
        return text

    # ---
    # Class methods (ie, not instance-dependent)

    @classmethod
    def get_prereq_errors(
            cls, project: Project, instance: TtsBaseModel | None, short_format: bool
    ) -> list[str]:
        """
        Returns warning messages as to why generate is not possible.
        Applies to both main gen and realtime gen.

        Some prereq errors can only be known with a concrete instance (param `instance`).

        :param short_format:
            When true, should return very short phrase, meant to be concatenated on a single line
            Else, should return full messages meant to be displayed on separate lines
        """

        # Default implementation is for model whose only possible requirement is voice clone-related
        
        err = cls._get_standard_voice_prereq_error(project, short_format)
        return [err] if err else []
   
    def get_prereq_warnings(self, project: Project) -> list[str]:
        """ Returns "non-blocking" warning info based on the state of `project` and `self` """

        # Default implementation returns random voice warning if any
        warning = self._get_standard_random_voice_reason(project)
        return [warning] if warning else []

    # ---

    @classmethod
    def get_voice_tag(cls, project: Project) -> str:
        """
        Gets "voice tag" used for sound segment filenames.
        
        This should effectively be a condensed version of the the value string 
        returned by get_voice_display_info()
        """
        
        # Default implementation is for model whose 'salient info' consists of voice clone only
        
        # Get voice filename
        if not cls.INFO.voice_file_name_attr:
            raise Exception("Logic error - must override this method")        
        voice_file_name = getattr(project, cls.INFO.voice_file_name_attr, "")
        if not voice_file_name: 
            return "none"
        
        # Remove file suffix
        voice_file_name = Path(voice_file_name).stem
        
        # Remove filename 'postfix decorator'; not great
        voice_file_name = voice_file_name.strip("_" + cls.INFO.file_tag)
        
        tag = TextUtil.sanitize_for_filename(voice_file_name[:30])
        return tag
    
    @classmethod
    def get_voice_display_info(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> tuple[str, str]:
        """
        Returns salient voice-related setting/s, with prefix label

        Returns:
            (1) Colorized prefix for the following value
            (2) Colorized string with short voice-related label
                (typically the voice clone reference filename)

            Ex: "current voice clone", "sample1.wav"
        """
        
        # Default implementation is for model where 'salient info' consists of voice clone only
        
        info = cls.INFO
        if not info.voice_file_name_attr:
            raise Exception("Logic error - must override this method")
        
        voice_file_name = getattr(project, info.voice_file_name_attr, "")
        voice_file_name = ellipsize_path_for_menu(voice_file_name)

        match (info.requires_voice, bool(voice_file_name)):
            case (True, False):
                prefix = COL_ERROR + "required"
                value = ""
            case (True, True):
                prefix = COL_DIM + "current voice clone"
                value = COL_ACCENT + voice_file_name
            case (False, False):
                prefix = COL_DIM + "current voice clone"
                value = COL_ERROR + "none" # ie, not required, but rly should be set
            case (False, True):
                prefix = COL_DIM + "current voice clone"
                value = COL_ACCENT + voice_file_name

        return prefix, value

    # ---

    @classmethod
    def _get_standard_voice_prereq_error(cls, project: Project, short_format: bool) -> str:

        if not cls.INFO.voice_file_name_attr:
            raise Exception("Logic error - must override this method")

        voice_file_name = getattr(project, cls.INFO.voice_file_name_attr, "")
        if cls.INFO.requires_voice and not voice_file_name:
            err = "requires voice sample" if short_format else "Voice sample required"
            return err
        else:
            return ""

    def _get_standard_random_voice_reason(self, project: Project) -> str:

        if self.INFO.requires_voice:
            return ""
        if not self.INFO.voice_file_name_attr:
            return ""

        voice_file_name = getattr(project, self.INFO.voice_file_name_attr, "")
        if voice_file_name:
            return ""        

        # Voice is not required, and no voice file specified
        return "Model may generate random voices because no voice clone reference has been specified"
