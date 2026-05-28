from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback, Strictness
from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfo
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

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
            Implements the @classmethods and related non-instance-dependent static functions
            Gets called for any non-instance-related functionality

        MyModel(MyBaseModel)
            The concrete class, with generate() function and any other required abstractmethod implementations.
            Model library imports live here, and here alone.

    """

    # Gets assigned by concrete class
    INFO: TtsModelInfo

    # Optional persistent callback for streamed audio chunks
    stream_chunk_callback: StreamChunkCallback | None = None
    # Optional persistent callback invoked when a streaming generation finishes
    stream_end_callback: StreamEndCallback | None = None

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
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool=False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
    ) -> list[Sound] | str:
        """
        Generates Sound/s using the relevant TTS model attributes in the `project` object.
        Typically delegates-to/wraps a more "parameter-specific" concrete method.

        :param prompts:
            List of one or more prompts to process
        :param force_random_seed:
            Is ignored if implementation does not support seed
        :param on_stream_chunk:
            Optional callback for streaming audio chunks in the same format as
            `Sound.data`: a 1d `np.ndarray` with dtype `float32`
        :param on_stream_end:
            Optional callback invoked once when streaming generation completes
            (ie, after the last chunk)

        Returns:
            A list of Sounds (one per prompt), or error string
            Sound.data must be of np.dtype("float32")

        """
        ...

    def clear_stream_state(self) -> None:
        """
        Clears any transient stream callback/state references left over from a
        streaming generation.

        Concrete models with additional streaming-related state (custom
        streamer objects, etc.) may override-and-super.
        """
        self.stream_chunk_callback = None
        self.stream_end_callback = None

    def clear_continuation(self) -> None:
        """
        Clears any cached continuation context used to bridge one generation
        call/segment into the next.

        Default is a no-op. Concrete models that support rolling continuation
        can override this to reset model-specific cached text/audio state at
        caller-defined boundaries, such as paragraph or section breaks.
        """
        pass

    def massage_for_inference(self, text: str) -> str:
        """
        Applies text transformations from `TtsModelInfo.substitutions` (usually single character punctuation)
        to address any model-specific idiosyncrasies, etc.

        Concrete class may want to override-and-super with extra logic, as needed.
        """

        for before, after in self.INFO.substitutions:
            text = text.replace(before, after)
        return text

    def prepare_text_for_inference(self, project: Project, text: str) -> str:
        """
        Standard pre-inference text pipeline applied to any string before
        handing it to generate(). Order matters:
          1. Project word substitutions (case/plural-aware)
          2. Generic prompt normalization (numbers->words, ellipsis cleanup,
             optional un-all-caps per model)
          3. Model-specific massage_for_inference (overridable per model)
        """
        from tts_audiobook_tool.text_ops.prompt_normalizer import PromptNormalizer
        text = PromptNormalizer.apply_prompt_word_substitutions(
            text, project.word_substitutions, project.language_code
        )
        text = PromptNormalizer.normalize_prompt(
            text=text,
            language_code=project.language_code,
            un_all_caps=self.INFO.un_all_caps,
        )
        text = self.massage_for_inference(text)
        return text

    # ---
    # Class methods - these are not instance-dependent, and in some cases are "instance-optional"

    @classmethod
    def get_blocking_issues(
            cls, project: Project, instance: TtsBaseModel | None
    ) -> list[ReadinessIssue]:
        """
        Returns list of blocking issues describing why generate is not possible.
        Some issues can only be known with a concrete instance (param `instance`).
        """

        # Default implementation is for model whose only potential requirement is voice clone-related
        item = cls._get_standard_voice_blocker(project)
        return [item] if item else []
   
    def get_warning_issues(self, project: Project) -> list[str]:
        """ Returns warning info based on the state of `project` and `self` """

        # Default implementation returns random voice warning if any
        warning = self._get_standard_random_voice_reason(project)
        return [warning] if warning else []

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
        voice_file_name = cls.get_voice_value(project)
        if not voice_file_name:
            return "none"
        
        # Remove file suffix
        voice_file_name = Path(voice_file_name).stem
        
        # Remove filename 'postfix decorator'; not great
        voice_file_name = voice_file_name.strip("_" + cls.INFO.file_tag)
        
        tag = app_text.sanitize_for_filename(voice_file_name[:30])
        return tag
    
    @classmethod
    def get_model_display_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        """ 
        Formatted text describing model including potential 'variant' info, used for main menu, plus.
        Color formatting convention is: White-Model-Text Gray-Qualification-Text, with no parens

        TODO: No longer used; revisit
        """
        return cls.INFO.ui['proper_name']

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

        voice_file_name = cls.get_voice_value(project)
        voice_file_name = voice_file_name.removesuffix(f"_{info.file_tag}.flac")
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

    @classmethod
    def get_voice_value(cls, project: Project) -> str:
        """
        Returns the active voice reference for this model and project (could be filename or other).
        Override for models that store voice across multiple fields.
        """
        return getattr(project, cls.INFO.voice_file_name_attr, "")

    @classmethod
    def get_missing_voice_file_issue(
            cls,
            project: Project,
            voice_file_name_attr: str | None = None,
    ) -> ReadinessIssue | None:
        """
        Returns a blocking issue if the given configured voice filename is non-empty
        but does not exist under the project directory.
        """
        voice_file_name_attr = voice_file_name_attr or cls.INFO.voice_file_name_attr
        if not voice_file_name_attr:
            raise Exception("Logic error - must override this method")

        voice_file_name = getattr(project, voice_file_name_attr, "")
        if not voice_file_name:
            return None

        voice_path = os.path.join(project.dir_path, voice_file_name)
        if os.path.exists(voice_path):
            return None

        return ReadinessIssue(
            "voice sample",
            f"Voice clone sample file not found: {voice_file_name}"
        )

    @classmethod
    def should_trim_trailing_token_noise(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> bool:
        """ 
        Should run "trailing token noise" detector/trimmer after gen
        """
        return False


    @classmethod
    def _get_standard_voice_blocker(cls, project: Project) -> ReadinessIssue | None:

        if not cls.INFO.voice_file_name_attr:
            raise Exception("Logic error - must override this method")

        if cls.INFO.requires_voice and not getattr(project, cls.INFO.voice_file_name_attr, ""):
            return ReadinessIssue("voice sample", "A voice clone sample is required")

        err = cls.get_missing_voice_file_issue(project)
        if err:
            return err

        return None

    def _get_standard_random_voice_reason(self, project: Project) -> str:

        if self.INFO.requires_voice:
            return ""
        if not self.INFO.voice_file_name_attr:
            return ""

        if self.get_voice_value(project):
            return ""

        # Voice is not required, and no voice file specified
        return "Model may generate random voices because no voice clone reference has been specified"

    @classmethod
    def get_strictness_warning(cls, strictness: Strictness, project: Project, instance: TtsBaseModel | None) -> str:
        """
        Reason why the given validation `strictness` is discouraged, if any
        """
        return TtsBaseModel.default_strictness_warning_reason(strictness, project)
        
    @staticmethod
    def default_strictness_warning_reason(strictness: Strictness, project: Project) -> str:
        if strictness >= Strictness.HIGH and project.language_code != "en":
            return f"Not recommended when language code != en"
        return ""
