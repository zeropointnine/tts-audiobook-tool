from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from tts_audiobook_tool.constants import SEED_MAX as APP_SEED_MAX
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_DIM, ellipsize_path_for_menu

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class DotsCompileMode(tuple[bool, str], Enum):
    """Whether dots.tts uses its compiled (optimized) inference path.

    Compiled inference is much faster but pays a one-time compile/warm-up cost
    at load and reserves more working memory. Eager inference skips compilation
    entirely: faster to load and lower memory, but slower per-request generation.
    """

    ENABLED = (
        True,
        "Compile inference kernels for faster generation; slower load and higher memory use.",
    )
    DISABLED = (
        False,
        "Skip compilation (eager mode); fastest load and lower memory, but slower generation.",
    )

    @property
    def enabled(self) -> bool:
        return self.value[0]

    @property
    def description(self) -> str:
        return self.value[1]

    @property
    def label(self) -> str:
        return self.name.title()

    @staticmethod
    def default() -> DotsCompileMode:
        return DotsCompileMode.ENABLED

    @staticmethod
    def get_by_enabled(enabled: bool) -> DotsCompileMode:
        for item in list(DotsCompileMode):
            if item.enabled == enabled:
                return item
        return DotsCompileMode.ENABLED


class DotsBaseModel(TtsBaseModel):
    INFO = TtsModelType.DOTS.value

    SOAR_REPO_ID = "dots-studio/dots.tts-soar"
    MF_REPO_ID = "dots-studio/dots.tts-mf"
    MF_2STEPS_REPO_ID = "dots-studio/dots.tts-mf-2steps"
    MF_1STEP_REPO_ID = "dots-studio/dots.tts-mf-1step"
    PRESET_REPO_IDS = [
        SOAR_REPO_ID,
        MF_REPO_ID,
        MF_2STEPS_REPO_ID,
        MF_1STEP_REPO_ID,
    ]
    DEFAULT_REPO_ID = SOAR_REPO_ID

    # Fixed maximum audio length a single dots.tts request may produce. The
    # budget covers the voice-clone reference (its transcript and audio)
    # together with the newly generated speech; 500 (~80s of audio) is sized
    # to fit the app's 80-word segment limit without clipping.
    MAX_GENERATE_LENGTH = 500

    SEED_DEFAULT = 42
    SEED_MIN = -1
    SEED_MAX = APP_SEED_MAX

    SPEAKER_SCALE_DEFAULT = 1.5
    SPEAKER_SCALE_MIN = 0.0
    SPEAKER_SCALE_MAX = 3.0

    NUM_STEPS_SOAR_DEFAULT = 10
    NUM_STEPS_SOAR_MIN = 1
    NUM_STEPS_SOAR_MAX = 32

    NUM_STEPS_MF_DEFAULT = 4
    NUM_STEPS_MF_MIN = 1
    NUM_STEPS_MF_MAX = 8

    GUIDANCE_SCALE_DEFAULT = 1.2
    GUIDANCE_SCALE_MIN = 1.0
    GUIDANCE_SCALE_MAX = 3.0

    @classmethod
    def resolve_target(cls, target: str) -> str:
        return target or cls.DEFAULT_REPO_ID

    @classmethod
    def is_meanflow_target(cls, target: str) -> bool:
        return cls.resolve_target(target) == cls.MF_REPO_ID

    @classmethod
    def get_num_steps_config(cls, target: str) -> tuple[str, str, int, int, int]:
        """
        Returns the num-steps setting that applies to the given target, as
        (project attr, variant label, default, min, max)
        """
        if cls.is_meanflow_target(target):
            return (
                "dots_num_steps_mf",
                "mf",
                cls.NUM_STEPS_MF_DEFAULT,
                cls.NUM_STEPS_MF_MIN,
                cls.NUM_STEPS_MF_MAX,
            )
        return (
            "dots_num_steps_soar",
            "soar",
            cls.NUM_STEPS_SOAR_DEFAULT,
            cls.NUM_STEPS_SOAR_MIN,
            cls.NUM_STEPS_SOAR_MAX,
        )

    @classmethod
    def resolve_num_steps(cls, project: Project) -> int:
        attr, _, default, _, _ = cls.get_num_steps_config(project.dots_target)
        value: int = getattr(project, attr)
        return default if value == -1 else value

    @classmethod
    def is_sampling_locked_target(cls, target: str) -> bool:
        return cls.resolve_target(target) in {
            cls.MF_2STEPS_REPO_ID,
            cls.MF_1STEP_REPO_ID,
        }

    @classmethod
    def is_cfg_configurable_target(cls, target: str) -> bool:
        """
        Whether the target's solver actually consumes a guidance scale.

        Only the flow-matching (SOAR) artifact runs a CFG branch. The
        meanflow variants distill CFG into the model (the solver drops the
        parameter), and the fixed-step variants lock it at 0.
        """
        resolved = cls.resolve_target(target)
        return (
            not cls.is_sampling_locked_target(resolved)
            and not cls.is_meanflow_target(resolved)
        )

    @classmethod
    def get_menu_text(
        cls, project: Project, instance: TtsBaseModel | None = None
    ) -> str:
        target = cls.resolve_target(project.dots_target)
        target = ellipsize_path_for_menu(target.removeprefix("dots-studio/"))
        return f"{cls.INFO.ui['proper_name']} {COL_DIM}({target})"

    def get_warning_issues(self, project: Project) -> list[str]:
        warnings = super().get_warning_issues(project)
        language = project.language_code.strip() if project.language_code else ""
        warnings.append(
            f"Using dots.tts language hint: {language or 'auto'}"
        )
        return warnings
