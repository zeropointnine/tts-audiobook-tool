from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class NoneBaseModel(TtsBaseModel):

    INFO = TtsModelInfos.NONE.value

    def set_voice_clone(self, path: str) -> None:
        ...

    def clear_voice_clone(self) -> None:
        ...

    def set_params(self, temperature: float, max_new_tokens: int) -> None:
        ...

    @classmethod
    def get_prereq_errors(
            cls, project: Project, instance: TtsBaseModel | None, is_short: bool
    ) -> list[str]:
        err = "requires TTS model" if is_short else "No TTS model; the app must be launched from a properly configured virtual environment"
        return [err]

class NoneModel(NoneBaseModel):

    def kill(self) -> None:
        ...

    def generate_using_project(
            self, project: Project, prompts: list[str], force_random_seed: bool=False
    ) -> list[Sound] | str:
        return "No TTS model"

