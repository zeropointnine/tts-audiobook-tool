import random

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants import SEED_MAX
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.tts_models.auk_server_base_model import AuKServerBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class AuKServerModel(AuKServerBaseModel):
    """Shared SGL-Omni request implementation for AuK and AuK-Flash."""

    def kill(self) -> None:
        pass

    @staticmethod
    def estimate_generation_seconds(
        reference_seconds: float,
        input_text: str,
        reference_text: str,
        speed: float,
    ) -> float:
        """Apply AuK's duration estimate, adjusted by user-facing speech speed."""
        return (
            reference_seconds
            * len(input_text.encode("utf-8"))
            / max(1, len(reference_text.encode("utf-8")))
            / speed
        )

    def generate_using_project(
        self,
        project: Project,
        prompts: list[str],
        force_random_seed: bool = False,
        on_stream_chunk: StreamChunkCallback | None = None,
        on_stream_end: StreamEndCallback | None = None,
        voice_selection_index: int = 0,
        print_params: bool = False,
        print_generation_request: bool = False,
    ) -> list[Sound] | str:
        model_type = TtsModelType.get_by_id(self.INFO.id)
        voice_file_name, voice_transcript = ProjectVoiceUtil.current_voice_reference_pair(
            project, model_type, voice_selection_index
        )

        seed = -1 if force_random_seed else project.auk_seed
        if seed == -1:
            seed = random.randrange(0, SEED_MAX)

        voice_path = ProjectVoiceUtil.resolve_voice_file_path(project, voice_file_name)
        voice_data_uri = SoundUtil.make_audio_data_uri(voice_path)

        reference_seconds: float | None = None
        if project.auk_speed != self.SPEED_DEFAULT:
            reference_sound = SoundFileUtil.load(voice_path)
            if isinstance(reference_sound, str):
                return reference_sound
            reference_seconds = reference_sound.duration

        payloads = []
        for prompt in prompts:
            payload = {
                "input": prompt,
                "stream": False,
                "response_format": "wav",
                "seed": seed,
                "references": [
                    {
                        "audio_path": voice_data_uri,
                        "text": voice_transcript,
                    }
                ],
            }
            if reference_seconds is not None:
                gen_seconds = self.estimate_generation_seconds(
                    reference_seconds,
                    prompt,
                    voice_transcript,
                    project.auk_speed,
                )
                payload["stage_params"] = {
                    "auk_engine": {"gen_seconds": gen_seconds}
                }
            payloads.append(payload)

        return SglOmniUtil.generate_concurrent(
            SglOmniUtil.get_base_url(),
            payloads,
            print_request=print_generation_request,
        )


class AuKBaseServerModel(AuKServerModel):
    """SGL-Omni adapter for the base AuK checkpoint."""

    INFO = TtsModelType.AUK_SERVER.value


class AuKFlashServerModel(AuKServerModel):
    """SGL-Omni adapter for the four-step AuK-Flash checkpoint."""

    INFO = TtsModelType.AUK_FLASH_SERVER.value
