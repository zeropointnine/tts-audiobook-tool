import random

from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants import SEED_MAX
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel, MossConfigs
from tts_audiobook_tool.tts_models.moss_server_base_model import MossServerBaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil


class MossServerModel(MossServerBaseModel):
    """Shared SGL-Omni request implementation for MOSS architectures."""

    def kill(self) -> None:
        pass

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool = False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
            voice_selection_index: int = 0,
            print_generation_request: bool = False,
    ) -> list[Sound] | str:
       
        model_type = TtsModelType.get_by_id(self.INFO.id)
        voice_file_name, voice_transcript = ProjectVoiceUtil.current_voice_reference_pair(
            project, model_type, voice_selection_index
        )

        temperature, audio_top_p, audio_top_k = self.get_generation_params(project, self.CONFIG)
        
        seed = -1 if force_random_seed else project.moss_seed
        if seed == -1:
            seed = random.randrange(0, SEED_MAX) # use same seed for all prompts just to be safe

        payloads = []
        for prompt in prompts:

            payload = {
                "input": self.prepare_text_for_inference(project, prompt),
                "stream": False,
                "temperature": temperature,
                "audio_top_p": audio_top_p,
                "audio_top_k": audio_top_k,
                "seed": seed,
                "max_new_tokens": MossServerBaseModel.MAX_NEW_TOKENS
            }

            language = MossBaseModel.get_language_name(project.language_code)
            if language:
                payload["language"] = language

            # Add voice clone arguments
            if voice_file_name:
                """
                Note how we send the voice clone data as a base64 data uri ("audio_path").
                Server docs mention that soundfile is required in the server environment,
                implying they may not be there by default (??). However, it does exist by
                default on my install of SGL-Omni. Regardless, look out for reports of 
                failed voice clone, and revisit if necessary.
                (Would need to fall back to server file path + manually entered transcript).

                Also note that data URI functionality now exists for the server's Higgs V3
                implementation as well.
                """
                voice_path = os.path.join(project.dir_path, voice_file_name)
                data_uri = SoundUtil.make_audio_data_uri(voice_path)
                reference = {"audio_path": data_uri}
                
                if voice_transcript:
                    reference["text"] = voice_transcript
                
                payload["references"] = [reference]

            payloads.append(payload)

        results = SglOmniUtil.generate_concurrent(
            SglOmniUtil.get_base_url(),
            payloads,
            print_request=print_generation_request,
        )
        return results


class MossDelayServerModel(MossServerModel):
    """SGL-Omni adapter for the MOSS Delay architecture."""

    INFO = TtsModelType.MOSS_DELAY_SERVER.value
    CONFIG = MossConfigs.DELAY


class MossLocalServerModel(MossServerModel):
    """SGL-Omni adapter for the MOSS Local Transformer architecture."""

    INFO = TtsModelType.MOSS_LOCAL_SERVER.value
    CONFIG = MossConfigs.LOCAL
