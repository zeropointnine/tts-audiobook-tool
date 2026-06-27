import random

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants import SEED_MAX
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel
from tts_audiobook_tool.tts_models.moss_server_base_model import MossServerBaseModel
from tts_audiobook_tool.util import *


class MossServerModel(MossServerBaseModel):
    """ 
    'Server' version of MOSS (MOSS Delay model, v1.5)
    Uses the same project settings as the local version.
    """

    def kill(self) -> None:
        pass

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool = False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
            print_generation_request: bool = False,
    ) -> list[Sound] | str:
       
        voice_transcript = project.moss_voice_transcript

        temperature = project.moss_delay_temperature if project.moss_delay_temperature != -1 else MossServerBaseModel.CONFIG.value.temperature_default
        audio_top_p = project.moss_delay_top_p if project.moss_delay_top_p != -1 else MossServerBaseModel.CONFIG.value.audio_top_p_default
        audio_top_k = project.moss_delay_top_k if project.moss_delay_top_k != -1 else MossServerBaseModel.CONFIG.value.audio_top_k_default
        
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
            if project.moss_voice_file_name:
                """
                Note how we send the voice clone data as a base64 data uri ("audio_path").
                Server docs mention that soundfile is required in the server environment,
                implying they may not be there by default (??). However, it does exist by
                default on my install of SGL-Omni. Regardless, look out for reports of 
                failed voice clone, and revisit if necessary.
                (Would need to fall back to server file path + manually entered transcript).

                Also note that data uri functionality does exist for server's MOSS implementation
                but not for Higgs V3.
                """
                voice_path = os.path.join(project.dir_path, project.moss_voice_file_name)
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
