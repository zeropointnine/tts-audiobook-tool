import random

from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.tts_models.fish_s2_server_base_model import FishS2ServerBaseModel
from tts_audiobook_tool.util import *
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil


class FishS2ServerModel(FishS2ServerBaseModel):

    def kill(self):
        pass

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool = False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
            voice_rotation_index: int = 0,
            print_generation_request: bool = False,
    ) -> list[Sound] | str:

        voice_path, voice_transcript = ProjectVoiceUtil.current_voice_reference_pair(
            project, TtsModelType.FISH_S2_SERVER, voice_rotation_index
        )

        temperature = project.fish_s2_temperature if project.fish_s2_temperature != -1 else FishS2BaseModel.TEMPERATURE_DEFAULT
        
        top_p = project.fish_s2_top_p if project.fish_s2_top_p != -1 else FishS2BaseModel.TOP_P_DEFAULT
        top_p = min(top_p, FishS2ServerBaseModel.TOP_K_MAX) # upper limit of server version of fish s2 is lower than local version
        
        top_k = project.fish_s2_top_k if project.fish_s2_top_k != -1 else FishS2BaseModel.TOP_K_DEFAULT
        if top_k > FishS2ServerBaseModel.TOP_K_MAX:
            # Special case, where server version has different upper bound than local version
            top_k = FishS2ServerBaseModel.TOP_K_MAX
        
        seed = -1 if force_random_seed else project.fish_s2_seed
        if seed == -1:
            seed = random.randrange(0, SEED_MAX) # use same seed for all prompts just to be safe

        is_streaming = on_stream_chunk is not None or on_stream_end is not None
        payloads = []

        for prompt in prompts:

            # Rem, sgl-omni fish s2 does NOT support seed
            payload = {
                "input": self.prepare_text_for_inference(project, prompt),
                "stream": is_streaming,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
            
            if voice_path:
                reference = {"audio_path": voice_path}
                if voice_transcript:
                    reference["text"] = voice_transcript
                payload["references"] = [reference]

            payloads.append(payload)

        if is_streaming:
            if len(payloads) != 1:
                return "Streaming generation supports exactly one prompt"
            result = SglOmniUtil.generate_streaming(
                SglOmniUtil.get_base_url(),
                payloads[0],
                on_stream_chunk=on_stream_chunk,
                on_stream_end=on_stream_end,
                should_print=print_generation_request,
            )
            if isinstance(result, str):
                return result
            return [result]

        results = SglOmniUtil.generate_concurrent(
            SglOmniUtil.get_base_url(),
            payloads,
            print_request=print_generation_request,
        )
        return results
