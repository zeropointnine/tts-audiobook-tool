import random

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.higgs_v3_server_base_model import HiggsV3ServerBaseModel
from tts_audiobook_tool.util import *


class HiggsV3ServerModel(HiggsV3ServerBaseModel):

    def kill(self):
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
       
        voice_path = project.higgs_v3_voice_file_path
        voice_transcript = project.higgs_v3_voice_transcript

        temperature = project.higgs_v3_temperature if project.higgs_v3_temperature != -1 else HiggsV3ServerBaseModel.DEFAULT_TEMPERATURE
        top_p = project.higgs_v3_top_p if project.higgs_v3_top_p != -1 else HiggsV3ServerBaseModel.DEFAULT_TOP_P
        top_k = project.higgs_v3_top_k if project.higgs_v3_top_k != -1 else HiggsV3ServerBaseModel.DEFAULT_TOP_K
        
        seed = -1 if force_random_seed else project.higgs_v3_seed
        if seed == -1:
            seed = random.randrange(0, SEED_MAX) # use same seed for all prompts just to be safe

        is_streaming = on_stream_chunk is not None or on_stream_end is not None
        payloads = []

        for prompt in prompts:

            payload = {
                "input": self.prepare_text_for_inference(project, prompt),
                "stream": is_streaming,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": HiggsV3ServerBaseModel.MAX_TOKENS
            }
            # TODO: Re-add seed after presumed upstream bug is fixed: "seed": seed

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
