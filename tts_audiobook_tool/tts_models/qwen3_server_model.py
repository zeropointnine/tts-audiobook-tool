import os
import random

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_models.qwen3_server_base_model import Qwen3ServerBaseModel


class Qwen3ServerModel(Qwen3ServerBaseModel):

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

        voice_transcript = project.qwen3_voice_transcript

        temperature = project.qwen3_temperature
        if temperature == -1:
            temperature = Qwen3BaseModel.TEMPERATURE_FALLBACK_DEFAULT

        top_p = project.qwen3_top_p
        if top_p == -1:
            top_p = Qwen3BaseModel.TOP_P_DEFAULT

        top_k = project.qwen3_top_k
        if top_k == -1:
            top_k = Qwen3BaseModel.TOP_K_DEFAULT

        repetition_penalty = project.qwen3_repetition_penalty

        seed = -1 if force_random_seed else project.qwen3_seed
        if seed == -1:
            seed = random.randrange(0, SEED_MAX) # keep seed resolution parallel to local Qwen3/Fish server behavior

        is_streaming = on_stream_chunk is not None or on_stream_end is not None
        payloads = []
        for prompt in prompts:
            payload = {
                "input": self.prepare_text_for_inference(project, prompt),
                "stream": is_streaming,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
            if is_streaming:
                payload["response_format"] = "pcm"
            if repetition_penalty != -1:
                payload["repetition_penalty"] = repetition_penalty

            if project.qwen3_voice_file_name:
                voice_path = os.path.join(project.dir_path, project.qwen3_voice_file_name)
                data_uri = SoundUtil.make_audio_data_uri(voice_path)
                reference = {"audio_path": data_uri}
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

        return SglOmniUtil.generate_concurrent(
            SglOmniUtil.get_base_url(),
            payloads,
            print_request=print_generation_request,
        )
