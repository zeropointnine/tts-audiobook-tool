
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.tts_models.zonos2_server_base_model import Zonos2ServerBaseModel


class Zonos2ServerModel(Zonos2ServerBaseModel):

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
        """
        SGL-Omni notes/limitations circa 2026-08:

        - Voice transcript not supported, though local reference implementation does
        - Seed not supported 
        - "Accurate mode" is always True and cannot be changed (but why?).
        - Min-p is not actually hooked up, apparently
        - emotion_* featureset not supported
        - We are NOT using "language" param, w/c is used to apply model's own prompt normalization logic
        """

        voice_file_name = ProjectVoiceUtil.current_voice_value(
            project, TtsModelType.ZONOS2_SERVER, voice_selection_index
        )

        top_k = project.zonos2_top_k
        if top_k == -1:
            top_k = Zonos2ServerBaseModel.TOP_K_DEFAULT

        temperature = project.zonos2_temperature
        if temperature == -1:
            temperature = Zonos2ServerBaseModel.TEMPERATURE_DEFAULT

        repetition_penalty = project.zonos2_repetition_penalty
        if repetition_penalty == -1:
            repetition_penalty = Zonos2ServerBaseModel.REPETITION_PENALTY_DEFAULT

        is_streaming = on_stream_chunk is not None or on_stream_end is not None
        payloads = []
        for prompt in prompts:

            # Allows for 40 tokens per prompt word (conservative value), 
            # plus 200 tokens of overhead, capped to 4096.
            max_new_tokens = min(200 + (len(prompt.split()) * 40), 4096)
 
            payload = {
                "input": self.prepare_text_for_inference(project, prompt),
                "stream": is_streaming,
                "max_new_tokens": max_new_tokens,
                "top_k": top_k,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
            }
            if voice_file_name:
                voice_path = ProjectVoiceUtil.resolve_voice_file_path(project, voice_file_name)
                voice_data_uri = SoundUtil.make_audio_data_uri(voice_path)
                payload["references"] = [{
                    "audio_path": voice_data_uri,
                }]
            if is_streaming:
                payload["response_format"] = "pcm"
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
