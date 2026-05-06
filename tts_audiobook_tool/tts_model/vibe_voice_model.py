import numpy as np
import torch
from typing import Callable

from tts_audiobook_tool.app_util import AppUtil
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference # type: ignore
from vibevoice.modular.streamer import AudioStreamer # type: ignore
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor # type: ignore
from peft import PeftModel  # type: ignore
from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.tts_model.vibevoice_base_model import VibeVoiceBaseModel
from tts_audiobook_tool.util import *


class CallbackAudioStreamer(AudioStreamer):
    def __init__(
            self,
            batch_size: int,
            on_chunk: Callable[[int, torch.Tensor], None],
            on_end: Callable[[], None] | None = None,
    ):
        super().__init__(batch_size=batch_size)
        self.on_chunk = on_chunk
        self.on_end = on_end
        self.did_notify_end = False

    def put(self, audio_chunks: torch.Tensor, sample_indices: torch.Tensor):
        super().put(audio_chunks, sample_indices)
        for i, sample_idx in enumerate(sample_indices):
            self.on_chunk(int(sample_idx.item()), audio_chunks[i].detach().cpu())

    def end(self, sample_indices=None):
        super().end(sample_indices)
        if sample_indices is None and self.on_end is not None and not self.did_notify_end:
            self.did_notify_end = True
            self.on_end()


class VibeVoiceModel(VibeVoiceBaseModel):
    """
    VibeVoice TTS inference logic
    Mostly copy-pasted from: VibeVoice/demo/inference_from_file.py
    """

    def __init__(
            self,
            device_map: str,
            model_target: str = "",
            lora_path: str | None = None,
            max_new_tokens: int | None = None,
    ):

        self._device_map = device_map
        if not model_target:
            model_target = VibeVoiceBaseModel.DEFAULT_REPO_ID
        self.max_new_tokens = max_new_tokens
        self.audio_streamer: AudioStreamer | None = None
        self.stream_chunk_callback: StreamChunkCallback | None = None
        self.stream_end_callback: StreamEndCallback | None = None

        self.processor = VibeVoiceProcessor.from_pretrained(model_target)

        # Determine attention implementation type
        if "cuda" in device_map:
            try:
                from flash_attn import flash_attn_func # type: ignore
                attn_implementation = "flash_attention_2"
            except ImportError as e:
                attn_implementation = "sdpa" # fyi, in my testing, "eager" is unusable
                # ... note, triton is still required though
        else:
            attn_implementation = "sdpa" 
        printt(f"Attention implementation: {attn_implementation}")

        # Load model
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_target,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation=attn_implementation
        )
        
        if lora_path:
            # Load LoRA adapter
            printt()
            printt(f"Loading LoRA adapter from: {lora_path}")
            printt()
            try:
                # The LoRA is trained on the language model backbone (Qwen2)
                # VibeVoiceForConditionalGenerationInference wraps VibeVoiceModel
                # which contains the language_model as a submodule
                # We need to load the adapter onto the language_model submodule
                language_model = self.model.model.language_model
                
                if hasattr(language_model, 'load_adapter'):
                    language_model.load_adapter(lora_path)
                else:
                    # Fallback to PeftModel wrapper
                    printt("load_adapter() not available, using PeftModel wrapper on language_model")
                    self.model.model.language_model = PeftModel.from_pretrained(language_model, lora_path)
                printt("LoRA adapter activated successfully")
                printt()
                self._has_lora = True
            except Exception as e:
                printt(f"{COL_ERROR}WARNING: Failed to load LoRA adapter: {e}")
                printt()
                self._has_lora = False
        else:
            self._has_lora = False
        
        self.model.eval()

    @property
    def has_lora(self) -> bool:
        return self._has_lora

    def kill(self) -> None:
        if self.processor:
            self.processor.tokenizer = None
            self.processor.audio_processor = None
            self.processor.audio_normalizer = None
        self.audio_streamer = None
        self.clear_stream_state()
        self.processor = None
        self.model = None

    def clear_stream_state(self) -> None:
        super().clear_stream_state()
        self.audio_streamer = None

    def massage_for_inference(self, text: str) -> str:
        text = super().massage_for_inference(text)
        # Required speaker tag
        text = f"{SPEAKER_TAG}{text}" 
        return text

    def on_audio_chunk_received(self, sample_idx: int, chunk: torch.Tensor) -> None:
        if chunk.dtype == torch.bfloat16:
            chunk = chunk.to(torch.float32)

        chunk_data = chunk.numpy().astype(np.float32, copy=False).reshape(-1)

        if self.stream_chunk_callback is not None:
            self.stream_chunk_callback(chunk_data)

    def on_stream_end(self) -> None:
        if self.stream_end_callback is not None:
            self.stream_end_callback()

    def generate_using_project(
            self, 
            project: Project, 
            prompts: list[str], 
            force_random_seed: bool=False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
        ) -> list[Sound] | str:
        
        if project.vibevoice_voice_file_name:
            voice_path = os.path.join(project.dir_path, project.vibevoice_voice_file_name)
        else:
            voice_path = ""

        cfg_scale = VibeVoiceBaseModel.CFG_DEFAULT if project.vibevoice_cfg == -1 else project.vibevoice_cfg
        
        num_steps = VibeVoiceBaseModel.DEFAULT_NUM_STEPS if project.vibevoice_steps == -1 else project.vibevoice_steps

        seed = -1 if force_random_seed else project.vibevoice_seed

        result = self.generate(
            texts=prompts,
            voice_path=voice_path,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            seed=seed,
            on_stream_chunk=on_stream_chunk,
            on_stream_end=on_stream_end,
        )
        return result

    def generate(
            self,
            texts: list[str],
            voice_path: str,
            cfg_scale: float=VibeVoiceBaseModel.CFG_DEFAULT,
            num_steps: int=VibeVoiceBaseModel.DEFAULT_NUM_STEPS,
            seed: int = -1,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
    ) -> list[Sound] | str:
        """
        Returns list[Sound] or error string

        FYI: Couldn't pass pre-loaded sound data without inference issues for some reason,
        but in practice overhead is negligible
        """
        if self.model is None or self.processor is None:
            return "model or processor is not initialized" # logic error

        if seed <= -1:
            seed = random.randrange(0, SEED_MAX)
        AppUtil.set_seed(seed)

        try:
            self.model.set_ddpm_inference_steps(num_steps) # type: ignore
            if on_stream_chunk is not None:
                self.stream_chunk_callback = on_stream_chunk
            if on_stream_end is not None:
                self.stream_end_callback = on_stream_end

            # Ensure text is a list
            texts = texts if isinstance(texts, list) else [texts]

            voice_samples = None if not voice_path else [ [ voice_path ] for _ in texts ]

            inputs = self.processor(
                text=texts,
                voice_samples=voice_samples,  # type: ignore
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            self.audio_streamer = CallbackAudioStreamer(
                batch_size=len(texts),
                on_chunk=self.on_audio_chunk_received,
                on_end=self.on_stream_end,
            )

            # Generate audio
            outputs = self.model.generate(
                **inputs, # type: ignore
                max_new_tokens=self.max_new_tokens,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                audio_streamer=self.audio_streamer,
                # generation_config={'do_sample': False, 'temperature': 0.95, 'top_p': 0.95, 'top_k': 0},
                generation_config={'do_sample': False}, # type: ignore
                verbose=True,
            )
        except Exception as e:
            return make_error_string(e)

        if not outputs.speech_outputs: # type: ignore
            return "No audio output"
            
        sounds = []
        for i, data in enumerate(outputs.speech_outputs): # type: ignore
            if data is None:
                return f"No audio output for item {i}"

            # Is bfloat16
            tensor_data = data[0]
            if tensor_data.dtype == torch.bfloat16:
                tensor_data = tensor_data.to(torch.float32)

            ndarray_data = tensor_data.cpu().numpy()
            sound = Sound(ndarray_data, TtsModelInfos.VIBEVOICE.value.sample_rate)
            sounds.append(sound)

        return sounds

SPEAKER_TAG = "Speaker 1: "
