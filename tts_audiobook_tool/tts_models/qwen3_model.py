import random
from typing import cast

import torch

from qwen_tts import Qwen3TTSModel # type: ignore
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem # type: ignore

from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.util import *


class Qwen3Model(Qwen3BaseModel):
    """
    App wrapper for `Qwen3TTSModel`
    """

    def __init__(self, model_target: str, device: str): 
        
        self._model_target = model_target
        self._voice_info: tuple[str, str] | None = None
        self._voice_clone_prompt: VoiceClonePromptItem | None = None
        self.cached_continuation_history: list[tuple[str, torch.Tensor]] = []
        self._device = device

        if device == "cuda":             
            device_map = "cuda:0"
        else:
            device_map = device

        attn_implementation = None
        if device == "cuda":
            try:
                from flash_attn import flash_attn_func # type: ignore
                attn_implementation = "flash_attention_2"
                
                # FYI, can ignore:
                # You are attempting to use Flash Attention 2 without specifying a torch dtype. This might lead to unexpected behaviour
               
            except ImportError:
                # eat silently
                ... 

        self._model: Qwen3TTSModel = Qwen3TTSModel.from_pretrained(
                self._model_target,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )

    def kill(self) -> None:
        self.clear_continuation()
        self._model = None # type: ignore
        self._voice_clone_prompt = None

    @property
    def model_target(self) -> str:
        return self._model_target

    @property
    def model_type(self) -> str:
        return self._model.model.tts_model_type # type: ignore

    @property
    def supported_languages(self) -> list[str]:
        return self._model.get_supported_languages() or []

    @property
    def supported_speakers(self) -> list[str]:
        return self._model.get_supported_speakers() or []
    
    @property
    def generate_defaults(self) -> dict[str, Any]:
        return self._model.generate_defaults

    def clear_voice(self) -> None:
        if self._voice_info is not None or self._voice_clone_prompt is not None:
            self.clear_continuation()
        self._voice_info = None
        self._voice_clone_prompt = None

    def clear_continuation(self) -> None:
        self.cached_continuation_history.clear()

    def cache_continuation(self, prompt: str, codes: torch.Tensor, max_items: int) -> None:
        if max_items <= 0:
            self.clear_continuation()
            return
        self.cached_continuation_history.append((prompt.strip(), codes.detach().cpu().clone()))
        while self.is_continuation_history_over_budget(max_items):
            self.cached_continuation_history.pop(0)

    def is_continuation_history_over_budget(self, max_items: int) -> bool:
        return (
            len(self.cached_continuation_history) > max_items
            or self.get_continuation_history_word_count() > ROLLING_CONTINUATION_MAX_WORDS
        )

    def get_continuation_history_word_count(self) -> int:
        return sum(
            self.get_word_count(prompt)
            for prompt, _ in self.cached_continuation_history
        )

    @staticmethod
    def get_word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def get_prompt_log_preview(prompt: str) -> str:
        return prompt.replace("\n", " ")[:50]

    def build_continuation_voice_clone_prompt(self, use_continuation: bool) -> VoiceClonePromptItem:
        if self._voice_clone_prompt is None:
            raise ValueError("Voice clone prompt is not initialized")

        if not use_continuation:
            return self._voice_clone_prompt

        if self._voice_clone_prompt.ref_code is None:
            raise ValueError("Qwen3 Base rolling continuation requires ICL ref_code")

        text_parts = [self._voice_clone_prompt.ref_text or ""]
        code_parts = [self._voice_clone_prompt.ref_code.detach().cpu()]
        for continuation_text, continuation_codes in self.cached_continuation_history:
            text_parts.append(continuation_text)
            code_parts.append(continuation_codes.detach().cpu())

        combined_text = " ".join(part.strip() for part in text_parts if part.strip())
        combined_codes = torch.cat(code_parts, dim=0)

        return VoiceClonePromptItem(
            ref_code=combined_codes,
            ref_spk_embedding=self._voice_clone_prompt.ref_spk_embedding.detach().cpu(),
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text=combined_text,
        )

    def generate_using_project(
            self, 
            project: Project, 
            prompts: list[str], 
            force_random_seed: bool=False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
    ) -> list[Sound] | str:

        language = self.resolve_language_code_and_warning(project.language_code)[0]
        temperature = project.qwen3_temperature
        top_k = project.qwen3_top_k
        top_p = project.qwen3_top_p
        repetition_penalty = project.qwen3_repetition_penalty
        seed = -1 if force_random_seed else project.qwen3_seed

        match self.model_type:
            
            case "base":

                can = project.qwen3_voice_file_name and project.qwen3_voice_transcript
                if can:
                    voice_info = (
                        os.path.join(project.dir_path, project.qwen3_voice_file_name),
                        project.qwen3_voice_transcript
                    )
                    result = self.generate_base(
                        prompts=prompts,
                        voice_info=voice_info,
                        language=language,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        seed=seed,
                        rolling_continuation_max_segments=project.qwen3_rolling_cont,
                    )
                else:
                    result = "Missing voice path or transcript"

            case "custom_voice":

                speakers = self.supported_speakers
                speaker_id = project.qwen3_speaker_id
                if len(speakers) == 1:
                    if not speaker_id and speaker_id != speakers[0]:
                        ... # print warning maybe
                    speaker_id = speakers[0] # force default
                if speaker_id not in speakers:
                    result = f"Invalid speaker id for this model: {speaker_id}"
                else:
                    result = self.generate_custom_voice(
                        prompts=prompts,
                        speaker_id=speaker_id,
                        instruct=project.qwen3_instructions,
                        language=language,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        seed=seed
                    )
            
            case "voice_design":

                result = self.generate_voice_design(
                    prompts=prompts,
                    instruct=project.qwen3_instructions,
                    language=language,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    seed=seed
                )
            
            case _:
                result = f"Unsupported model type: {self.model_type}"

        return result

    def generate_base(
          self,
          prompts: list[str],
           voice_info: tuple[str, str],
           language: str,
           temperature: float = -1,
            top_k: int = -1,
            top_p: float = -1,
            repetition_penalty: float = -1,
            seed: int = -1,
             rolling_continuation_max_segments: int = 0,
    ) -> list[Sound] | str:
        """
        Generate function for model_type="base" (voice clone-based)
        """
        
        if not prompts or not voice_info[0] or not voice_info[1]:
            return "Missing required parameter"

        if not self._voice_clone_prompt or self._voice_info != voice_info:
            if self._voice_info != voice_info:
                self.clear_continuation()
            # Cache _voice_clone_prompt 
            try:
                self._voice_clone_prompt = self._model.create_voice_clone_prompt(
                    ref_audio=voice_info[0],
                    ref_text=voice_info[1],
                    x_vector_only_mode=False,
                )[0]
            except Exception as e:
                return f"Couldn't create voice clone for {voice_info[0]} - {make_error_string(e)}"
            self._voice_info = voice_info

        rolling_continuation = rolling_continuation_max_segments > 0

        if not rolling_continuation:
            self.clear_continuation()

        if rolling_continuation and len(prompts) != 1:
            return "Qwen3 Base rolling continuation does not support batching"

        if rolling_continuation and self.cached_continuation_history:
            printt(f"{COL_DIM_ITALICS}Rolling continuation enabled, history length: {len(self.cached_continuation_history)}")

        use_continuation = rolling_continuation and bool(self.cached_continuation_history)
        voice_clone_prompt = self.build_continuation_voice_clone_prompt(use_continuation)
        voice_clone_prompts = [voice_clone_prompt for _ in prompts]
        
        languages = [language for _ in prompts]

        if seed == -1:
            seed = random.randrange(0, SEED_MAX)
        app_support.set_seed(seed)

        # Inference code does not print its own feedback, so add some, matching behvior of other models
        printt(f"{COL_DIM_ITALICS}Generating...", dont_reset=True) 
        if use_continuation:
            printt(
                f"{COL_DIM_ITALICS}Qwen3 Base: using rolling continuation "
                f"history_items={len(self.cached_continuation_history)} "
                f"text={self.get_prompt_log_preview(prompts[0])!r}",
                dont_reset=True,
            )

        try:
            gen_kwargs = self._build_gen_kwargs(
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p,
                repetition_penalty=repetition_penalty, 
                seed=seed
            )
            wavs, sr, generated_codes = self.generate_base_with_codes(
                prompts=prompts,
                voice_clone_prompt=voice_clone_prompts,
                languages=languages,
                gen_kwargs=gen_kwargs,
            )
            if rolling_continuation:
                self.cache_continuation(prompts[0], generated_codes[0], rolling_continuation_max_segments)
            
            # FYI, this output can be ignored:
            # setting `pad_token_id` to `eos_token_id`:2150 for open-end generation.
        
        except Exception as e:
            return str(e)
        
        sounds = [Sound(wav, sr) for wav in wavs]
        return sounds

    def generate_base_with_codes(
        self,
        prompts: list[str],
        voice_clone_prompt: list[VoiceClonePromptItem],
        languages: list[str],
        gen_kwargs: dict[str, Any],
    ) -> tuple[list[Any], int, list[torch.Tensor]]:
        """
        Local equivalent of the library's generate_voice_clone(), except it
        returns generated codec codes so Base-mode rolling continuation can use
        them as future ICL reference codes.
        """
        input_texts = [self._model._build_assistant_text(text) for text in prompts]
        input_ids = self._model._tokenize_texts(input_texts)
        voice_clone_prompt_dict = self._model._prompt_items_to_voice_clone_prompt(voice_clone_prompt)

        ref_ids = []
        for item in voice_clone_prompt:
            ref_text = item.ref_text
            if ref_text is None or ref_text == "":
                ref_ids.append(None)
            else:
                ref_tok = self._model._tokenize_texts([self._model._build_ref_text(ref_text)])[0]
                ref_ids.append(ref_tok)

        merged_gen_kwargs = self._model._merge_generate_kwargs(**gen_kwargs)
        talker_codes_list, _ = self._model.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=cast(Any, voice_clone_prompt_dict),
            languages=languages,
            non_streaming_mode=True,
            **merged_gen_kwargs,
        )

        codes_for_decode = []
        ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
        for i, codes in enumerate(talker_codes_list):
            if ref_code_list is not None and ref_code_list[i] is not None:
                codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
            else:
                codes_for_decode.append(codes)

        speech_tokenizer = self._model.model.speech_tokenizer
        if speech_tokenizer is None:
            raise RuntimeError("Qwen3 speech tokenizer is not initialized")

        wavs_all, sample_rate = speech_tokenizer.decode(
            [{"audio_codes": code} for code in codes_for_decode]
        )

        wavs_out = []
        for i, wav in enumerate(wavs_all):
            if ref_code_list is not None and ref_code_list[i] is not None:
                ref_len = int(ref_code_list[i].shape[0])
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wavs_out.append(wav[cut:])
            else:
                wavs_out.append(wav)

        generated_codes = [codes.detach().cpu() for codes in talker_codes_list]
        return wavs_out, sample_rate, generated_codes

    def generate_custom_voice(
        self,
        prompts: list[str],
        speaker_id: str,
        instruct: str,
        language: str,
        temperature: float = -1,
        top_k: int = -1,
        top_p: float = -1,
        repetition_penalty: float = -1,
        seed: int = -1
    ) -> list[Sound] | str:

        speaker_ids = [speaker_id for _ in prompts]
        languages = [language for _ in prompts]
        instructs = [instruct for _ in prompts] if instruct else None

        if seed == -1:
            seed = random.randrange(0, SEED_MAX)
        app_support.set_seed(seed)

        # Inference code does not print its own feedback, so add some, matching behvior of other models
        printt(f"{COL_DIM_ITALICS}Generating...", dont_reset=True) 

        try:
            gen_kwargs = self._build_gen_kwargs(
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty, seed=seed
            )
            wavs, sr = self._model.generate_custom_voice(
                text=prompts,
                speaker=speaker_ids,
                language=languages,
                instruct = instructs,
                non_streaming_mode=True,
                **gen_kwargs,
            )
        except Exception as e:
            return str(e)

        sounds = [Sound(wav, sr) for wav in wavs]
        return sounds

    def generate_voice_design(
        self,
        prompts: list[str],
        instruct: str,
        language: str,
        temperature: float = -1,
        top_k: int = -1,
        top_p: float = -1,
        repetition_penalty: float = -1,
        seed: int = -1
    ) -> list[Sound] | str:

        languages = [language for _ in prompts]
        instructs = [instruct for _ in prompts] 

        if seed == -1:
            seed = random.randrange(0, SEED_MAX)
        app_support.set_seed(seed)

        # Inference code does not print its own feedback, so add some, matching behvior of other models
        printt("{COL_DIM_ITALICS}Generating...", dont_reset=True) 

        try:
            gen_kwargs = self._build_gen_kwargs(
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty, seed=seed
            )
            wavs, sr = self._model.generate_voice_design(
                text=prompts,
                language=languages,
                instruct = instructs,
                non_streaming_mode=True,
                **gen_kwargs,
            )
        except Exception as e:
            return str(e)

        sounds = [Sound(wav, sr) for wav in wavs]
        return sounds

    def _build_gen_kwargs(
            self,
            temperature: float = -1,
            top_k: int = -1,
            top_p: float = -1,
            repetition_penalty: float = -1,
            seed: int = -1
    ) -> dict[str, Any]:
        """
        Build kwargs dict for library generate calls.
        Only includes parameters where the user has explicitly set a value (not -1 sentinel).
        The library's _merge_generate_kwargs will apply defaults for any missing params.

        The Qwen3-TTS architecture has two sampling components: a "main talker" that generates
        the lead codebook token at each step, and a "subtalker" (code_predictor) that generates
        the remaining codebook tokens. The subtalker has its own parallel sampling parameters
        (subtalker_top_p, subtalker_top_k, subtalker_temperature) that use the same value ranges
        as the conventional top_p, top_k, and temperature. Because the subtalker generates
        (num_code_groups - 1) tokens per step vs the main talker's 1, the subtalker's sampling
        parameters actually have a greater effect on the output than the conventional ones.
        To ensure user settings apply uniformly, top_p, top_k, and temperature are mirrored
        to their subtalker counterparts. repetition_penalty has no subtalker equivalent.
        """
        kwargs: dict[str, Any] = {}
        if temperature != -1:
            kwargs["temperature"] = temperature
            kwargs["subtalker_temperature"] = temperature
        else:
            resolved = self._model.generate_defaults.get("temperature", None)
            if resolved is None:
                resolved = Qwen3BaseModel.TEMPERATURE_FALLBACK_DEFAULT
            kwargs["temperature"] = resolved
        if top_k != -1:
            kwargs["top_k"] = top_k
            kwargs["subtalker_top_k"] = top_k
        if top_p != -1:
            kwargs["top_p"] = top_p
            kwargs["subtalker_top_p"] = top_p
        if repetition_penalty != -1:
            kwargs["repetition_penalty"] = repetition_penalty
        if seed != -1:
            kwargs["seed"] = seed
        return kwargs
