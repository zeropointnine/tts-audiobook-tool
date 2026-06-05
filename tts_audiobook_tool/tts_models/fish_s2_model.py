import importlib
from pathlib import Path
import random
import sys
import torch
import torchaudio # type: ignore
import huggingface_hub
from huggingface_hub.errors import GatedRepoError

from tts_audiobook_tool.app_types import Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
from tts_audiobook_tool.util import *


class FishS2Model(FishS2BaseModel): 
    """
    Fish S2 TTS inference logic

    Almost identical to FishS1Model class. Not worth de-duping code IME.
    
    Note that the updated fish_speech lib does not seem to be compatible with the older S1-mini model.
    So this class and the FishS1Model class must use different fish lib github commit points as a dependency
    (ie, must use different venvs).
    """

    def __init__(self, device: str, compile_enabled: bool):

        # TODO verify mpc; also mpc + compile? probably not presumably?
        
        self.device = device
        self._compile_enabled = (device == "cuda" and compile_enabled)

        # ------------------------------------------------------------------------------------------
        # Fish module executes this line upon import
        # `pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)`
        # Must create dummy file before import to prevent RTE
        module_name = "fish_speech.models.dac.inference"
        spec = importlib.util.find_spec(module_name) # type: ignore
        dummy_path = Path(spec.origin).parent / ".project-root"
        if not dummy_path.exists():
            open(dummy_path, 'a').close()

        from fish_speech.models.dac.inference import load_model as load_dac_model # type: ignore
        from fish_speech.models.text2semantic.inference import init_model as init_t2s_model # type: ignore
        # ------------------------------------------------------------------------------------------

        # Download model to default hf cache location if not already
        try:
            REPO_ID = "fishaudio/s2-pro"
            model_dir = huggingface_hub.snapshot_download(
                repo_id=REPO_ID,
                cache_dir=huggingface_hub.constants.HF_HUB_CACHE,
                local_files_only=False
            )

        except (FileNotFoundError, GatedRepoError) as e:
            printt()
            printt(f"{COL_ERROR}{e}")
            printt()
            printt("Make sure you have done the following:")
            printt()
            printt("[1] Visit https://huggingface.co/fishaudio/s2-pro")
            printt("    and authorize access using a logged-in Hugging Face account.")
            printt("[2] Run `hf auth login` and enter valid Hugging Face access token.")
            printt()
            printt("Will now exit.")
            exit(1)
        except Exception as e:
            printt()
            printt(f"Download failed: {type(e)}")
            printt()
            printt(f"{e}")
            printt()
            printt("Will now exit.")
            exit(1)

        dac_path = os.path.join(model_dir, "codec.pth")
        self.dac_model: Any = load_dac_model("modded_dac_vq", dac_path, self.device)

        t2s_path = model_dir
        self.t2s_model, self.decode_one_token = init_t2s_model(
            t2s_path, self.device, torch.float16, compile=self._compile_enabled
        )

        self._voice_clone: VoiceClone | None = None
        self.cached_continuation_history: list[tuple[str, torch.Tensor]] = []

        # Now that fish has printed init info, lower log level
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING", filter="fish_speech")

    def set_voice_clone_using(self, source_path: str, transcribed_text: str) -> None:

        if (
                self._voice_clone
                and source_path == self._voice_clone.source_path
                and transcribed_text == self._voice_clone.transcribed_text
        ):
            return

        self.clear_continuation()

        ref_audio, sr = torchaudio.load(source_path) # TODO error handling
        if ref_audio.shape[0] > 1:
            ref_audio = ref_audio.mean(0, keepdim=True)
        ref_audio = torchaudio.functional.resample(ref_audio, sr, self.dac_model.sample_rate)
        audios = ref_audio[None].to(self.device)

        self._voice_clone = VoiceClone(
            source_path=source_path, transcribed_text=transcribed_text, audios=audios
        )

    def clear_voice_clone(self) -> None:
        if self._voice_clone:
            self.clear_continuation()
        self._voice_clone = None

    @staticmethod
    def get_prompt_log_preview(prompt: str) -> str:
        return prompt.replace("\n", " ")[:50]

    def clear_continuation(self) -> None:
        self.cached_continuation_history.clear()

    def cache_continuation(self, prompt: str, tokens: torch.Tensor) -> None:
        self.cached_continuation_history.append((prompt.strip(), tokens.detach().cpu().clone()))

    def cache_continuation_with_budget(self, prompt: str, tokens: torch.Tensor, max_items: int) -> None:
        if max_items <= 0:
            self.clear_continuation()
            return
        self.cache_continuation(prompt, tokens)
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

    def build_prompt_reference(self, use_continuation: bool) -> tuple[list[str] | None, list[torch.Tensor] | None]:

        text_parts: list[str] = []
        token_parts: list[torch.Tensor] = []

        if self._voice_clone and self._voice_clone.prompt_tokens is not None:
            text_parts.append(self._voice_clone.transcribed_text)
            token_parts.append(self._voice_clone.prompt_tokens)

        if use_continuation:
            for continuation_text, continuation_tokens in self.cached_continuation_history:
                text_parts.append(continuation_text)
                token_parts.append(continuation_tokens)

        if not token_parts:
            return None, None

        combined_text = " ".join(part.strip() for part in text_parts if part.strip())
        token_parts_on_device = [tokens.to(self.device) for tokens in token_parts]
        combined_tokens = torch.cat(token_parts_on_device, dim=1)

        return [combined_text], [combined_tokens]

    def generate_semantic_tokens(
            self,
            prompt: str,
            prompt_text: list[str] | None,
            prompt_tokens: list[torch.Tensor] | None,
            temperature: float,
            top_p: float,
            top_k: int,
    ) -> torch.Tensor | str:

        from fish_speech.models.text2semantic.inference import generate_long # type: ignore

        for response in generate_long(
            model=self.t2s_model,
            device=self.device,
            decode_one_token=self.decode_one_token,  # type: ignore
            text=prompt,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            max_new_tokens=FISH_S2_MAX_NEW_TOKENS,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k, # type: ignore  # rem, when venv is s1, gets flagged as an error
            
            # the app already segments audiobook/conversation text into bounded
            # prompt-sized units before model inference. Keep Fish S2 from doing
            # its own internal text chunking so this wrapper remains responsible
            # for segment boundaries, continuation cache updates, and validation
            # semantics. If a prompt plus continuation context is still too long,
            # Fish's encoded prompt-length check will raise.
            chunk_length=FISH_S2_DISABLE_INTERNAL_CHUNKING_LENGTH,
        ):
            if response.action == "sample":
                if response.codes is None:
                    return "No tensor while generating semantic tokens"
                codes = response.codes.detach().cpu()
                if codes.shape[1] >= FISH_S2_MAX_NEW_TOKENS - 1:
                    return FISH_S2_MAX_TOKEN_LIMIT_ERROR
                return codes

        return "Semantic token generation failed"

    def kill(self) -> None:
        self.clear_continuation()
        # Clear all member variables in attempt to clear all resources
        self.dac_model = None
        self._voice_clone = None
        self.t2s_model = None
        self.decode_one_token = None

    def generate_using_project(
            self, 
            project: Project, 
            prompts: list[str], 
            force_random_seed: bool=False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
        ) -> list[Sound] | str:

        if len(prompts) != 1:
            raise ValueError("Implementation does not support batching")
        prompt = prompts[0]

        if project.fish_s2_voice_file_name:
            source_path = os.path.join(project.dir_path, project.fish_s2_voice_file_name)
            self.set_voice_clone_using(
                source_path=source_path,
                transcribed_text=project.fish_s2_voice_transcript
            )
        else:
            self.clear_voice_clone()

        if project.fish_s2_temperature == -1:
            temperature = FishS2BaseModel.DEFAULT_TEMPERATURE
        else:
            temperature = project.fish_s2_temperature

        if project.fish_s2_top_p == -1:
            top_p = FishS2BaseModel.DEFAULT_TOP_P
        else:
            top_p = project.fish_s2_top_p

        if project.fish_s2_top_k == -1:
            top_k = FishS2BaseModel.DEFAULT_TOP_K
        else:
            top_k = project.fish_s2_top_k

        seed = -1 if force_random_seed else project.fish_s2_seed
        rolling_continuation_max_segments = project.fish_s2_rolling_cont

        result = self.generate(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            rolling_continuation_max_segments=rolling_continuation_max_segments,
        )

        if isinstance(result, Sound):
            return [result]
        else:
            return result    

    def generate(
            self,
            prompt: str,
            temperature: float,
            top_p: float,
            top_k: int,
            seed: int,
            rolling_continuation_max_segments: int,
    ) -> Sound | str:

        rolling_continuation = rolling_continuation_max_segments > 0

        if not rolling_continuation:
            self.clear_continuation()

        if seed == -1:
            seed = random.randrange(0, SEED_MAX)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        try:
            with torch.no_grad(): # !!important

                # Step 1: Prompt tokens

                if self._voice_clone and self._voice_clone.prompt_tokens is None:
                    audio_lengths = torch.tensor([self._voice_clone.audios.shape[2]], device=self.device, dtype=torch.long)
                    voice_prompt_tokens, _ = self.dac_model.encode(self._voice_clone.audios, audio_lengths)
                    if voice_prompt_tokens.ndim == 3:
                        voice_prompt_tokens = voice_prompt_tokens[0]
                    self._voice_clone.prompt_tokens = voice_prompt_tokens

                use_continuation = rolling_continuation and bool(self.cached_continuation_history)
                if use_continuation:
                    printt(f"{COL_DIM_ITALICS}Rolling continuation enabled, history length: {len(self.cached_continuation_history)}")

                prompt_text, prompt_tokens = self.build_prompt_reference(use_continuation)

                # Step 2: Make semantic tokens using prompt tokens

                generated_tokens = self.generate_semantic_tokens(
                    prompt,
                    prompt_text,
                    prompt_tokens,
                    temperature,
                    top_p,
                    top_k,
                )

                del prompt_tokens

                if isinstance(generated_tokens, str):
                    return generated_tokens

                if rolling_continuation:
                    self.cache_continuation_with_budget(prompt, generated_tokens, rolling_continuation_max_segments)
                semantic_tokens = generated_tokens.numpy()

                # Step 3: Make audio data using semantic tokens

                decode_tokens = torch.from_numpy(semantic_tokens).to(self.device).long()
                if decode_tokens.ndim == 2:
                    decode_tokens = decode_tokens[None]  # Add batch dimension
                tensor = self.dac_model.from_indices(decode_tokens)

                del decode_tokens
                del semantic_tokens

                data = tensor[0, 0].float().cpu().detach().numpy()
                # print(data.shape)  # Should be (N,) where N is total elements

                del tensor
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                return Sound(data, self.dac_model.sample_rate)

        except Exception as e:
            return make_error_string(e)

# ---

class VoiceClone:

    def __init__(self, source_path: str, transcribed_text: str, audios: Any):

        self.source_path = source_path
        self.transcribed_text = transcribed_text
        self.audios = audios

        # Gets set on first generation
        self.prompt_tokens: Any = None


FISH_S2_MAX_NEW_TOKENS = 2048
FISH_S2_MAX_TOKEN_LIMIT_ERROR = "Fish S2 generation reached max token limit without terminating"
FISH_S2_DISABLE_INTERNAL_CHUNKING_LENGTH = 1_000_000
