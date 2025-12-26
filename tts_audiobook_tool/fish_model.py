import importlib
from pathlib import Path
import sys
import time
import torch
import torchaudio # type: ignore
import huggingface_hub
from huggingface_hub.errors import GatedRepoError

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.tts_model import FishModelProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *


class FishModel(FishModelProtocol):
    """
    Fish TTS inference logic

    Pieced together from the two main fish inference scripts:
    fish_speech/models/dac/inference.py
    fish_speech/models/text2semantic/inference.py
    See: https://github.com/fishaudio/fish-speech/blob/main/docs/en/inference.md
    """

    def __init__(self, device: str):

        super().__init__(info=TtsModelInfos.FISH.value)

        # TODO verify mpc; also mpc + compile? probably not presumably?
        self.device = device

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

        # Download model to default hf cache location
        try:
            REPO_ID = "fishaudio/openaudio-s1-mini"

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
            printt("[1] Visit https://huggingface.co/fishaudio/openaudio-s1-mini")
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
            t2s_path, self.device, torch.float16, compile=(self.device == "cuda")
        )

        self._voice_clone: VoiceClone | None = None

        # Now that fish has printed init info, lower log level
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING", filter="fish_speech")


    def set_voice_clone_using(self, source_path: str, transcribed_text: str) -> None:

        if self._voice_clone and source_path == self._voice_clone.source_path:
            return

        ref_audio, sr = torchaudio.load(source_path) # TODO error handling
        if ref_audio.shape[0] > 1:
            ref_audio = ref_audio.mean(0, keepdim=True)
        ref_audio = torchaudio.functional.resample(ref_audio, sr, self.dac_model.sample_rate)
        audios = ref_audio[None].to(self.device)

        self._voice_clone = VoiceClone(
            source_path=source_path, transcribed_text=transcribed_text, audios=audios
        )

    def clear_voice_clone(self) -> None:
        self._voice_clone = None


    def kill(self) -> None:
        # Clear all member variables in attempt to clear all resources
        self.dac_model = None
        self._voice_clone = None
        self.t2s_model = None
        self.decode_one_token = None


    def generate(
            self, 
            text: str, 
            temperature: float = -1,
            seed: int = -1
    ) -> Sound | str:

        if temperature == -1:
            temperature = FishModelProtocol.DEFAULT_TEMPERATURE

        if seed == -1:
            seed = random.randrange(0, 2**32 - 1)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        try:
            with torch.no_grad(): # !!important

                # Step 1: Prompt tokens

                if self._voice_clone and self._voice_clone.prompt_tokens is None:
                    audio_lengths = torch.tensor([self._voice_clone.audios.shape[2]], device=self.device, dtype=torch.long)
                    prompt_tokens, _ = self.dac_model.encode(self._voice_clone.audios, audio_lengths)
                    if prompt_tokens.ndim == 3:
                        prompt_tokens = prompt_tokens[0]
                    self._voice_clone.prompt_tokens = prompt_tokens

                prompt_tokens = self._voice_clone.prompt_tokens if self._voice_clone else None

                # Step 2: Make semantic tokens using prompt tokens

                from fish_speech.models.text2semantic.inference import generate_long # type: ignore

                prompt_text = self._voice_clone.transcribed_text if self._voice_clone else None

                # Not currently changing these params:
                # top_p=0.5,
                # repetition_penalty=1.5

                semantic_tokens = None
                for response in generate_long(
                    model=self.t2s_model,
                    device=self.device,
                    decode_one_token=self.decode_one_token,  # type: ignore
                    text=text,
                    prompt_text=prompt_text,
                    prompt_tokens=prompt_tokens,
                    temperature=temperature
                ):
                    if response.action == "sample":
                        if response.codes is None:
                            return "No tensor while generating semantic tokens"
                        semantic_tokens = response.codes.cpu().numpy()
                        break  # Assuming we only need the first sample

                del prompt_tokens

                if semantic_tokens is None:
                    return "Semantic token generation failed"

                # Step 3: Make audio data using semantic tokens

                prompt_tokens = torch.from_numpy(semantic_tokens).to(self.device).long()
                if prompt_tokens.ndim == 2:
                    prompt_tokens = prompt_tokens[None]  # Add batch dimension
                indices_lens = torch.tensor([prompt_tokens.shape[2]], device=self.device, dtype=torch.long)
                tensor, _ = self.dac_model.decode(prompt_tokens, indices_lens)

                del prompt_tokens
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
