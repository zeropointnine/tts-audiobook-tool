import logging
import os
import random
import time
import torch

from glm_tts.cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd # type: ignore
from glm_tts.utils import seed_util # type: ignore
from glm_tts.utils import tts_model_util, yaml_util # type: ignore
from transformers import AutoTokenizer, LlamaForCausalLM # type: ignore
from glm_tts.llm.glmtts import GLMTTS # type: ignore
from glm_tts.utils.audio import mel_spectrogram # type: ignore
from functools import partial

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model import GlmModelProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import printt

class GlmModel(GlmModelProtocol):
    """
    Code adapted from: https://github.com/zai-org/GLM-TTS, glmtts_inference.py
    """

    def __init__(self, device: str, sample_rate: int, use_phoneme: bool=False):

        super().__init__(TtsModelInfos.GLM.value)

        if not device.startswith("cuda"):
            # TODO: reference code only mentions cuda and cpu; try mps
            raise ValueError("Only CUDA is supported for the GLM-TTS model at the moment")
        self.device = device

        if sample_rate not in [24000, 32000]:
            raise ValueError(f"Unsupported sampling_rate: {sample_rate}")
        self.sample_rate = sample_rate

        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Download huggingface model if not already
        import huggingface_hub
        try:
            ckpt_path = huggingface_hub.snapshot_download(
                repo_id="zai-org/GLM-TTS",
                cache_dir=huggingface_hub.constants.HF_HUB_CACHE,
                local_files_only=False
            )
        except Exception as e:
            printt()
            printt(f"Download failed: {type(e)}")
            printt()
            printt(f"{e}")
            printt()
            printt("Will now exit.")
            exit(1)

        # glm_tts.utils.audio uses global variables ("bad!"),
        # which must be cleared on subsequent instantiations of GlmModel
        import glm_tts.utils.audio as audio # type: ignore
        audio.mel_basis = {}
        audio.hann_window = {}

        # TODO: Not currently parameterized
        self.use_phoneme = False
        self.use_cache = True

        self.uttid_counter = 0

        # Load Models 
        self.frontend, \
        self.text_frontend, \
        self.speech_tokenizer, \
        self.llm, \
        self.flow \
            = load_models(device=device, ckpt_path=ckpt_path, use_phoneme=self.use_phoneme, sample_rate=sample_rate)

    def kill(self) -> None:
        self.frontend = None
        self.text_frontend = None
        self.speech_tokenizer = None
        self.llm = None
        self.flow = None

    def generate(
        self,
        prompt_text: str,
        prompt_speech: str,
        syn_text: str,
        seed=-1
    ):
        assert(self.frontend is not None)
        assert(self.text_frontend is not None)

        # Text Normalization
        prompt_text = self.text_frontend.text_normalize(prompt_text) or ""
        synth_text = self.text_frontend.text_normalize(syn_text)

        prompt_text_token = self.frontend._extract_text_token(prompt_text+" ")
        prompt_speech_token = self.frontend._extract_speech_token( [prompt_speech] )
        speech_feat = self.frontend._extract_speech_feat(prompt_speech, sample_rate=self.sample_rate)
        embedding = self.frontend._extract_spk_embedding(prompt_speech)
        cache_speech_token = [prompt_speech_token.squeeze().tolist()]
        flow_prompt_token = torch.tensor(
            cache_speech_token, dtype=torch.int32
        ).to(self.device)

        if seed == -1:
            seed = random.randrange(0, 2**32 - 1)

        # Initialize Cache
        cache = {
            "cache_text": [prompt_text],
            "cache_text_token": [prompt_text_token],
            "cache_speech_token": cache_speech_token,
            "use_cache": self.use_cache,
        }

        logging.info(f"Processing: {self.uttid_counter}, Syn_text: {syn_text}")

        # Run Generation
        tts_speech, _, _, text_tn_dict = generate_long(
            frontend=self.frontend,
            text_frontend=self.text_frontend,
            llm=self.llm,
            flow=self.flow,
            text_info=[str(self.uttid_counter), synth_text],
            cache=cache,
            embedding=embedding,
            seed=seed,
            flow_prompt_token=flow_prompt_token,
            speech_feat=speech_feat,
            device=self.device,
            use_phoneme=self.use_phoneme,
        )

        # TODO: Not rly sure what uttid does; I'm mostly just following the glm library example, 
        # where each item in the example*.jsonl files is given an incrementing value
        self.uttid_counter += 1
        
        # TODO: Inspect this more
        # json.dumps(text_tn_dict, ensure_ascii=False, indent=2)

        # Save Wave and Tokens
        
        sound_data = tts_speech.detach().cpu().flatten().numpy()
        return Sound(sound_data, self.sample_rate)

# ---

def get_special_token_ids(tokenize_fn):
    """
    Get special token IDs based on the tokenizer name.
    """
    _special_token_ids = {
        "ats": "<|audio_0|>",
        "ate": "<|audio_32767|>",
        "boa": "<|begin_of_audio|>",
        "eoa": "<|user|>",
        "pad": "<|endoftext|>",
    }

    special_token_ids = {}

    # Validation
    endoftext_id = tokenize_fn("<|endoftext|>")[0]
    for k, v in _special_token_ids.items():
        __ids = tokenize_fn(v)
        # Check 1: Special token length must be 1
        if len(__ids) != 1:
            raise AssertionError(
                f"Token '{k}' ({v}) encoded to multiple tokens: {__ids}"
            )
        # Check 2: Special token ID must be >= endoftext_id
        if __ids[0] < endoftext_id:
            raise AssertionError(
                f"Token '{k}' ({v}) ID {__ids[0]} is smaller than endoftext ID {endoftext_id}"
            )

        special_token_ids[k] = __ids[0]

    return special_token_ids

def _assert_shape_and_get_len(token):
    assert token.ndim == 2 and token.shape[0] == 1
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(token.device)
    return token_len

def load_frontends(
        ckpt_path: str,
        speech_tokenizer, device: str, 
        sample_rate=24000, 
        use_phoneme=False
):
    
    import glm_tts.frontend # type: ignore
    frontend_dir = os.path.dirname(glm_tts.frontend.__file__)

    if sample_rate == 32000:
        feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=640, n_fft=2560, num_mels=80, win_size=2560, fmin=0, fmax=8000, center=False)
        print("Configured for 32kHz frontend.")
    elif sample_rate == 24000:
        feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=480, n_fft=1920, num_mels=80, win_size=1920, fmin=0, fmax=8000, center=False)
        print("Configured for 24kHz frontend.")
    else:
        raise ValueError(f"Unsupported sampling_rate: {sample_rate}")

    glm_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(ckpt_path, 'vq32k-phoneme-tokenizer'), trust_remote_code=True
    )

    tokenize_fn = lambda text: glm_tokenizer.encode(text)

    frontend = TTSFrontEnd(
        tokenize_fn,
        speech_tokenizer,
        feat_extractor,
        os.path.join(frontend_dir, "campplus.onnx"),
        os.path.join(frontend_dir, "spk2info.pt"),
        device
    )
    text_frontend = TextFrontEnd(use_phoneme)
    return frontend, text_frontend


def local_llm_forward(
    llm,
    prompt_text_token,
    tts_text_token,
    prompt_speech_token,
    beam_size=1,
    sampling=25,
    sample_method="ras",
):
    """
    Single LLM forward pass.
    """
    prompt_text_token_len = _assert_shape_and_get_len(prompt_text_token)
    tts_text_token_len = _assert_shape_and_get_len(tts_text_token)
    prompt_speech_token_len = _assert_shape_and_get_len(prompt_speech_token)

    tts_speech_token = llm.inference(
        text=tts_text_token,
        text_len=tts_text_token_len,
        prompt_text=prompt_text_token,
        prompt_text_len=prompt_text_token_len,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        beam_size=beam_size,
        sampling=sampling,
        sample_method=sample_method,
        spk=None,  # No specific speaker embedding needed for generic pretrain inference here
    )
    return tts_speech_token[0].tolist()


def local_flow_forward(flow, token_list, prompt_speech_tokens, speech_feat, embedding):
    """
    Single Flow forward pass.
    """
    wav, full_mel = flow.token2wav_with_cache(
        token_list,
        prompt_token=prompt_speech_tokens,
        prompt_feat=speech_feat,
        embedding=embedding,
    )
    return wav.detach().cpu(), full_mel


# --- Helper Function: Get Prompt from Cache ---
def get_cached_prompt(cache, synth_text_token, device):
    """
    Constructs prompt tokens from the cache.
    Prunes the cache if the sequence length exceeds MAX_LLM_SEQ_INP_LEN.
    """
    cache_text = cache["cache_text"]
    cache_text_token = cache["cache_text_token"]
    cache_speech_token = cache["cache_speech_token"]

    def __len_cache_text_token():
        return sum(map(lambda x: x.shape[1], cache_text_token))

    def __len_cache_speech_token():
        return sum(map(len, cache_speech_token))

    # Estimate required length ratio
    # Avoid division by zero
    text_len = __len_cache_text_token()
    ta_ratio = __len_cache_speech_token() / (text_len if text_len > 0 else 1.0)

    __len_synth_text_token = synth_text_token.shape[1]
    __len_synth_audi_token_estim = int(ta_ratio * __len_synth_text_token)

    # Prune cache if too long.
    # Logic: Keep the first item (original prompt), remove from the second item onwards.
    while (
        __len_cache_speech_token() + __len_synth_audi_token_estim > MAX_LLM_SEQ_INP_LEN
    ):
        if len(cache_speech_token) <= 1:
            break  # Always keep at least the original prompt
        # logging.debug(f'[get_cached_prompt] Cache pop. Text count before: {len(cache_text)}')
        cache_text.pop(1)
        cache_text_token.pop(1)
        cache_speech_token.pop(1)

    # Construct Text Prompt
    prompt_text_token_from_cache = []
    for a_token in cache_text_token:
        prompt_text_token_from_cache.extend(a_token.squeeze().tolist())

    prompt_text_token = torch.tensor([prompt_text_token_from_cache]).to(device)

    # Construct Speech Prompt
    speech_tokens = []
    for a_cache_speech_token in cache_speech_token:
        speech_tokens.extend(a_cache_speech_token)

    llm_speech_token = torch.tensor([speech_tokens], dtype=torch.int32).to(device)

    return prompt_text_token, llm_speech_token


# --- Main Generation Logic ---

def generate_long(
    frontend: TTSFrontEnd,
    text_frontend: TextFrontEnd,
    llm,
    flow,
    text_info,
    cache,
    device,
    embedding,
    seed=0,
    sample_method="ras",
    flow_prompt_token=None,
    speech_feat=None,
    local_llm_forward=local_llm_forward,
    local_flow_forward=local_flow_forward,
    use_phoneme=False,
):
    outputs = []
    full_mels = []
    output_token_list = []
    uttid = text_info[0]
    syn_text = text_info[1]
    text_tn_dict = {
        "uttid": uttid,
        "syn_text": syn_text,
        "syn_text_tn": [],
        "syn_text_phoneme": [],
    }
    short_text_list = text_frontend.split_by_len(syn_text)

    for _, tts_text in enumerate(short_text_list):
        seed_util.set_seed(seed)
        tts_text_tn = text_frontend.text_normalize(tts_text) # Normalize again after splitting
        tts_text_tn = tts_text_tn or ""
        text_tn_dict["syn_text_tn"].append(tts_text_tn)
        if use_phoneme:
            tts_text_tn = text_frontend.g2p_infer(tts_text_tn)
            text_tn_dict["syn_text_phoneme"].append(tts_text_tn)
        tts_text_token = frontend._extract_text_token(tts_text_tn)

        # Access cache references
        cache_text = cache["cache_text"]
        cache_text_token = cache["cache_text_token"]
        cache_speech_token = cache["cache_speech_token"]

        # Determine Prompts
        if cache["use_cache"] and len(cache_text_token) > 1:
            prompt_text_token, prompt_speech_token = get_cached_prompt(
                cache, tts_text_token, device
            )
        else:
            # Initial prompt case
            prompt_text_token = cache_text_token[0].to(device)
            prompt_speech_token = torch.tensor(
                [cache_speech_token[0]], dtype=torch.int32
            ).to(device)
            logging.debug("[generate_long] Using initial prompt (empty cache history)")

        # LLM Inference
        token_list_res = local_llm_forward(
            llm=llm,
            prompt_text_token=prompt_text_token,
            tts_text_token=tts_text_token,
            prompt_speech_token=prompt_speech_token,
            sample_method=sample_method
        )

        output_token_list.extend(token_list_res)

        # Flow Inference
        output, full_mel = local_flow_forward(
            flow=flow,
            token_list=token_list_res,
            prompt_speech_tokens=flow_prompt_token,
            speech_feat=speech_feat,
            embedding=embedding
        )

        # Update Cache
        if cache is not None:
            cache_text.append(tts_text_tn)
            cache_text_token.append(tts_text_token)
            cache_speech_token.append(token_list_res)

        outputs.append(output)
        if full_mel is not None:
            full_mels.append(full_mel)

    tts_speech = torch.concat(outputs, dim=1)
    tts_mel = torch.concat(full_mels, dim=-1) if full_mels else None

    return tts_speech, tts_mel, output_token_list, text_tn_dict


def load_models(
        device: str, 
        ckpt_path: str,
        use_phoneme=False, 
        sample_rate=24000
):
    # Load Speech Tokenizer
    speech_tokenizer_path = os.path.join(ckpt_path, "speech_tokenizer")
    _model, _feature_extractor = yaml_util.load_speech_tokenizer(
        speech_tokenizer_path
    )
    speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)

    # Load Frontends
    frontend, text_frontend = load_frontends(
        ckpt_path=ckpt_path, 
        speech_tokenizer=speech_tokenizer, 
        device=device, 
        sample_rate=sample_rate, 
        use_phoneme=use_phoneme
    )

    llama_path = os.path.join(ckpt_path, "llm")

    llm = GLMTTS(
        llama_cfg_path=os.path.join(llama_path, "config.json"), 
        mode="PRETRAIN",
        lora_adapter_config=None,
        spk_prompt_dict_path=None
    )
    llm.llama = LlamaForCausalLM.from_pretrained(
        llama_path, dtype=torch.float32
    ).to(device) # type: ignore

    llm.llama_embedding = llm.llama.model.embed_tokens # type: ignore

    special_token_ids = get_special_token_ids(frontend.tokenize_fn)
    llm.set_runtime_vars(special_token_ids=special_token_ids)

    flow_ckpt = os.path.join(ckpt_path, "flow", "flow.pt")
    flow_config = os.path.join(ckpt_path, "flow", "config.yaml")
    flow = yaml_util.load_flow_model(
        flow_ckpt, flow_config, device
    )

    token2wav = tts_model_util.Token2Wav(
        flow, 
        sample_rate=sample_rate, 
        device=device, 
        hift_path=os.path.join(ckpt_path, "hift", "hift.pt"),
        vocos_jit_path=os.path.join(ckpt_path, "vocos2d", "generator_jit.ckpt")
    )

    return frontend, text_frontend, speech_tokenizer, llm, token2wav

# ---

MAX_LLM_SEQ_INP_LEN = 750
TOKEN_RATE = 25
EOS_TOKEN_ID_AFTER_MINUS_BOS = None
