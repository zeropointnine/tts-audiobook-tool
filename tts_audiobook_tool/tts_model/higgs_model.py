import sys
import langid # type: ignore
import jieba # type: ignore
import os
import re
import copy
from numpy import ndarray
import tqdm
from typing import List
from typing import Optional
from dataclasses import asdict
import torch

from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent # type: ignore

from boson_multimodal.model.higgs_audio import HiggsAudioModel # type: ignore
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator # type: ignore
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer # type: ignore
from boson_multimodal.dataset.chatml_dataset import ( # type: ignore
    ChatMLDatasetSample,
    prepare_chatml_sample,
)
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern # type: ignore
from transformers import AutoConfig, AutoTokenizer # type: ignore
from transformers.cache_utils import StaticCache

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model import HiggsModelProtocol, HiggsProtocol
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *

class HiggsModel(HiggsModelProtocol):
    """
    Pared-down logic from higgs-audio lib script `generation.py`
    """

    def __init__(self, device: str):

        super().__init__(TtsModelInfos.HIGGS.value)

        if device == "cuda":
            device = f"cuda:0"
            use_static_kv_cache = True
        else:
            device = device
            use_static_kv_cache = False

        self.audio_tokenizer = load_higgs_audio_tokenizer(AUDIO_TOKENIZER_PATH, device=device)

        self.model_client = HiggsAudioModelClient(
            model_path=MODEL_PATH,
            audio_tokenizer=self.audio_tokenizer,
            device=device,
            max_new_tokens=MAX_NEW_TOKENS,
            use_static_kv_cache=use_static_kv_cache,
        )

    def kill(self) -> None:
        self.audio_tokenizer = None
        self.model_client.kill() # type: ignore
        self.model_client = None

    def generate_using_project(
            self, 
            project: Project, 
            prompts: list[str], 
            force_random_seed: bool=False
        ) -> list[Sound] | str:
        
        if len(prompts) != 1:
            raise ValueError("Implementation does not support batching")
        prompt = prompts[0]

        if project.higgs_voice_file_name:
            voice_path = os.path.join(project.dir_path, project.higgs_voice_file_name)
            voice_transcript = project.higgs_voice_transcript
        else:
            voice_path = ""
            voice_transcript = ""

        if project.higgs_temperature == -1:
            temperature = HiggsModelProtocol.DEFAULT_TEMPERATURE
        else:
            temperature = project.higgs_temperature

        result = self.generate(
            p_voice_path=voice_path, # TODO is this loading every gen?
            p_voice_transcript=voice_transcript,
            text=prompt,
            seed=random.randint(1, sys.maxsize),
            temperature=temperature
        )

        if isinstance(result, Sound):
            return [result]
        else:
            return result

    def generate(
            self,
            p_voice_path: str,
            p_voice_transcript: str,
            text: str,
            seed: int,
            temperature: float = HiggsProtocol.DEFAULT_TEMPERATURE
    ) -> Sound | str:

        if p_voice_path:
            voice_path = p_voice_path
            voice_transcript = p_voice_transcript
        else:
            voice_path = None
            voice_transcript = None

        scene_prompt = text
        top_k = 50
        top_p = 0.95
        ras_win_len = 7
        ras_win_max_num_repeat = 2
        ref_audio_in_system_message = False
        chunk_method = None
        chunk_max_word_num = 200
        chunk_max_num_turns = 1
        generation_chunk_buffer_size = None

        pattern = re.compile(r"\[(SPEAKER\d+)\]")
        speaker_tags = sorted(set(pattern.findall(text)))

        try:

            messages, audio_ids = prepare_generation_context(
                scene_prompt=scene_prompt,
                voice_path=voice_path,
                voice_transcript=voice_transcript,
                ref_audio_in_system_message=ref_audio_in_system_message,
                audio_tokenizer=self.audio_tokenizer,
                speaker_tags=speaker_tags,
            )

            chunked_text = prepare_chunk_text(
                text,
                chunk_method=chunk_method,
                chunk_max_word_num=chunk_max_word_num,
                chunk_max_num_turns=chunk_max_num_turns,
            )

            # text_output is the input prompt (not sure if transformed in any way or not)
            assert(self.model_client is not None)
            sound_data, sr, text_output = self.model_client.generate(
                messages=messages,
                audio_ids=audio_ids,
                chunked_text=chunked_text,
                generation_chunk_buffer_size=generation_chunk_buffer_size,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed,
            )
            assert( isinstance(sound_data, ndarray) )
            sound_data = sound_data.squeeze() # for good measure
            return Sound(sound_data, sr )

        except Exception as e:
            return make_error_string(e)


# --------------------------------------------------------------------------------------------------

def prepare_chunk_text(
    text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1
):
    """Chunk the text into smaller pieces. We will later feed the chunks one by one to the model.

    Parameters
    ----------
    text : str
        The text to be chunked.
    chunk_method : str, optional
        The method to use for chunking. Options are "speaker", "word", or None. By default, we won't use any chunking and
        will feed the whole text to the model.
    replace_speaker_tag_with_special_tags : bool, optional
        Whether to replace speaker tags with special tokens, by default False
        If the flag is set to True, we will replace [SPEAKER0] with <|speaker_id_start|>SPEAKER0<|speaker_id_end|>
    chunk_max_word_num : int, optional
        The maximum number of words for each chunk when "word" chunking method is used, by default 100
    chunk_max_num_turns : int, optional
        The maximum number of turns for each chunk when "speaker" chunking method is used,

    Returns
    -------
    List[str]
        The list of text chunks.

    """
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        # TODO: We may improve the logic in the future
        # For long-form generation, we will first divide the text into multiple paragraphs by splitting with "\n\n"
        # After that, we will chunk each paragraph based on word count
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            if language == "zh":
                # For Chinese, we will chunk based on character count
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def _build_system_message_with_audio_prompt(system_message):
    contents = []

    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN) :]

    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    ret = Message(
        role="system",
        content=contents,
    )
    return ret


class HiggsAudioModelClient:

    def kill(self) -> None:
        # Clear all member variables in attempt to clear all resources
        self._audio_tokenizer = None
        self._model = None
        self._collator = None
        self._kv_caches = None

    def __init__(
        self,
        model_path,
        audio_tokenizer,
        device,
        max_new_tokens=2048,
        kv_cache_lengths: List[int] = [1024, 4096, 8192],  # Multiple KV cache sizes,
        use_static_kv_cache=False,
    ):

        self._device = device # fyi, _device can also use format "cuda:[integer]"

        self._audio_tokenizer = (
            load_higgs_audio_tokenizer(audio_tokenizer, device=self._device)
            if isinstance(audio_tokenizer, str)
            else audio_tokenizer
        )
        self._model = HiggsAudioModel.from_pretrained(
            model_path,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()
        self._kv_cache_lengths = kv_cache_lengths
        self._use_static_kv_cache = use_static_kv_cache

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
        self._max_new_tokens = max_new_tokens
        self._collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self._config.audio_in_token_idx,
            audio_out_token_id=self._config.audio_out_token_idx,
            audio_stream_bos_id=self._config.audio_stream_bos_id,
            audio_stream_eos_id=self._config.audio_stream_eos_id,
            encode_whisper_embed=self._config.encode_whisper_embed,
            pad_token_id=self._config.pad_token_id,
            return_audio_in_tokens=self._config.encode_audio_in_tokens,
            use_delay_pattern=self._config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self._config.audio_num_codebooks,
        )
        self.kv_caches = None
        if use_static_kv_cache:
            self._init_static_kv_cache()

    def _init_static_kv_cache(self):
        assert(self._model is not None)
        cache_config = copy.deepcopy(self._model.config.text_config)
        cache_config.num_hidden_layers = self._model.config.text_config.num_hidden_layers
        if self._model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self._model.config.audio_dual_ffn_layers)
        # A list of KV caches for different lengths
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self._model.device,
                dtype=self._model.dtype,
            )
            for length in sorted(self._kv_cache_lengths)
        }
        # Capture CUDA graphs for each KV cache length
        if "cuda" in self._device:
            self._model.capture_model(self.kv_caches.values())

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values(): # type: ignore
            kv_cache.reset()

    @torch.inference_mode()
    def generate(
        self,
        messages,
        audio_ids,
        chunked_text,
        generation_chunk_buffer_size,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        seed=123,
        *args,
        **kwargs,
    ):
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None
        sr = 24000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []
        for idx, chunk_text in tqdm.tqdm(
            enumerate(chunked_text), desc="Generating audio chunks", total=len(chunked_text)
        ):
            generation_messages.append(
                Message(
                    role="user",
                    content=chunk_text,
                )
            )
            chatml_sample = ChatMLSample(messages=messages + generation_messages)
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self._tokenizer)
            postfix = self._tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
            input_tokens.extend(postfix)

            # logger.info(f"========= Chunk {idx} Input =========")
            # logger.info(self._tokenizer.decode(input_tokens))
            context_audio_ids = audio_ids + generated_audio_ids

            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1)
                if context_audio_ids
                else None,
                audio_ids_start=torch.cumsum(
                    torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0
                )
                if context_audio_ids
                else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            batch_data = self._collator([curr_sample]) # type: ignore
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)

            if self._use_static_kv_cache:
                self._prepare_kv_caches()

            # Generate audio
            outputs = self._model.generate( # type: ignore
                **batch,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self._tokenizer,
                seed=seed,
            )

            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self._config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                step_audio_out_ids_l.append(audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1]) # type: ignore
            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)

            generation_messages.append(
                Message(
                    role="assistant",
                    content=AudioContent(audio_url=""),
                )
            )
            if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
                generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
                generation_messages = generation_messages[(-2 * generation_chunk_buffer_size) :]

        # logger.info(f"========= Final Text output =========")
        # logger.info(self._tokenizer.decode(outputs[0][0]))
        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)
        concat_wv = self._audio_tokenizer.decode(concat_audio_out_ids.unsqueeze(0))[0, 0] # type: ignore
        text_result = self._tokenizer.decode(outputs[0][0]) # type: ignore
        return concat_wv, sr, text_result


def prepare_generation_context(scene_prompt, voice_path, voice_transcript, ref_audio_in_system_message, audio_tokenizer, speaker_tags):
    """
    Prepare the context for generation.
    The context contains the system message, user message, assistant message, and audio prompt if any.
    """
    system_message = None
    messages = []
    audio_ids = []
    if voice_path is not None:
        num_speakers = 1
        if ref_audio_in_system_message:
            speaker_desc = [f"SPEAKER0: {AUDIO_PLACEHOLDER_TOKEN}"]
            if scene_prompt:
                system_message = (
                    "Generate audio following instruction."
                    "\n\n"
                    f"<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>"
                )
            else:
                system_message = (
                    "Generate audio following instruction.\n\n"
                    + f"<|scene_desc_start|>\n"
                    + "\n".join(speaker_desc)
                    + "\n<|scene_desc_end|>"
                )
            system_message = _build_system_message_with_audio_prompt(system_message)
        else:
            if scene_prompt:
                system_message = Message(
                    role="system",
                    content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>",
                )

        assert os.path.exists(voice_path), f"Voice prompt audio file {voice_path} does not exist."

        prompt_text = voice_transcript
        if voice_transcript and os.path.exists(voice_transcript):
            with open(voice_transcript, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()

        audio_tokens = audio_tokenizer.encode(voice_path)
        audio_ids.append(audio_tokens)

        if not ref_audio_in_system_message:
            messages.append(
                Message(
                    role="user",
                    content=prompt_text,
                )
            )
            messages.append(
                Message(
                    role="assistant",
                    content=AudioContent(
                        audio_url=voice_path,
                    ),
                )
            )
    else:
        if len(speaker_tags) > 1:
            # By default, we just alternate between male and female voices
            speaker_desc_l = []

            for idx, tag in enumerate(speaker_tags):
                if idx % 2 == 0:
                    speaker_desc = f"feminine"
                else:
                    speaker_desc = f"masculine"
                speaker_desc_l.append(f"{tag}: {speaker_desc}")

            speaker_desc = "\n".join(speaker_desc_l)
            scene_desc_l = []
            if scene_prompt:
                scene_desc_l.append(scene_prompt)
            scene_desc_l.append(speaker_desc)
            scene_desc = "\n\n".join(scene_desc_l)

            system_message = Message(
                role="system",
                content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
            )
        else:
            system_message_l = ["Generate audio following instruction."]
            if scene_prompt:
                system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
            system_message = Message(
                role="system",
                content="\n\n".join(system_message_l),
            )
    if system_message:
        messages.insert(0, system_message)
    return messages, audio_ids


# ---

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
MAX_NEW_TOKENS = 2048

AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


"""
@click.command()
@click.option(
    "--model_path",
    type=str,
    default="bosonai/higgs-audio-v2-generation-3B-base",
    help="Output wav file path.",
)
@click.option(
    "--audio_tokenizer",
    type=str,
    default="bosonai/higgs-audio-v2-tokenizer",
    help="Audio tokenizer path, if not set, use the default one.",
)
@click.option(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="The maximum number of new tokens to generate.",
)
@click.option(
    "--transcript",
    type=str,
    default="transcript/single_speaker/en_dl.txt",
    help="The prompt to use for generation. If not set, we will use a default prompt.",
)
@click.option(
    "--scene_prompt",
    type=str,
    default=f"{CURR_DIR}/scene_prompts/quiet_indoor.txt",
    help="The scene description prompt to use for generation. If not set, or set to `empty`, we will leave it to empty.",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="The value used to module the next token probabilities.",
)
@click.option(
    "--top_k",
    type=int,
    default=50,
    help="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
)
@click.option(
    "--top_p",
    type=float,
    default=0.95,
    help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
)
@click.option(
    "--ras_win_len",
    type=int,
    default=7,
    help="The window length for RAS sampling. If set to 0 or a negative value, we won't use RAS sampling.",
)
@click.option(
    "--ras_win_max_num_repeat",
    type=int,
    default=2,
    help="The maximum number of times to repeat the RAS window. Only used when --ras_win_len is set.",
)
@click.option(
    "--voice_path",
    type=str,
    default=None,
    help="Path to the voice prompt wav file to use for generation.",
)
@click.option(
    "--voice_transcript",
    type=str,
    default=None,
    help="The transcript of the voice prompt.",
)
@click.option(
    "--ref_audio_in_system_message",
    is_flag=True,
    default=False,
    help="Whether to include the voice prompt description in the system message.",
    show_default=True,
)
@click.option(
    "--chunk_method",
    default=None,
    type=click.Choice([None, "speaker", "word"]),
    help="The method to use for chunking the prompt text. Options are 'speaker', 'word', or None. By default, we won't use any chunking and will feed the whole text to the model.",
)
@click.option(
    "--chunk_max_word_num",
    default=200,
    type=int,
    help="The maximum number of words for each chunk when 'word' chunking method is used. Only used when --chunk_method is set to 'word'.",
)
@click.option(
    "--chunk_max_num_turns",
    default=1,
    type=int,
    help="The maximum number of turns for each chunk when 'speaker' chunking method is used. Only used when --chunk_method is set to 'speaker'.",
)
@click.option(
    "--generation_chunk_buffer_size",
    default=None,
    type=int,
    help="The maximal number of chunks to keep in the buffer. We will always keep the reference audios, and keep `max_chunk_buffer` chunks of generated audio.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for generation.",
)
@click.option(
    "--device_id",
    type=int,
    default=None,
    help="The device to run the model on.",
)
@click.option(
    "--out_path",
    type=str,
    default="generation.wav",
)
@click.option(
    "--use_static_kv_cache",
    type=int,
    default=1,
    help="Whether to use static KV cache for faster generation. Only works when using GPU.",
)
"""
