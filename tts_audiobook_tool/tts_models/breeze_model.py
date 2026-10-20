import random
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import LogitsProcessorList

from breeze_tts.infer import templates as breeze_templates # type: ignore
from breeze_tts.infer.batch import BatchDepthDecoder # type: ignore
from breeze_tts.infer.runtime import load_runtime, set_all_seeds, update_generation_config_for_breeze # type: ignore
from breeze_tts.infer.templates import get_template, prepare_inputs, select_template_name # type: ignore
from breeze_tts.models.logits_process import GeneratedTokenRepetitionPenaltyLogitsProcessor # type: ignore

from tts_audiobook_tool.app_types import DeviceType, Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants_config import PROJECT_BATCH_SIZE_MAX
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.seed_util import get_random_seed_max
from tts_audiobook_tool.tts_models.breeze_base_model import BreezeBaseModel, BreezeSampling
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *


class BreezeModel(BreezeBaseModel):
    """
    Upstream's batch-API generation path (`prepare_inputs` -> `model.generate(output_audio=True)`),
    SDPA attention, no fast path/compile/streaming runtime. Batched (orchestration `batch_size`).

    Template per voice (upstream `select_template_name`):
        sample + transcript              -> `ref_clone_tata` (pure voice clone)
        sample + transcript + instructions -> `ref_edit_tata` (voice direction, CFG)
        instructions only                -> `tts_instruction` (voice design, CFG)
        neither                          -> `tts_plain` (arbitrary voice per generation)

    CFG (instruction templates only) decodes the depth decoder with upstream's batch-API CUDA graph
    (`BatchDepthDecoder`, as `api.py` `_generate_batch_pcm`); see `_prepare_cfg_depth_graph()`.
    """

    # Upstream's eager CFG depth loop reruns the whole codebook prefix twice per codebook, without a KV cache
    # (about half the speed of non-CFG generation). False only for comparison probes.
    USE_CFG_DEPTH_GRAPH = True

    def __init__(self, device: DeviceType):
        if device != DeviceType.CUDA:
            raise ValueError(f"{self.INFO.ui['proper_name']} requires CUDA")
        self._device_type = device

        model_dir = Path(snapshot_download(BreezeBaseModel.HF_REPO_ID, revision=BreezeBaseModel.HF_REVISION))
        # `load_runtime()` calls `torch.cuda.set_device()`, which requires an index.
        # SDPA attention (backbone, depth decoder and text encoder), not upstream's eager. Must be explicit:
        # the checkpoint config prefers flash_attention_2 (not installed, and no gain expected: decode is dispatch-bound).
        # Validated as superior to eager (3080 Ti, Linux, same seed/texts): ~1.5x faster at both batch 1
        # (0.65x -> 0.99x realtime) and batch 4, same peak VRAM; first-step logits equal to bf16 rounding
        # (same top-5 tokens), and 0% Whisper WER for both at batch 1 and 4 (incl. left-padded rows).
        self._tokenizer, self._model, self._audio_tokenizer = load_runtime(
            model_dir, device="cuda:0", attn_implementation="sdpa"
        )
        # Created on the first CFG generation; holds one captured graph (the current batch-size bucket)
        self._batch_depth_decoder: BatchDepthDecoder | None = None
        # The fp32 codebooks-head weight the captured graph reads (see `_prepare_cfg_depth_graph()`)
        self._graph_head_weight: torch.Tensor | None = None

    def kill(self) -> None:
        self.clear_voice_clone_cache()
        self._batch_depth_decoder = None
        self._graph_head_weight = None
        self._model = None
        self._tokenizer = None
        self._audio_tokenizer = None

    def release_inference_memory(self) -> None:
        """Trim unused PyTorch cache after an idle chat synthesis (weights stay loaded), as GLM."""
        from tts_audiobook_tool.app_support import app_memory
        app_memory.gc_ram_vram()

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool = False,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
            voice_selection_index: int = 0,
            print_params: bool = False,
            max_random_seed: int = -1,
    ) -> list[Sound] | str:

        model_type = TtsModelType.require_by_id("breeze_tts_2_local")
        voice_file_name, transcript = ProjectVoiceUtil.current_voice_reference_pair(
            project, model_type, voice_selection_index
        )
        voice_path = ""
        transcript = transcript.strip()
        if voice_file_name:
            # Upstream requires the reference transcript with the reference audio
            if not transcript:
                return "Voice clone sample transcript is required"
            voice_path = ProjectVoiceUtil.resolve_voice_file_path(project, voice_file_name)
        else:
            transcript = ""
        # Instructions belong to the selected voice, keyed by model id
        voice = ProjectVoiceUtil.current_voice(project, model_type, voice_selection_index)
        instructions = (voice.instructions.get(model_type.id, "") if voice is not None else "").strip()

        sampling = BreezeBaseModel.resolve_sampling(project)

        seed = -1 if force_random_seed else project.get_model_setting("breeze_tts_2_local", "seed")
        if seed == -1:
            seed = random.randrange(0, get_random_seed_max(SEED_MAX - 1, max_random_seed) + 1)

        if print_params:
            params = {
                "temperature": sampling.temperature,
                "top_k": sampling.top_k,
                "top_p": sampling.top_p,
                "repetition_penalty": sampling.repetition_penalty,
                "seed": seed,
            }
            if instructions:
                params["cfg"] = sampling.cfg
            BreezeBaseModel.print_params(params)

        # The prompts arrive as one batch (orchestration `batch_size`, one voice per batch)
        return self.generate(
            prompts, voice_path=voice_path, transcript=transcript, instructions=instructions,
            sampling=sampling, seed=seed,
        )

    def generate(
            self,
            texts: list[str],
            voice_path: str,
            transcript: str,
            instructions: str,
            sampling: BreezeSampling,
            seed: int,
    ) -> list[Sound] | str:
        """
        Generates all texts in one batched `generate()` call (as upstream `api.py` `_generate_batch_pcm`).
        Rows are left-padded by `prepare_inputs()` and each is trimmed at its own EOS.
        With CFG (instructions only), depth decoding replays upstream's batch CUDA graph (`_prepare_cfg_depth_graph()`);
        everything else is eager.
        One seed per call, so a given text's output depends on the batch it's in.
        `voice_path` and `transcript` are both set or both empty.
        """
        if self._model is None:
            return "Logic error - model not initialized"
        if not texts:
            return []

        requests = []
        for index, text in enumerate(texts):
            request = {"id": f"seg-{index}", "text": text, "speaker": BreezeBaseModel.SPEAKER}
            if voice_path:
                request["ref_audio_path"] = voice_path
                request["ref_text"] = transcript
            if instructions:
                request["instruction"] = instructions
            requests.append(request)
        # Only the instruction templates define a CFG negative prompt; upstream raises for CFG != 1 otherwise
        cfg = sampling.cfg if instructions else 1.0

        # Print something here b/c model library code does not
        printt(f"{COL_DIM_ITALICS}Generating...")

        try:
            template = get_template(select_template_name(requests[0]))
            # Same values for the backbone and the depth decoder, as upstream
            update_generation_config_for_breeze(self._model, {
                "do_sample": True,
                "temperature": sampling.temperature,
                "top_k": sampling.top_k,
                "top_p": sampling.top_p,
                "depth_decoder_do_sample": True,
                "depth_decoder_temperature": sampling.temperature,
                "depth_decoder_top_k": sampling.top_k,
                "depth_decoder_top_p": sampling.top_p,
                "max_new_tokens": BreezeBaseModel.MAX_NEW_TOKENS,
            })
            set_all_seeds(seed)
            with self._cached_reference_codes():
                inputs = prepare_inputs(
                    self._tokenizer, self._audio_tokenizer, self._model, requests, template,
                    guidance_scale=cfg, guidance_scale_ref=None, guidance_scale_ins=None,
                )
            prompt_lengths = [int(length) for length in inputs["attention_mask"].sum(dim=1).tolist()]
            logits_processor = LogitsProcessorList(
                [GeneratedTokenRepetitionPenaltyLogitsProcessor(sampling.repetition_penalty)]
            )
            cfg_depth_graph = nullcontext()
            if cfg != 1.0 and self.USE_CFG_DEPTH_GRAPH:
                batch_depth_decoder = self._prepare_cfg_depth_graph(len(texts), sampling)
                # As upstream: graph warmup/capture consumes RNG, so re-seed after it
                set_all_seeds(seed)
                cfg_depth_graph = batch_depth_decoder.bound(self._model)
            with torch.inference_mode(), cfg_depth_graph:
                audio = self._model.generate(
                    **inputs,
                    output_audio=True,
                    audio_tokenizer=self._audio_tokenizer,
                    logits_processor=logits_processor,
                    max_new_tokens=BreezeBaseModel.MAX_NEW_TOKENS,
                )
            if len(audio) != len(texts):
                return f"Batch size mismatch: sent {len(texts)} texts, got {len(audio)} audio segments"
            datas = [segment.detach().float().cpu().numpy().reshape(-1) for segment in audio]
            del audio, inputs
        except Exception as e:
            return make_error_string(e)
        finally:
            # Long segments raise PyTorch's reserved pool well above the weights (eg 7.2 -> 9 GiB for ~18s of audio).
            # Whisper validation (CTranslate2, separate allocator) runs next and can't use those cached blocks,
            # so hand them back; this lets Breeze + Whisper large-v3-turbo share a 12 GB card. Cost is negligible.
            torch.cuda.empty_cache()

        sr = self.INFO.default_output_sample_rate
        for index, (data, prompt_length) in enumerate(zip(datas, prompt_lengths)):
            BreezeModel._warn_if_truncated(index, len(texts), len(data) / sr, prompt_length)

        return [Sound(data, sr) for data in datas]

    def _prepare_cfg_depth_graph(self, batch_size: int, sampling: BreezeSampling) -> BatchDepthDecoder:
        """
        Upstream's batch-API CFG depth decoding (`BatchDepthDecoder`): one captured graph for the current bucket
        (2 x next power of two of the batch size: cond + uncond rows), captured lazily and replaced when the bucket
        changes. No `torch.compile` (graph `fast=False`), so no Triton requirement.

        Graph setup casts the depth decoder's codebooks head to fp32 *in place* (`weight.data = weight.data.float()`).
        Upstream (`api.py`) keeps that cast and swaps checkpoint-dtype copies in around every batch call
        (`_checkpoint_head_weights` / `_swapped_weights`; it also covers `lm_head`, which only its streaming runtime
        casts). Equivalent here, without copies or per-call swaps: keep a reference to the fp32 tensor the graph
        captured (so its memory stays valid for replay) and put the original checkpoint tensor straight back,
        so the eager paths (backbone; non-CFG depth decoding) never see the cast.

        Upstream's `eager_batch_modules()` isn't needed: it only undoes compile/text-encoder-graph state set up by the
        fast streaming runtime, which isn't used here. Sampling values are runtime buffers in the graph, set per call
        (upstream reads them only at capture).
        """
        assert self._model is not None
        if self._batch_depth_decoder is None:
            self._batch_depth_decoder = BatchDepthDecoder(self._model, max_batch_size=PROJECT_BATCH_SIZE_MAX)
        head = self._model.depth_decoder.codebooks_head
        checkpoint_weight = head.weight.data
        # Captures only when the bucket changes
        self._batch_depth_decoder.prepare(batch_size)
        # Compare storage, not tensor identity (`.data` returns a new tensor object on every access). Dropping the
        # captured fp32 tensor here would free the memory the graph replays from.
        if head.weight.data_ptr() != checkpoint_weight.data_ptr():
            self._graph_head_weight = head.weight.data
            head.weight.data = checkpoint_weight
        graph = self._batch_depth_decoder.graph
        # The graph's buffers are inference tensors (allocated during capture)
        with torch.inference_mode():
            graph.set_temperature(sampling.temperature)
            graph.set_top_k(sampling.top_k)
            graph.set_top_p(sampling.top_p)
        return self._batch_depth_decoder

    @contextmanager
    def _cached_reference_codes(self) -> Iterator[None]:
        """
        Reuses reference-clip codes across calls (upstream encodes the clip for every request, and again for the
        CFG negative prompt). Swaps upstream's module-level encode hook only for the duration of `prepare_inputs()`.
        Keyed by the voice-clone cache (path + mtime + size); codes don't depend on the transcript.
        """
        original = breeze_templates._encode_prompt_audio

        def encode(audio_tokenizer, audio_path):
            return self._get_or_create_voice_clone(
                str(audio_path), "", lambda: original(audio_tokenizer, audio_path)
            )

        breeze_templates._encode_prompt_audio = encode
        try:
            yield
        finally:
            breeze_templates._encode_prompt_audio = original

    @staticmethod
    def _warn_if_truncated(index: int, count: int, seconds: float, prompt_length: int) -> None:
        """
        Upstream stops silently at `max_new_tokens`; the 2048-position trained context is not enforced by eager
        `generate()` at all. Warning only: Whisper validation catches cut-off or runaway output.
        """
        frames = round(seconds * BreezeBaseModel.FRAME_RATE)
        row = f" (batch item {index + 1}/{count})" if count > 1 else ""
        if frames >= BreezeBaseModel.MAX_NEW_TOKENS - 1:
            printt(f"{COL_ERROR}Warning: generation hit the {BreezeBaseModel.MAX_NEW_TOKENS}-frame limit "
                   f"({seconds:.0f}s) and may be cut off{row}")
        elif prompt_length + frames > BreezeBaseModel.MAX_POSITIONS:
            printt(f"{COL_ERROR}Warning: prompt + generated audio ({prompt_length + frames} positions) exceeds the "
                   f"model's {BreezeBaseModel.MAX_POSITIONS}-position context; quality may degrade{row}")
