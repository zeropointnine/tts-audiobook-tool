"""
Trace CUDA memory for any local TTS model across three voices and three
round-robin cycles.

No command-line arguments: edit the constants below.

The model is not selected explicitly: the script probes the venv it is
run in (the same Tts.init_local_model_type() the app uses at startup) and
traces whichever local model library that venv provides. The voices are
set by VOICES.

Run it from the appropriate venv:

    source venv-q/bin/activate
    python -m testx.multi_voice_cache_vram

In a venv without a local model library (e.g. venv-base) or in an
SGL-Omni-mode venv, it exits with an explanation instead.

Every model loads its built-in default: Qwen3 loads the default Base
repo, OmniVoice loads k2-fsa/OmniVoice, MOSS loads its default repo, and
the remaining models load pinned ones. There is no option to point the
script at a different model.

The transcript element of VOICES is only meaningful for the models that require it.

The default order is A, B, C, repeated for CYCLES cycles, with BATCH_SIZE
identical prompts per call. Use equally sized reference samples and the
same prompt to make the per-voice numbers directly comparable.

Important columns:

- alloc_after: live torch CUDA tensors after the generation call returned.
- peak_alloc: highest live torch CUDA allocation during that call.
- reserved_after: CUDA memory retained by PyTorch's caching allocator.
- peak_reserved: highest reserved value during that call.
- nv_used: device-wide usage reported by NVML, including allocator
  reservations and memory belonging to other processes.
- prepared: yes only when the voice-clone factory ran for that voice, i.e.
  a miss in the base-class voice clone cache.

For a multi-voice cache (SUPPORTS_MULTIPLE_VOICE_CLONES) `prepared` should
be yes for the first A/B/C calls and no for the remaining calls, and the
cache should end with len(VOICES) entries. For the single-slot cache
(Pocket) each voice switch evicts the previous clone, so every call
prepares and the cache ends with 1 entry. In both cases `alloc_after`
should not grow by one voice prompt on each switch; `reserved_after` may
remain high or ratchet because PyTorch retains reusable allocator blocks.
"""

import os
import sys
import time

import torch

from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


GB = 1024 ** 3

# --- Configuration (edit before running) -------------------------------

# Project dir that the VOICES file names resolve against.
# "" = current working directory; absolute paths in VOICES override it.
VOICE_DIR: str = ""

# (label, voice file name/path, transcript)
# The transcript is only used by the models that take one (see the module
# docstring); leave it "" for the ones that ignore it.
VOICES: list[tuple[str, str, str]] = [
    ("A", "/d/w/voice/amanda lee - jingliu/amanda lee jingliu 1.wav", "Transcript for voice A"),
    ("B", "/d/w/voice/amanda lee - jingliu/amanda lee jingliu 2.wav", "Transcript for voice B"),
    ("C", "/d/w/voice/amanda lee - jingliu/amanda lee jingliu 3.wav", "Transcript for voice C"),
]

# Identical prompts per generation call, and the number of complete
# A -> B -> C round-robin cycles to run
BATCH_SIZE = 1 # rem, is not necessarily important to use batching for the objectives of this test script
CYCLES = 3
PROMPT = (
    "The lantern burned steadily beside the window while the rain moved "
    "quietly across the garden."
)


def nvml_used_bytes() -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def gb(value: int) -> float:
    return value / GB


def cache_entry_count(model: TtsBaseModel) -> int:
    return len(getattr(model, "_voice_clone_cache", {}))


def validate_voice_files(model_type: TtsModelType, voices: list[tuple[str, str, str]]) -> None:
    if model_type == TtsModelType.POCKET:
        # A Pocket voice reference may be a bare predefined-voice name
        # rather than a file, so no file is required
        return
    for label, path, _ in voices:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Voice {label} not found: {path}")


def print_voice_inventory(voices: list[tuple[str, str, str]]) -> None:
    print("Voice references:")
    for label, path, transcript in voices:
        size_text = (
            f"{os.path.getsize(path) / (1024 ** 2):.2f} MB"
            if os.path.isfile(path)
            else "n/a"
        )
        print(
            f"  {label}: {path} | file={size_text} | transcript_chars={len(transcript)}"
        )


def create_model(model_type: TtsModelType) -> tuple[TtsBaseModel, str | None]:
    """
    Instantiates the model class found in the current venv the same way
    Tts's lazy factory does (tts_audiobook_tool/tts.py), standalone: the
    Tts singleton never instantiates the model, and only the selected
    model's heavy module gets imported. Every model loads its built-in
    default (see the module docstring).

    Returns (model, target); target is the repo/dir actually loaded (for
    logging), or None for models with a built-in model.
    """
    match model_type:

        case TtsModelType.QWEN3TTS:
            from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
            from tts_audiobook_tool.tts_models.qwen3_model import Qwen3Model
            # The voice clone flow needs the 'base' variant; that is
            # determined by the model config, so pick a Base repo
            target = Qwen3BaseModel.DEFAULT_REPO_ID
            return Qwen3Model(target, DeviceType.CUDA), target

        case TtsModelType.OMNIVOICE:
            from tts_audiobook_tool.tts_models.omnivoice_base_model import OmniVoiceBaseModel
            from tts_audiobook_tool.tts_models.omnivoice_model import OmniVoiceModel
            target = OmniVoiceBaseModel.DEFAULT_REPO_ID
            return OmniVoiceModel(model_target=target, device=DeviceType.CUDA), target

        case TtsModelType.MOSS:
            from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs
            from tts_audiobook_tool.tts_models.moss_model import MossModel
            target = MossConfigs.get_default_repo_id()
            return MossModel(device=DeviceType.CUDA, model_target=target), target

        case TtsModelType.CHATTERBOX:
            from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
            from tts_audiobook_tool.tts_models.chatterbox_model import ChatterboxModel
            # Project default: the first enum member (multilingual)
            return ChatterboxModel(ChatterboxType.MULTILINGUAL, DeviceType.CUDA), None

        case TtsModelType.FISH_S1:
            from tts_audiobook_tool.tts_models.fish_s1_base_model import FishS1BaseModel
            from tts_audiobook_tool.tts_models.fish_s1_model import FishS1Model
            # Loads the library's built-in model
            return FishS1Model(DeviceType.CUDA, FishS1BaseModel.DEFAULT_COMPILE_ENABLED), None

        case TtsModelType.FISH_S2:
            from tts_audiobook_tool.tts_models.fish_s2_base_model import FishS2BaseModel
            from tts_audiobook_tool.tts_models.fish_s2_model import FishS2Model
            return FishS2Model(DeviceType.CUDA, FishS2BaseModel.DEFAULT_COMPILE_ENABLED), None

        case TtsModelType.GLM:
            from tts_audiobook_tool.tts_models.glm_base_model import GlmBaseModel
            from tts_audiobook_tool.tts_models.glm_model import GlmModel
            # CUDA-only model; downloads zai-org/GLM-TTS itself.
            # 24000 = the app default (GlmBaseModel.SAMPLE_RATES[0])
            return GlmModel(DeviceType.CUDA, GlmBaseModel.SAMPLE_RATES[0]), None

        case TtsModelType.HIGGS_V2:
            from tts_audiobook_tool.tts_models.higgs_v2_model import HiggsV2Model
            # Loads a pinned local model file
            return HiggsV2Model(DeviceType.CUDA), None

        case TtsModelType.MIRA:
            from tts_audiobook_tool.tts_models.mira_model import MiraModel
            # No constructor parameters; CUDA-only under the hood
            return MiraModel(), None

        case TtsModelType.POCKET:
            from tts_audiobook_tool.tts_models.pocket_model import PocketModel
            # language="" loads the language-agnostic default model
            return PocketModel(device=DeviceType.CUDA, language=""), None

        case _:
            raise SystemExit(
                f"The local model detected in this venv ({model_type.value.id}) "
                "is not supported by this script: IndexTTS2/Oute/VibeVoice "
                "are not part of the voice clone cache work"
            )


def make_project(model_type: TtsModelType, voices: list[tuple[str, str, str]]):
    """
    Builds the Project the trace generates against: the voice lists land
    on the spec's own attribute names, so per-voice selection in
    generate_using_project(voice_selection_index=...) picks A/B/C in turn,
    exactly as the app's round-robin does. Requires the model type to have
    been initialized first (main() calls Tts.init_local_model_type()),
    since Project's load-time normalization consults it.
    """
    from tts_audiobook_tool.project import Project

    spec = model_type.value
    project = Project.model_validate({"dir_path": VOICE_DIR})

    if spec.voice_target_attr:
        setattr(project, spec.voice_target_attr, [path for _, path, _ in voices])
    if spec.voice_transcript_attr:
        setattr(project, spec.voice_transcript_attr, [transcript for _, _, transcript in voices])

    # MOSS's generate_using_project reads project.moss_target to pick the
    # LOCAL vs DELAY hyperparams; it must agree with what got loaded
    if model_type == TtsModelType.MOSS:
        from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs
        project.moss_target = MossConfigs.get_default_repo_id()

    return project


def install_preparation_timer(model: TtsBaseModel) -> list[float]:
    """
    Times the base-class voice clone factory on each call. Every model in
    scope routes its voice preparation through
    _get_or_create_voice_clone, so a timed factory observes a cache miss
    for any of them.
    """
    preparation_times: list[float] = []
    original_get_or_create = model._get_or_create_voice_clone

    def timed_get_or_create(source_path: str, transcript: str, factory):
        def timed_factory():
            started = time.perf_counter()
            value = factory()
            preparation_times.append(time.perf_counter() - started)
            return value

        return original_get_or_create(source_path, transcript, timed_factory)

    model._get_or_create_voice_clone = timed_get_or_create
    return preparation_times


def main() -> int:
    if BATCH_SIZE < 1:
        print("BATCH_SIZE must be at least 1")
        return 1
    if CYCLES < 1:
        print("CYCLES must be at least 1")
        return 1

    # Probe the venv the same way the app does at startup; this script
    # traces whichever local model library that venv provides
    model_type, num_matches = Tts.init_local_model_type()
    if model_type == TtsModelType.NONE:
        if Tts.get_backend_mode() == TtsBackendKind.SGL_OMNI:
            print(
                "This venv is in SGL-Omni mode (TTS served by an external "
                "server); this script traces local inference, so it cannot "
                "run here"
            )
        else:
            print(
                "No local TTS model found in this venv; run it from the "
                "venv of the model you want to trace"
            )
        return 1
    if num_matches > 1:
        print(
            f"Note: {num_matches} local model libraries found in this "
            f"venv; tracing {model_type.value.id} (the first match)"
        )

    if not torch.cuda.is_available():
        print("CUDA is not available in this environment")
        return 1

    voices = list(VOICES)
    try:
        validate_voice_files(model_type, voices)
    except FileNotFoundError as error:
        print(error)
        return 1

    print_voice_inventory(voices)

    print(f"\nLoading {model_type.value.ui['proper_name']} (detected from venv) ...")
    load_start = time.perf_counter()
    model, target = create_model(model_type)
    torch.cuda.synchronize()
    if target:
        print(f"  target: {target}")
    print(
        f"Loaded in {time.perf_counter() - load_start:.1f}s | "
        f"allocated={gb(torch.cuda.memory_allocated()):.2f} GB | "
        f"reserved={gb(torch.cuda.memory_reserved()):.2f} GB"
    )

    project = make_project(model_type, voices)
    preparation_times = install_preparation_timer(model)

    prompts = [PROMPT for _ in range(BATCH_SIZE)]
    total_calls = CYCLES * len(voices)

    print(
        "\n"
        f"{'step':>4} {'cycle':>5} {'voice':>5} {'prepared':>8} {'cache':>5} "
        f"{'alloc_before':>12} {'peak_alloc':>10} {'alloc_after':>11} "
        f"{'resv_after':>10} {'peak_resv':>10} {'nv_used':>9} {'seconds':>8}"
    )

    for step in range(total_calls):
        cycle = step // len(voices) + 1
        label, _voice_path, _transcript = voices[step % len(voices)]

        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        preparations_before = len(preparation_times)
        torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        result = model.generate_using_project(
            project,
            prompts,
            voice_selection_index=step % len(voices),
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        allocated_after = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        nv_used = nvml_used_bytes()
        was_prepared = len(preparation_times) > preparations_before
        nv_used_text = f"{gb(nv_used):.2f}" if nv_used is not None else "n/a"

        print(
            f"{step + 1:4d} {cycle:5d} {label:>5} "
            f"{('yes' if was_prepared else 'no'):>8} {cache_entry_count(model):5d} "
            f"{gb(allocated_before):12.2f} {gb(peak_allocated):10.2f} "
            f"{gb(allocated_after):11.2f} {gb(reserved_after):10.2f} "
            f"{gb(peak_reserved):10.2f} {nv_used_text:>9} {elapsed:8.1f}"
        )

        if isinstance(result, str):
            print(f"Generation failed: {result}")
            if "out of memory" in result.lower() or "outofmemory" in result.lower():
                print("OOM detected; stopping the trace")
            return 1

        del result

    preparation_count = len(preparation_times)
    cache_count = cache_entry_count(model)
    multi_voice = type(model).SUPPORTS_MULTIPLE_VOICE_CLONES
    expected_preparations = len(voices) if multi_voice else total_calls
    expected_cache = len(voices) if multi_voice else 1

    print("\nSummary:")
    print(f"  model: {model_type.value.id} ({model_type.value.ui['proper_name']})")
    if target:
        print(f"  target: {target}")
    print(f"  multi-voice cache: {multi_voice}")
    print(f"  generation calls: {total_calls}")
    print(f"  cache entries: {cache_count} (expected {expected_cache})")
    print(f"  voice preparation calls: {preparation_count} (expected {expected_preparations})")
    if preparation_times:
        print(f"  average voice preparation: {sum(preparation_times) / len(preparation_times):.2f}s")
    print(f"  final allocated: {gb(torch.cuda.memory_allocated()):.2f} GB")
    print(f"  final reserved: {gb(torch.cuda.memory_reserved()):.2f} GB")

    if preparation_count == expected_preparations and cache_count == expected_cache:
        print("  cache behavior: PASS")
        return 0

    print(
        f"  cache behavior: WARNING - expected {expected_preparations} "
        f"preparations and {expected_cache} cache entries"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
