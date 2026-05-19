from __future__ import annotations

"""
Application-level support helpers that span multiple runtime concerns.

This module is the shared landing place for cross-cutting app behaviors that
do not fit cleanly into the narrower path, hashing, or display helper modules.

It intentionally keeps these convenience functions together at the package root
while the more domain-specific support code lives in dedicated submodules.
"""

import logging
import os
import random
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import ModelWarmUpResult, SttVariant
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.l import L
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

if TYPE_CHECKING:
    from tts_audiobook_tool.prefs import Prefs

is_logging_initialized = False


def init_logging() -> None:
    global is_logging_initialized

    if is_logging_initialized:
        return
    is_logging_initialized = True

    L.init(APP_NAME)
    L.i("START " + "-" * 60)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)

    logging.getLogger("pocket_tts.models.tts_model").setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore", module="pyloud")


def set_seed(seed: int) -> None:
    """
    Sets random seed across torch, numpy, and random.

    This is oftentimes sufficient to make the inference of torch-based TTS models idempotent.
    (Must be tested carefully to verify, ofc)
    """
    import numpy as np
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def show_pre_inference_hints(prefs: Prefs, p_project) -> None:
    """ Shows one-time hints or warnings related to doing inference """
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.tts import Tts

    project: Project = p_project

    if Tts.get_type() == TtsModelInfos.FISH_S1 and project.fish_s1_compile_enabled:
        hints.show_hint_if_necessary(prefs, HINT_FISH_S1_FIRST_COMPILE)
    elif Tts.get_type() == TtsModelInfos.FISH_S2 and project.fish_s2_compile_enabled:
        hints.show_hint_if_necessary(prefs, HINT_FISH_S2_FIRST_COMPILE)

    import torch
    if platform.system() == "Linux" and torch.cuda.is_available():
        if prefs.stt_variant != SttVariant.DISABLED and prefs.stt_config.device == "cuda":
            version = torch.backends.cudnn.version()
            if version and version > CTRANSLATE_REQUIRED_CUDNN_VERSION:
                hints.show_hint(HINT_LINUX_CUDNN_VERSION, and_prompt=True)


def play_done_sound() -> None:
    from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil

    done_wav_path = make_assets_file_path("done.wav")
    SoundFileUtil.play_sound_file_async(done_wav_path)


def make_memory_string(base_color=COL_DIM) -> str:
    from tts_audiobook_tool.memory_util import MemoryUtil

    result = MemoryUtil.get_nv_vram()
    if result is None:
        vram_string = ""
    else:
        used, total = result
        vram_string = f"{base_color}VRAM: {COL_ACCENT}{make_gb_string(used)}{base_color}/{make_gb_string(total)}"

    shared_result = MemoryUtil.get_shared_gpu_memory()
    if shared_result is None:
        shared_string = ""
    else:
        _, shared = shared_result
        shared_string = f"{base_color}shared VRAM: {COL_ACCENT}{make_gb_string(shared)}"

    result = MemoryUtil.get_system_ram()
    if result is None:
        ram_string = ""
    else:
        used, total = result
        ram_string = f"{base_color}RAM: {COL_ACCENT}{make_gb_string(used)}{base_color}/{make_gb_string(total)}"

    parts = [s for s in [vram_string, shared_string, ram_string] if s]
    if not parts:
        return ""
    return f"{base_color}, ".join(parts)


def log_unload_memory_snapshot(label: str) -> None:
    parts: list[str] = []

    import torch

    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            max_reserved = torch.cuda.max_memory_reserved()
            parts.extend([
                f"torch_allocated={make_gb_string(allocated)}",
                f"torch_reserved={make_gb_string(reserved)}",
                f"torch_max_allocated={make_gb_string(max_allocated)}",
                f"torch_max_reserved={make_gb_string(max_reserved)}",
            ])

            stats = torch.cuda.memory_stats()

            def add_stat_bytes(label_text: str, stat_key: str) -> None:
                value = stats.get(stat_key)
                if value is not None:
                    parts.append(f"{label_text}={make_gb_string(int(value))}")

            def add_stat_int(label_text: str, stat_key: str) -> None:
                value = stats.get(stat_key)
                if value is not None:
                    parts.append(f"{label_text}={int(value)}")

            add_stat_bytes("active_bytes", "active_bytes.all.current")
            add_stat_bytes("inactive_split_bytes", "inactive_split_bytes.all.current")
            add_stat_bytes("requested_bytes", "requested_bytes.all.current")
            add_stat_bytes("allocated_bytes", "allocated_bytes.all.current")
            add_stat_bytes("reserved_bytes", "reserved_bytes.all.current")
            add_stat_int("active_blocks", "active.all.current")
            add_stat_int("segment_count", "segment.all.current")
        except Exception as e:
            parts.append(f"torch_cuda_error={e}")
    else:
        parts.append("torch_cuda_unavailable")

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pid = os.getpid()
        process_bytes = None

        try:
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if proc.pid == pid:
                    process_bytes = int(proc.usedGpuMemory)
                    break
        except Exception as e:
            parts.append(f"nvml_compute_proc_error={e}")

        try:
            for proc in pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle):
                if proc.pid == pid:
                    used = int(proc.usedGpuMemory)
                    process_bytes = used if process_bytes is None else process_bytes + used
                    break
        except Exception as e:
            parts.append(f"nvml_graphics_proc_error={e}")

        if process_bytes is None:
            parts.append(f"pid_gpu_memory=none pid={pid}")
        else:
            parts.append(f"pid_gpu_memory={make_gb_string(process_bytes)} pid={pid}")

        pynvml.nvmlShutdown()
    except Exception as e:
        parts.append(f"nvml_error={e}")

    L.i(f"Unload memory snapshot [{label}] | " + " | ".join(parts))


def get_chromium_info() -> tuple[str, str] | None:
    from tts_audiobook_tool.system_support.browser import get_chromium_info as browser_get_chromium_info

    return browser_get_chromium_info()


def print_warm_up_result_stop(result: ModelWarmUpResult) -> None:
    """ Prints user-facing feedback for a failed warm-up result. """
    if result.did_interrupt:
        print_feedback("\nCancelled")
    elif result.error:
        print_feedback(f"\n{COL_ERROR}Model initialization failed:\n\n{result.error}")