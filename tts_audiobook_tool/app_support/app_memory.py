import gc
import sys

from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_types import Hint

if sys.platform == "win32":
    import win32pdh
else:
    win32pdh = None


def gc_ram_vram() -> None:
    """ Trigger Python garbage collector, plus torch"""
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_system_ram() -> tuple[int, int] | None:
    """
    Returns used and total RAM in bytes
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.used, memory.total
    except:
        return None


def get_nv_vram() -> tuple[int, int] | None:
    """
    Returns VRAM bytes used, bytes total from NVIDIA GPU (*device 0 only*),
    or None if no nvidia gpu.

    Requires nvidia-ml-py (pip install nvidia-ml-py)
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return int(info.used), int(info.total)
    except Exception:
        return None


def get_shared_gpu_memory() -> tuple[str, int] | None:
    """
    Returns Windows shared GPU info in bytes, or None
    (Rem, presence of some shared gpu memory is normal)
    """
    if not win32pdh:
        return

    counter_path = r'\GPU Adapter Memory(*)\Shared Usage'
    query = win32pdh.OpenQuery()
    counter_handle = win32pdh.AddCounter(query, counter_path)

    max_usage_instance = None
    max_usage = 0

    try:
        win32pdh.CollectQueryData(query)
        results: dict[str, float] = win32pdh.GetFormattedCounterArray(counter_handle, win32pdh.PDH_FMT_DOUBLE)

        for instance, value_bytes in results.items():  # type: ignore
            if value_bytes == 0:
                continue
            if value_bytes > max_usage:
                max_usage_instance = instance
                max_usage = value_bytes
    finally:
        win32pdh.CloseQuery(query)

    if not max_usage_instance:
        return None
    return max_usage_instance, int(max_usage)


def get_nv_windows_vram() -> tuple[int, int, int] | None:
    """
    Returns vram used, vram total, and shared gpu memory in bytes.
    Applicable to Windows + Nvidia only.
    """
    result = get_nv_vram()
    if result is None:
        return None
    vram_used, vram_total = result

    result = get_shared_gpu_memory()
    if not result:
        return None
    _, shared = result

    return vram_used, vram_total, shared


def is_vram_spillover_likely(info: tuple[int, int, int] | None) -> bool:
    """
    Is VRAM-to-shared memory "spillover" likely
    Applicable to Windows + Nvidia only.
    """
    if not info:
        return False
    vram_used, vram_total, shared = info
    vram_pct = vram_used / vram_total

    vram_pct_threshold = 0.95
    shared_threshold = 1.0 * (1024**3)

    return vram_pct > vram_pct_threshold and shared > shared_threshold


def show_vram_memory_warning_if_necessary() -> bool:
    info = get_nv_windows_vram()
    if not info:
        return False
    if not is_vram_spillover_likely(info):
        return False
    shared_gb = info[2] / (1024**3)
    message = f"VRAM is full and shared GPU memory is {shared_gb:.1f} GB"
    message += "\nDegraded inference speed is possible/likely"
    hint = Hint("", "Windows VRAM warning", message)
    hints.show_hint(hint)
    return True