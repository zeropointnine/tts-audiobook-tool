import gc
import sys

from tts_audiobook_tool.hint import Hint

if sys.platform == "win32":
    import win32pdh
else:
    win32pdh = None

class MemoryUtil:
    """
    Memory-related util functions
    """

    @staticmethod
    def gc_ram_vram() -> None:
        """ Trigger Python garbage collector, plus torch"""
        # Garbage collect "RAM"
        gc.collect()
        # "Garbage collect" VRAM
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
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

    @staticmethod
    def get_nv_vram() -> tuple[int, int] | None:
        """
        Returns VRAM bytes used, bytes total from NVIDIA GPU (*device 0 only*), 
        or None if no nvidia gpu.

        Requires nvidia-ml-py (pip install nvidia-ml-py)
        """
        # TODO: Look into gpu-tracker (nv + amd)

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return int(info.used), int(info.total)
        except Exception as e:
            return None

    @staticmethod
    def get_shared_gpu_memory() -> tuple[str, int] | None:
        """ 
        Returns Windows shared GPU info in bytes, or None
        (Rem, presence of some shared gpu memory is normal)
        """

        if not win32pdh:
            return

        # Path for Shared GPU Usage across all adapters
        # The (*) wildcard handles multiple GPUs if they exist
        counter_path = r'\GPU Adapter Memory(*)\Shared Usage'
        
        # Initialize the query
        query = win32pdh.OpenQuery()
        counter_handle = win32pdh.AddCounter(query, counter_path)

        # Will return only the value with the highest value (may need to revisit this)
        max_usage_instance = None
        max_usage = 0
        
        try:
            win32pdh.CollectQueryData(query)
            
            results: dict[str, float] = win32pdh.GetFormattedCounterArray(counter_handle, win32pdh.PDH_FMT_DOUBLE)

            for instance, value_bytes in results.items():
                # Filter out non-GPU entries maybe
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

    @staticmethod
    def get_nv_windows_vram() -> tuple[int, int, int] | None:
        """
        Returns vram used, vram total, and shared gpu memory in bytes.
        Applicable to Windows + Nvidia only.
        """
        result = MemoryUtil.get_nv_vram()
        if result is None:
            return None
        vram_used, vram_total = result

        result = MemoryUtil.get_shared_gpu_memory()
        if not result:
            return None
        _, shared = result

        return vram_used, vram_total, shared
        
    @staticmethod
    def is_vram_spillover_likely(info: tuple[int, int, int] | None) -> bool:
        """
        Is VRAM-to-shared memory "spillover" likely
        Applicable to Windows + Nvidia only.
        """
        if not info:
            return False
        vram_used, vram_total, shared = info
        vram_pct = vram_used / vram_total

        VRAM_PCT_THRESHOLD = 0.95
        SHARED_THRESHOLD = 1.0 * (1024**3) 
        # fyi, a better test wd be checking shared gpu memory "delta" before/after model init/inference/etc

        # Is VRAM is basically full, and is shared gpu memory usage more than "nominal"
        b = vram_pct > VRAM_PCT_THRESHOLD and shared > SHARED_THRESHOLD
        return b

    @staticmethod
    def show_vram_memory_warning_if_necessary() -> bool:
        info = MemoryUtil.get_nv_windows_vram()
        if not info:
            return False
        if not MemoryUtil.is_vram_spillover_likely(info):
            return False
        shared_gb = info[2] / (1024**3)
        message = f"VRAM is full and shared GPU memory is {shared_gb:.1f} GB"
        message += "\nDegraded inference speed is possible/likely"
        hint = Hint("", "Windows VRAM warning", message)
        Hint.show_hint(hint)
        return True
