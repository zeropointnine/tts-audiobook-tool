import gc
import sys
from typing import cast

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
        Returns VRAM bytes used, bytes total from NVIDIA GPU (device 0), or None if no nvidia gpu
        Requires nvidia-ml-py (pip install nvidia-ml-py)
        """
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
    def get_shared_gpu_memory_gb() -> tuple[str, float] | None:
        """ 
        Returns Windows shared GPU info in GBs, or None
        (Rem, ~1.5 GB or less is oftentimes not a sign of VRAM "overflow")
        
        TODO: Untested
        """

        if not win32pdh:
            return

        # Path for Shared GPU Usage across all adapters
        # The (*) wildcard handles multiple GPUs if they exist
        counter_path = r'\GPU Adapter Memory(*)\Shared Usage'
        
        # Initialize the query
        query = win32pdh.OpenQuery()
        counter_handle = win32pdh.AddCounter(query, counter_path)

        # Return only the value with highest value (may need to revisit)
        max_usage_instance = None
        max_usage = 0
        
        try:
            # Collect data
            win32pdh.CollectQueryData(query)
            
            # Get the formatted values (returns a list of tuples: (instance_name, value))
            _, results = cast(
                tuple[int, list[tuple[str, float]]],
                win32pdh.GetFormattedCounterArray(counter_handle, win32pdh.PDH_FMT_DOUBLE)
            )
            
            for instance, value_bytes in results:

                usage_gb = value_bytes / (1024**3)
                
                # Filter out system instances that aren't actual GPUs (usually luid_0x...)
                # and only check those with actual usage.
                if usage_gb > 0.001:

                    print(f"xxx instance: {instance} | Current Shared: {usage_gb:.2f} GB")

                    if usage_gb > max_usage:
                        max_usage_instance = instance
                        max_usage = usage_gb
                    
        finally:
            win32pdh.CloseQuery(query)

        if not max_usage_instance:
            return None
        return max_usage_instance, max_usage 
