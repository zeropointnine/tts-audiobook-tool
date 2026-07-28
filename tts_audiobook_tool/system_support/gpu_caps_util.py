class GpuCapsUtil:
    """Capability checks for optional accelerator runtimes."""

    @staticmethod
    def has_torch_cuda_api_device() -> bool:
        """Return whether Torch exposes a device through its CUDA-compatible API."""
        try:
            import torch
        except (ImportError, OSError):
            return False

        return bool(torch.cuda.is_available())

    @staticmethod
    def is_torch_nvidia_cuda() -> bool:
        """Return whether Torch exposes an NVIDIA CUDA device."""
        try:
            import torch
            from torch import version as torch_version
        except (ImportError, OSError):
            return False

        return bool(
            torch.cuda.is_available()
            and getattr(torch_version, "cuda", None) is not None
            and getattr(torch_version, "hip", None) is None
        )

    @staticmethod
    def is_torch_rocm() -> bool:
        """Return whether Torch exposes an AMD ROCm device."""
        try:
            import torch
            from torch import version as torch_version
        except (ImportError, OSError):
            return False

        return bool(
            torch.cuda.is_available()
            and getattr(torch_version, "hip", None) is not None
        )

    @staticmethod
    def has_ctranslate2_float16_gpu() -> bool:
        """Return whether CTranslate2 exposes a GPU with float16 support."""
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() < 1:
                return False
            compute_types = ctranslate2.get_supported_compute_types("cuda", 0)
            return "float16" in compute_types
        except (ImportError, OSError, RuntimeError):
            return False

    @staticmethod
    def cudnn_version() -> int | None:
        """Return the NVIDIA cuDNN version, or None when it is not applicable."""
        if not GpuCapsUtil.is_torch_nvidia_cuda():
            return None

        try:
            import torch

            version = torch.backends.cudnn.version()
            return version if isinstance(version, int) else None
        except (ImportError, OSError, RuntimeError):
            return None
