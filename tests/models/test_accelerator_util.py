import sys
from types import SimpleNamespace

from tts_audiobook_tool.app_types import SttConfig
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.system_support.gpu_caps_util import GpuCapsUtil


def make_torch(*, available: bool, cuda_version=None, hip_version=None, cudnn_version=None):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: available),
        version=SimpleNamespace(cuda=cuda_version, hip=hip_version),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: cudnn_version),
        ),
    )


def test_detects_nvidia_cuda_and_cudnn(monkeypatch):
    torch = make_torch(
        available=True,
        cuda_version="12.8",
        cudnn_version=91003,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert GpuCapsUtil.has_torch_cuda_api_device()
    assert GpuCapsUtil.is_torch_nvidia_cuda()
    assert not GpuCapsUtil.is_torch_rocm()
    assert GpuCapsUtil.cudnn_version() == 91003


def test_detects_rocm_without_treating_it_as_nvidia_cudnn(monkeypatch):
    torch = make_torch(
        available=True,
        cuda_version=None,
        hip_version="6.4",
        cudnn_version=91003,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert GpuCapsUtil.has_torch_cuda_api_device()
    assert not GpuCapsUtil.is_torch_nvidia_cuda()
    assert GpuCapsUtil.is_torch_rocm()
    assert GpuCapsUtil.cudnn_version() is None


def test_detects_no_torch_accelerator(monkeypatch):
    torch = make_torch(available=False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert not GpuCapsUtil.has_torch_cuda_api_device()
    assert not GpuCapsUtil.is_torch_nvidia_cuda()
    assert not GpuCapsUtil.is_torch_rocm()
    assert GpuCapsUtil.cudnn_version() is None


def test_detects_ctranslate2_float16_gpu(monkeypatch):
    ctranslate2 = SimpleNamespace(
        get_cuda_device_count=lambda: 1,
        get_supported_compute_types=lambda device, device_index: {
            "float16",
            "int8_float16",
        },
    )
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)

    assert GpuCapsUtil.has_ctranslate2_float16_gpu()


def test_rejects_ctranslate2_gpu_without_float16(monkeypatch):
    ctranslate2 = SimpleNamespace(
        get_cuda_device_count=lambda: 1,
        get_supported_compute_types=lambda device, device_index: {"float32", "int8"},
    )
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)

    assert not GpuCapsUtil.has_ctranslate2_float16_gpu()


def test_rejects_missing_ctranslate2_gpu(monkeypatch):
    ctranslate2 = SimpleNamespace(
        get_cuda_device_count=lambda: 0,
        get_supported_compute_types=lambda device, device_index: {"float16"},
    )
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)

    assert not GpuCapsUtil.has_ctranslate2_float16_gpu()


def test_stt_config_defaults_to_ctranslate2_gpu_when_float16_is_supported(monkeypatch):
    monkeypatch.setattr(GpuCapsUtil, "has_ctranslate2_float16_gpu", lambda: True)

    assert SttConfig.get_default() == SttConfig.CUDA_FLOAT16


def test_stt_config_defaults_to_cpu_without_ctranslate2_float16_gpu(monkeypatch):
    monkeypatch.setattr(GpuCapsUtil, "has_ctranslate2_float16_gpu", lambda: False)

    assert SttConfig.get_default() == SttConfig.CPU_INT8FLOAT32


def test_stt_description_uses_effective_ctranslate2_device(monkeypatch):
    monkeypatch.setattr(Stt, "should_use_mlx_whisper", lambda: False)
    monkeypatch.setattr(Stt, "_config", SttConfig.CUDA_FLOAT16)
    monkeypatch.setattr(GpuCapsUtil, "has_ctranslate2_float16_gpu", lambda: False)

    assert Stt.short_description().endswith(", cpu")


def test_stt_runtime_falls_back_without_ctranslate2_float16_gpu(monkeypatch):
    init_args = {}

    class FakeFasterWhisperAdapter:
        def __init__(self, model, device, compute_type, cpu_threads):
            init_args.update(
                model=model,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )

    monkeypatch.setattr(Stt, "should_use_mlx_whisper", lambda: False)
    monkeypatch.setattr(Stt, "_whisper", None)
    monkeypatch.setattr(Stt, "_config", SttConfig.CUDA_FLOAT16)
    monkeypatch.setattr(GpuCapsUtil, "has_ctranslate2_float16_gpu", lambda: False)
    monkeypatch.setattr("tts_audiobook_tool.stt.FasterWhisperAdapter", FakeFasterWhisperAdapter)

    Stt.get_whisper()

    assert init_args["device"] == "cpu"
    assert init_args["compute_type"] == "int8_float32"
