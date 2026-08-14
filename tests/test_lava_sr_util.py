from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tts_audiobook_tool.app_types import DeviceType, Sound
from tts_audiobook_tool.sound import lava_sr_util
from tts_audiobook_tool.sound.lava_sr_util import LavaSrUtil


class FakeFastLRMerge:
    instances = []

    def __init__(
        self,
        sample_rate=48_000,
        cutoff=4_000,
        transition_bins=256,
        device="cpu",
    ):
        self.sample_rate = sample_rate
        self.cutoff = cutoff
        self.transition_bins = transition_bins
        self.device = device
        self.mask_cache = {}
        FakeFastLRMerge.instances.append(self)


class FakeLavaModel:
    instances = []

    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.calls = []
        self.bwe_model = SimpleNamespace(lr_refiner=None)
        FakeLavaModel.instances.append(self)

    def enhance(self, wav, enhance=True, denoise=True, batch=False):
        self.calls.append(
            {
                "wav": wav.clone(),
                "enhance": enhance,
                "denoise": denoise,
                "batch": batch,
            }
        )
        return wav.repeat_interleave(3, dim=-1)


@pytest.fixture(autouse=True)
def reset_fake_models():
    FakeLavaModel.instances = []
    FakeFastLRMerge.instances = []


@pytest.fixture
def fake_lava_module(monkeypatch):
    real_import_module = lava_sr_util.importlib.import_module

    def fake_import_module(name, package=None):
        if name == "LavaSR.model":
            return SimpleNamespace(LavaEnhance2=FakeLavaModel)
        if name == "LavaSR.enhancer.linkwitz_merge":
            return SimpleNamespace(FastLRMerge=FakeFastLRMerge)
        return real_import_module(name, package)

    monkeypatch.setattr(lava_sr_util.importlib, "import_module", fake_import_module)


def set_accelerators(monkeypatch, *, cuda=False, mps=False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


class FakeWorkerConnection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        return self.responses.pop(0)

    def poll(self, timeout=None):
        return bool(self.responses)

    def close(self):
        self.closed = True


class FakeWorkerProcess:
    def __init__(self, connection):
        self.connection = connection
        self.alive = False
        self.terminated = False
        self.stops_on_kill_command = True

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        if self.stops_on_kill_command and any(
            message[0] == "kill" for message in self.connection.sent
        ):
            self.alive = False

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.alive = False


class FakeWorkerContext:
    def __init__(self, responses):
        self.parent_connection = FakeWorkerConnection(responses)
        self.child_connection = FakeWorkerConnection([])
        self.process = FakeWorkerProcess(self.parent_connection)

    def Pipe(self):
        return self.parent_connection, self.child_connection

    def Process(self, **kwargs):
        return self.process


def test_has_lava_sr_detects_model_class(monkeypatch):
    monkeypatch.setattr(
        lava_sr_util.importlib,
        "import_module",
        lambda name: SimpleNamespace(LavaEnhance2=object),
    )

    assert LavaSrUtil.has_lava_sr()


def test_has_lava_sr_is_false_when_optional_package_is_missing(monkeypatch):
    def fail_import(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(lava_sr_util.importlib, "import_module", fail_import)

    assert not LavaSrUtil.has_lava_sr()


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_resolve_device_uses_preferred_available_accelerator(
    monkeypatch,
    cuda_available,
    mps_available,
    expected,
):
    set_accelerators(
        monkeypatch,
        cuda=cuda_available,
        mps=mps_available,
    )

    assert LavaSrUtil.resolve_device() == expected


def test_explicit_device_override_accepts_enum(fake_lava_module, monkeypatch):
    set_accelerators(monkeypatch, cuda=True, mps=True)

    util = LavaSrUtil(device=DeviceType.CPU, model_path="local-model")

    assert util.device == "cpu"
    assert FakeLavaModel.instances[0].model_path == "local-model"


def test_explicit_unavailable_accelerator_is_rejected(monkeypatch):
    set_accelerators(monkeypatch, cuda=False, mps=False)

    with pytest.raises(ValueError, match="CUDA is not available"):
        LavaSrUtil.resolve_device("cuda")

    with pytest.raises(ValueError, match="MPS is not available"):
        LavaSrUtil.resolve_device("mps")


def test_cuda_uses_disposable_worker_and_returns_worker_result(monkeypatch):
    set_accelerators(monkeypatch, cuda=True)
    output = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    context = FakeWorkerContext(
        [
            ("ready", None),
            ("result", (output, 48_000)),
            ("stopped", None),
        ]
    )
    monkeypatch.setattr(
        lava_sr_util.multiprocessing,
        "get_context",
        lambda method: context,
    )

    util = LavaSrUtil(device="cuda")
    result = util.process(Sound(np.ones(2, dtype=np.float32), 16_000))
    util.kill()

    assert not isinstance(result, str)
    np.testing.assert_array_equal(result.data, output)
    assert result.sr == 48_000
    assert context.parent_connection.sent[0][0] == "process"
    assert context.parent_connection.sent[1] == ("kill", None)
    assert context.parent_connection.closed
    assert not context.process.is_alive()
    assert util.worker_process is None
    assert util.worker_connection is None
    assert util.model is None


def test_cuda_worker_is_terminated_if_graceful_shutdown_times_out(monkeypatch):
    set_accelerators(monkeypatch, cuda=True)
    context = FakeWorkerContext([("ready", None)])
    context.process.stops_on_kill_command = False
    monkeypatch.setattr(
        lava_sr_util.multiprocessing,
        "get_context",
        lambda method: context,
    )

    util = LavaSrUtil(device="cuda")
    util.kill()

    assert context.process.terminated
    assert not context.process.is_alive()


def test_process_resamples_and_forwards_bandwidth_only_defaults(
    fake_lava_module,
    monkeypatch,
):
    set_accelerators(monkeypatch)
    resample_calls = []

    def fake_resample(sound, target_sr):
        resample_calls.append((sound.sr, target_sr))
        return Sound(np.array([0.25, -0.5, 0.75], dtype=np.float64), target_sr)

    monkeypatch.setattr(
        lava_sr_util.SoundUtil,
        "resample_if_necessary",
        fake_resample,
    )
    util = LavaSrUtil(device="cpu")

    result = util.process(Sound(np.ones(9, dtype=np.float32), 48_000))

    assert not isinstance(result, str)
    assert result.sr == 48_000
    assert result.data.dtype == np.float32
    assert len(result.data) == 9
    assert resample_calls == [(48_000, 16_000)]
    call = FakeLavaModel.instances[0].calls[0]
    assert call["wav"].shape == (1, 3)
    assert call["wav"].dtype == torch.float32
    assert call["enhance"] is True
    assert call["denoise"] is False
    assert call["batch"] is False
    refiner = FakeFastLRMerge.instances[0]
    assert refiner.cutoff == 24_000
    assert refiner.transition_bins == 1024
    assert refiner.device == "cpu"


@pytest.mark.parametrize(
    ("source_sr", "expected_cutoff"),
    [
        (16_000, 8_000),
        (24_000, 12_000),
        (44_100, 22_050),
        (48_000, 24_000),
    ],
)
def test_process_configures_refiner_from_original_sample_rate(
    fake_lava_module,
    monkeypatch,
    source_sr,
    expected_cutoff,
):
    set_accelerators(monkeypatch)
    util = LavaSrUtil(device="cpu")

    result = util.process(Sound(np.ones(8, dtype=np.float32), source_sr))

    assert not isinstance(result, str)
    refiner = FakeFastLRMerge.instances[0]
    assert refiner.cutoff == expected_cutoff
    assert refiner.transition_bins == 1024
    assert refiner.device == "cpu"


def test_process_forwards_optional_denoising(fake_lava_module, monkeypatch):
    set_accelerators(monkeypatch)
    util = LavaSrUtil(device="cpu")

    result = util.process(
        Sound(np.ones(16, dtype=np.float32), 16_000),
        denoise=True,
    )

    assert not isinstance(result, str)
    assert FakeLavaModel.instances[0].calls[0]["denoise"] is True


def test_long_audio_is_chunked_crossfaded_and_keeps_exact_duration(
    fake_lava_module,
    monkeypatch,
):
    set_accelerators(monkeypatch)
    monkeypatch.setattr(lava_sr_util, "CHUNK_SAMPLES", 6)
    monkeypatch.setattr(lava_sr_util, "OVERLAP_SAMPLES", 2)
    util = LavaSrUtil(device="cpu")
    call_index = 0

    def constant_chunk(wav, enhance=True, denoise=False, batch=False):
        nonlocal call_index
        result = torch.full((1, wav.shape[-1] * 3), float(call_index))
        call_index += 1
        return result

    FakeLavaModel.instances[0].enhance = constant_chunk

    result = util.process(Sound(np.ones(10, dtype=np.float32), 16_000))

    assert not isinstance(result, str)
    assert len(result.data) == 30
    assert call_index == 2
    np.testing.assert_allclose(result.data[:12], 0.0)
    np.testing.assert_allclose(result.data[12:18], np.linspace(0, 1, 6))
    np.testing.assert_allclose(result.data[18:], 1.0)


def test_short_model_output_is_extended_to_preserve_duration(
    fake_lava_module,
    monkeypatch,
):
    set_accelerators(monkeypatch)
    util = LavaSrUtil(device="cpu")
    FakeLavaModel.instances[0].enhance = lambda *args, **kwargs: torch.tensor([[0.5]])

    result = util.process(Sound(np.ones(4, dtype=np.float32), 16_000))

    assert not isinstance(result, str)
    assert len(result.data) == 12
    np.testing.assert_allclose(result.data, 0.5)


def test_process_rejects_non_mono_audio(fake_lava_module, monkeypatch):
    set_accelerators(monkeypatch)
    util = LavaSrUtil(device="cpu")

    result = util.process(Sound(np.ones((2, 8), dtype=np.float32), 16_000))

    assert isinstance(result, str)
    assert "only mono" in result


def test_kill_is_idempotent_and_prevents_further_processing(
    fake_lava_module,
    monkeypatch,
):
    set_accelerators(monkeypatch)
    util = LavaSrUtil(device="cpu")

    util.kill()
    util.kill()
    result = util.process(Sound(np.ones(8, dtype=np.float32), 16_000))

    assert util.model is None
    assert isinstance(result, str)
    assert "kill() was called" in result


def test_mps_inference_failure_warns_and_retries_once_on_cpu(monkeypatch):
    set_accelerators(monkeypatch, mps=True)
    warnings = []
    real_import_module = lava_sr_util.importlib.import_module

    class DeviceSensitiveModel(FakeLavaModel):
        def enhance(self, wav, enhance=True, denoise=False, batch=False):
            if self.device == "mps":
                raise RuntimeError("complex FFT unsupported")
            return super().enhance(wav, enhance, denoise, batch)

    monkeypatch.setattr(
        lava_sr_util.importlib,
        "import_module",
        lambda name, package=None: (
            SimpleNamespace(LavaEnhance2=DeviceSensitiveModel)
            if name == "LavaSR.model"
            else (
                SimpleNamespace(FastLRMerge=FakeFastLRMerge)
                if name == "LavaSR.enhancer.linkwitz_merge"
                else real_import_module(name, package)
            )
        ),
    )
    monkeypatch.setattr(lava_sr_util, "printt", warnings.append)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    util = LavaSrUtil()

    result = util.process(Sound(np.ones(8, dtype=np.float32), 24_000))

    assert not isinstance(result, str)
    assert util.device == "cpu"
    assert [model.device for model in DeviceSensitiveModel.instances] == ["mps", "cpu"]
    assert len(warnings) == 1
    assert "Retrying on CPU" in warnings[0]
    assert [refiner.device for refiner in FakeFastLRMerge.instances] == ["mps", "cpu"]
    assert [refiner.cutoff for refiner in FakeFastLRMerge.instances] == [12_000, 12_000]


def test_mps_initialization_failure_warns_and_loads_on_cpu(monkeypatch):
    set_accelerators(monkeypatch, mps=True)
    warnings = []

    class MpsInitFailureModel(FakeLavaModel):
        attempted_devices = []

        def __init__(self, model_path, device="cpu"):
            self.attempted_devices.append(device)
            if device == "mps":
                raise RuntimeError("MPS operation unsupported")
            super().__init__(model_path, device)

    monkeypatch.setattr(
        lava_sr_util.importlib,
        "import_module",
        lambda name: SimpleNamespace(LavaEnhance2=MpsInitFailureModel),
    )
    monkeypatch.setattr(lava_sr_util, "printt", warnings.append)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)

    util = LavaSrUtil()

    assert util.device == "cpu"
    assert MpsInitFailureModel.attempted_devices == ["mps", "cpu"]
    assert len(warnings) == 1
    assert "initialization" in warnings[0]
    assert "Retrying on CPU" in warnings[0]


def test_cpu_inference_failure_returns_error_without_retry(
    fake_lava_module,
    monkeypatch,
):
    set_accelerators(monkeypatch)
    util = LavaSrUtil(device="cpu")

    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    FakeLavaModel.instances[0].enhance = fail

    result = util.process(Sound(np.ones(8, dtype=np.float32), 16_000))

    assert result == "inference failed"
    assert len(FakeLavaModel.instances) == 1
