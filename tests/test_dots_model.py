from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("dots_tts")

from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models import dots_model
from tts_audiobook_tool.tts_models.dots_base_model import DotsBaseModel
from tts_audiobook_tool.tts_models.dots_model import DotsModel


class FakeRuntime:
    def __init__(self, *, sampling=None, meanflow=None, sample_rate=48_000):
        self.sample_rate = sample_rate
        self.model = SimpleNamespace(
            config=SimpleNamespace(sampling=sampling, meanflow=meanflow)
        )
        self.calls: list[dict] = []
        self.stream_calls: list[dict] = []
        self.stream_chunks = [
            torch.tensor([[0.1]], dtype=torch.float64),
            torch.tensor([[-0.2, 0.3]], dtype=torch.float64),
        ]
        self.error: Exception | None = None
        self.stream_error: Exception | None = None

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {
            "audio": torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64),
            "sample_rate": self.sample_rate,
        }

    def generate_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if self.stream_error is not None:
            raise self.stream_error
        yield from self.stream_chunks


def make_model(runtime: FakeRuntime, target: str = DotsBaseModel.SOAR_REPO_ID) -> DotsModel:
    model = DotsModel.__new__(DotsModel)
    model._runtime = runtime
    model._model_target = target
    model._device_type = DeviceType.CPU
    return model


def test_dots_constructor_uses_device_precision_and_cuda_optimization(monkeypatch):
    calls: list[tuple[str, dict]] = []
    matmul_precisions: list[str] = []

    def fake_from_pretrained(target, **kwargs):
        calls.append((target, kwargs))
        return FakeRuntime()

    monkeypatch.setattr(
        dots_model.DotsTtsRuntime, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(
        dots_model.torch, "set_float32_matmul_precision", matmul_precisions.append
    )

    monkeypatch.setattr(dots_model.torch.cuda, "is_available", lambda: False)
    cpu_model = DotsModel(DotsBaseModel.SOAR_REPO_ID, DeviceType.CPU)
    assert calls[-1] == (
        DotsBaseModel.SOAR_REPO_ID,
        {
            "precision": "float32",
            "optimize": False,
            "max_generate_length": DotsBaseModel.MAX_GENERATE_LENGTH,
        },
    )
    assert cpu_model.get_device_type() is DeviceType.CPU
    assert matmul_precisions == []

    monkeypatch.setattr(dots_model.torch.cuda, "is_available", lambda: True)
    cuda_model = DotsModel(
        DotsBaseModel.MF_REPO_ID,
        DeviceType.CUDA,
    )
    assert calls[-1] == (
        DotsBaseModel.MF_REPO_ID,
        {
            "precision": "bfloat16",
            "optimize": True,
            "max_generate_length": DotsBaseModel.MAX_GENERATE_LENGTH,
        },
    )
    assert cuda_model.get_device_type() is DeviceType.CUDA
    assert matmul_precisions == ["high"]


def test_dots_constructor_rejects_forced_cpu_and_bad_sample_rate(monkeypatch):
    monkeypatch.setattr(dots_model.torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="cannot force CPU"):
        DotsModel(DotsBaseModel.SOAR_REPO_ID, DeviceType.CPU)

    monkeypatch.setattr(dots_model.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        dots_model.DotsTtsRuntime,
        "from_pretrained",
        lambda *args, **kwargs: FakeRuntime(sample_rate=24_000),
    )
    with pytest.raises(ValueError, match="Unexpected dots.tts output sample rate"):
        DotsModel(DotsBaseModel.SOAR_REPO_ID, DeviceType.CPU)


def test_dots_constructor_compile_setting_gates_cuda_optimization(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_from_pretrained(target, **kwargs):
        calls.append((target, kwargs))
        return FakeRuntime()

    monkeypatch.setattr(
        dots_model.DotsTtsRuntime, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(dots_model.torch, "set_float32_matmul_precision", lambda _: None)

    monkeypatch.setattr(dots_model.torch.cuda, "is_available", lambda: True)
    DotsModel(
        DotsBaseModel.SOAR_REPO_ID,
        DeviceType.CUDA,
        compile_enabled=False,
    )
    assert calls[-1][1]["optimize"] is False

    monkeypatch.setattr(dots_model.torch.cuda, "is_available", lambda: False)
    DotsModel(
        DotsBaseModel.SOAR_REPO_ID,
        DeviceType.CPU,
        compile_enabled=True,
    )
    assert calls[-1][1]["optimize"] is False  # CPU never compiles


def test_dots_generation_passes_voice_language_defaults_and_float32(tmp_path, monkeypatch):
    runtime = FakeRuntime()
    model = make_model(runtime)
    seeded: list[int] = []
    monkeypatch.setattr(dots_model, "seed_everything", seeded.append)
    monkeypatch.setattr(dots_model, "printt", lambda *args, **kwargs: None)
    project = Project.model_validate(
        {
            "dir_path": str(tmp_path),
            "language_code": "en",
            "dots_voice_file_name": ["voice.flac"],
            "dots_voice_transcript": ["Reference words."],
        }
    )

    result = model.generate_using_project(project, ["Hello world."])

    assert not isinstance(result, str)
    assert len(result) == 1
    assert result[0].sr == 48_000
    assert result[0].data.dtype == np.float32
    assert seeded == [DotsBaseModel.SEED_DEFAULT]
    assert runtime.stream_calls == []
    assert runtime.calls == [
        {
            "text": "Hello world.",
            "prompt_audio_path": str(tmp_path / "voice.flac"),
            "prompt_text": "Reference words.",
            "template_name": None,
            "language": "en",
            "speaker_scale": DotsBaseModel.SPEAKER_SCALE_DEFAULT,
            "ode_method": None,
            "num_steps": DotsBaseModel.NUM_STEPS_SOAR_DEFAULT,
            "guidance_scale": DotsBaseModel.GUIDANCE_SCALE_DEFAULT,
            "normalize_text": False,
        }
    ]


def test_dots_streaming_emits_float32_chunks_and_returns_sounds(tmp_path, monkeypatch):
    runtime = FakeRuntime()
    runtime.stream_chunks = [
        torch.tensor([[0.1]], dtype=torch.float64),
        torch.empty((1, 0), dtype=torch.float64),
        torch.tensor([[-0.2, 0.3]], dtype=torch.float64),
    ]
    model = make_model(runtime)
    monkeypatch.setattr(dots_model, "seed_everything", lambda _: None)
    monkeypatch.setattr(dots_model, "printt", lambda *args, **kwargs: None)
    project = Project.model_validate(
        {
            "dir_path": str(tmp_path),
            "language_code": "en",
            "dots_voice_file_name": ["voice.flac"],
            "dots_voice_transcript": ["Reference words."],
        }
    )
    emitted: list[np.ndarray] = []
    ended: list[bool] = []

    result = model.generate_using_project(
        project,
        ["First prompt.", "Second prompt."],
        on_stream_chunk=emitted.append,
        on_stream_end=lambda: ended.append(True),
    )

    assert not isinstance(result, str)
    assert runtime.calls == []
    assert len(runtime.stream_calls) == 2
    assert [call["text"] for call in runtime.stream_calls] == [
        "First prompt.",
        "Second prompt.",
    ]
    assert runtime.stream_calls[0] == {
        "text": "First prompt.",
        "prompt_audio_path": str(tmp_path / "voice.flac"),
        "prompt_text": "Reference words.",
        "template_name": None,
        "language": "en",
        "speaker_scale": DotsBaseModel.SPEAKER_SCALE_DEFAULT,
        "ode_method": None,
        "num_steps": DotsBaseModel.NUM_STEPS_SOAR_DEFAULT,
        "guidance_scale": DotsBaseModel.GUIDANCE_SCALE_DEFAULT,
        "normalize_text": False,
    }
    assert len(emitted) == 4
    assert all(chunk.dtype == np.float32 and chunk.ndim == 1 for chunk in emitted)
    np.testing.assert_allclose(emitted[0], [0.1])
    np.testing.assert_allclose(emitted[1], [-0.2, 0.3])
    assert ended == [True]
    assert len(result) == 2
    for sound in result:
        assert sound.sr == 48_000
        assert sound.data.dtype == np.float32
        np.testing.assert_allclose(sound.data, [0.1, -0.2, 0.3])


def test_dots_streaming_is_selected_when_only_end_callback_is_supplied(monkeypatch):
    runtime = FakeRuntime()
    model = make_model(runtime)
    monkeypatch.setattr(dots_model, "seed_everything", lambda _: None)
    monkeypatch.setattr(dots_model, "printt", lambda *args, **kwargs: None)
    ended: list[bool] = []

    result = model.generate(
        prompts=["End callback only."],
        voice_path=None,
        voice_transcript="",
        language=None,
        speaker_scale=1.5,
        num_steps=10,
        guidance_scale=1.2,
        seed=42,
        on_stream_end=lambda: ended.append(True),
    )

    assert not isinstance(result, str)
    assert runtime.calls == []
    assert len(runtime.stream_calls) == 1
    assert ended == [True]


def test_dots_streaming_empty_output_and_errors_do_not_signal_end(monkeypatch):
    runtime = FakeRuntime()
    model = make_model(runtime)
    monkeypatch.setattr(dots_model, "seed_everything", lambda _: None)
    monkeypatch.setattr(dots_model, "printt", lambda *args, **kwargs: None)
    ended: list[bool] = []
    generate_kwargs = {
        "prompts": ["Streaming failure."],
        "voice_path": None,
        "voice_transcript": "",
        "language": None,
        "speaker_scale": 1.5,
        "num_steps": 10,
        "guidance_scale": 1.2,
        "seed": 42,
        "on_stream_end": lambda: ended.append(True),
    }

    runtime.stream_chunks = [torch.empty((1, 0))]
    result = model.generate(**generate_kwargs)
    assert result == "ValueError: dots.tts returned empty audio"
    assert ended == []

    runtime.stream_error = RuntimeError("stream failed")
    result = model.generate(**generate_kwargs)
    assert result == "RuntimeError: stream failed"
    assert ended == []


def test_dots_mf_defaults_and_explicit_sampling_overrides(monkeypatch):
    # Mirrors the real dots.tts-mf artifact: meanflow enabled, sampling open
    runtime = FakeRuntime(meanflow=SimpleNamespace(enabled=True))
    model = make_model(runtime, DotsBaseModel.MF_REPO_ID)
    monkeypatch.setattr(dots_model, "seed_everything", lambda _: None)
    monkeypatch.setattr(dots_model, "printt", lambda *args, **kwargs: None)

    project = Project.model_validate({"dots_target": DotsBaseModel.MF_REPO_ID})
    result = model.generate_using_project(project, ["Default MF."])
    assert not isinstance(result, str)
    assert runtime.calls[-1]["num_steps"] == DotsBaseModel.NUM_STEPS_MF_DEFAULT
    # Meanflow has no CFG branch: guidance is distilled into the model, so
    # the value must not be forwarded even when the project carries one
    assert runtime.calls[-1]["guidance_scale"] is None

    project.dots_num_steps_mf = 7
    project.dots_guidance_scale = 1.7
    project.dots_speaker_scale = 2.1
    result = model.generate_using_project(project, ["Overrides."])
    assert not isinstance(result, str)
    assert runtime.calls[-1]["num_steps"] == 7
    assert runtime.calls[-1]["guidance_scale"] is None
    assert runtime.calls[-1]["speaker_scale"] == 2.1

    # A meanflow section that is present but disabled still means plain
    # flow matching, so guidance stays live
    runtime = FakeRuntime(meanflow=SimpleNamespace(enabled=False))
    model = make_model(runtime, DotsBaseModel.MF_REPO_ID)
    monkeypatch.setattr(dots_model, "seed_everything", lambda _: None)
    monkeypatch.setattr(dots_model, "printt", lambda *args, **kwargs: None)
    result = model.generate_using_project(project, ["Disabled meanflow."])
    assert not isinstance(result, str)
    assert runtime.calls[-1]["num_steps"] == 7
    assert runtime.calls[-1]["guidance_scale"] == 1.7


def test_dots_fixed_artifact_omits_sampling_overrides(monkeypatch):
    runtime = FakeRuntime(sampling=SimpleNamespace(solver="scm"))
    model = make_model(runtime, DotsBaseModel.MF_2STEPS_REPO_ID)
    seeded: list[int] = []
    monkeypatch.setattr(dots_model, "seed_everything", seeded.append)
    monkeypatch.setattr(dots_model.random, "randrange", lambda *_: 99)
    monkeypatch.setattr(dots_model, "printt", lambda *args, **kwargs: None)
    project = Project.model_validate(
        {
            "dots_target": DotsBaseModel.MF_2STEPS_REPO_ID,
            "dots_seed": -1,
            "dots_num_steps_soar": 20,
            "dots_guidance_scale": 2.5,
        }
    )

    result = model.generate_using_project(project, ["Fixed sampling."])

    assert not isinstance(result, str)
    assert seeded == [99]
    assert runtime.calls[-1]["num_steps"] is None
    assert runtime.calls[-1]["guidance_scale"] is None

    result = model.generate_using_project(
        project,
        ["Fixed streaming sampling."],
        on_stream_chunk=lambda _: None,
    )
    assert not isinstance(result, str)
    assert runtime.stream_calls[-1]["num_steps"] is None
    assert runtime.stream_calls[-1]["guidance_scale"] is None


def test_dots_generation_returns_standard_error_and_kill_clears_runtime(monkeypatch):
    runtime = FakeRuntime()
    runtime.error = RuntimeError("generation failed")
    model = make_model(runtime)
    monkeypatch.setattr(dots_model, "seed_everything", lambda _: None)
    monkeypatch.setattr(dots_model, "printt", lambda *args, **kwargs: None)

    result = model.generate(
        prompts=["Failure."],
        voice_path=None,
        voice_transcript="",
        language=None,
        speaker_scale=1.5,
        num_steps=10,
        guidance_scale=1.2,
        seed=42,
    )

    assert result == "RuntimeError: generation failed"
    model.kill()
    assert model._runtime is None


def test_dots_get_warning_issues_reports_language_hint():
    runtime = FakeRuntime()
    model = make_model(runtime)

    warnings = model.get_warning_issues(
        Project.model_validate({"language_code": "ja"})
    )
    assert any("language hint: ja" in warning for warning in warnings)

    warnings = model.get_warning_issues(
        Project.model_validate({"language_code": ""})
    )
    assert any("language hint: auto" in warning for warning in warnings)
