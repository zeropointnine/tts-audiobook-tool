from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("qwen_tts")

from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.qwen3_model import Qwen3Model
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel


class FakeQwenWrapper:
    def __init__(self):
        self.create_calls: list[tuple[str, str]] = []
        self.created_prompts: list[VoiceClonePromptItem] = []
        self.generate_defaults = {}
        self.merge_calls: list[dict] = []

    def _merge_generate_kwargs(self, **kwargs):
        self.merge_calls.append(kwargs)
        merged = {
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
            "repetition_penalty": 1.05,
            "subtalker_dosample": True,
            "subtalker_top_k": 50,
            "subtalker_top_p": 1.0,
            "subtalker_temperature": 0.9,
            "max_new_tokens": 2048,
        }
        merged.update(self.generate_defaults)
        merged.update(kwargs)
        return merged

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        self.create_calls.append((ref_audio, ref_text))
        prompt = VoiceClonePromptItem(
            ref_code=torch.tensor([[1.0, 2.0]], requires_grad=True),
            ref_spk_embedding=torch.tensor([3.0, 4.0], requires_grad=True),
            x_vector_only_mode=x_vector_only_mode,
            icl_mode=True,
            ref_text=ref_text,
        )
        self.created_prompts.append(prompt)
        return [prompt]


def make_model() -> Qwen3Model:
    model = Qwen3Model.__new__(Qwen3Model)
    model._model_target = "fake"
    model._model = FakeQwenWrapper()
    model._voice_info = None
    model.cached_continuation_history = []

    def fake_generate_base_with_codes(prompts, voice_clone_prompt, languages, gen_kwargs):
        wavs = [np.zeros(8, dtype=np.float32) for _ in prompts]
        codes = [torch.zeros((1, 2), dtype=torch.long) for _ in prompts]
        return wavs, 24_000, codes

    model.generate_base_with_codes = fake_generate_base_with_codes
    return model


def generate(
    model: Qwen3Model,
    voice_path: str,
    transcript: str,
    print_params: bool = False,
):
    return model.generate_base(
        prompts=["Test sentence."],
        voice_info=(voice_path, transcript),
        language="english",
        gen_kwargs=model._build_gen_kwargs(seed=1),
        rolling_continuation_max_segments=0,
        print_params=print_params,
    )


def test_qwen_build_gen_kwargs_resolves_defaults_and_mirrors_overrides():
    model = make_model()
    model._model.generate_defaults = {
        "temperature": 0.8,
        "top_k": 40,
        "subtalker_top_k": 30,
    }

    defaults = model._build_gen_kwargs(seed=7)
    assert defaults["temperature"] == 0.8
    assert defaults["top_k"] == 40
    assert defaults["subtalker_top_k"] == 30
    assert defaults["top_p"] == 1.0
    assert defaults["repetition_penalty"] == 1.05
    assert defaults["max_new_tokens"] == 2048
    assert defaults["seed"] == 7

    overrides = model._build_gen_kwargs(
        temperature=0.6,
        top_k=25,
        top_p=0.85,
        repetition_penalty=1.1,
        seed=8,
    )
    assert overrides["temperature"] == 0.6
    assert overrides["subtalker_temperature"] == 0.6
    assert overrides["top_k"] == 25
    assert overrides["subtalker_top_k"] == 25
    assert overrides["top_p"] == 0.85
    assert overrides["subtalker_top_p"] == 0.85
    assert overrides["repetition_penalty"] == 1.1
    assert overrides["seed"] == 8


def test_qwen_base_prints_resolved_params_without_voice_info(tmp_path, monkeypatch):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice")
    captured = {}

    def capture_params(cls, values, keys_blacklist=None):
        captured.update(values)

    monkeypatch.setattr(TtsBaseModel, "print_params", classmethod(capture_params))

    model = make_model()
    result = generate(model, str(voice_path), "secret transcript", print_params=True)

    assert not isinstance(result, str)
    assert captured["seed"] == 1
    assert captured["temperature"] == 0.9
    assert captured["top_k"] == 50
    assert captured["subtalker_top_k"] == 50
    assert captured["language"] == "english"
    assert "rolling_continuation_max_segments" not in captured
    assert "voice_info" not in captured
    assert str(voice_path) not in captured.values()
    assert "secret transcript" not in captured.values()


def test_qwen_reuses_cpu_voice_prompts_for_a_b_a(tmp_path):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    voice_a.write_bytes(b"a")
    voice_b.write_bytes(b"b")
    model = make_model()

    assert not isinstance(generate(model, str(voice_a), "words a"), str)
    assert not isinstance(generate(model, str(voice_b), "words b"), str)
    assert not isinstance(generate(model, str(voice_a), "words a"), str)

    wrapper = model._model
    assert wrapper.create_calls == [
        (str(voice_a), "words a"),
        (str(voice_b), "words b"),
    ]
    assert len(model._voice_clone_cache) == 2

    for source_prompt in wrapper.created_prompts:
        matching = [
            value
            for key, value in model._voice_clone_cache.items()
            if key[1] == source_prompt.ref_text
        ]
        assert len(matching) == 1
        cached_prompt = matching[0]
        assert cached_prompt.ref_code.device.type == "cpu"
        assert cached_prompt.ref_spk_embedding.device.type == "cpu"
        assert not cached_prompt.ref_code.requires_grad
        assert not cached_prompt.ref_spk_embedding.requires_grad
        assert cached_prompt.ref_code.data_ptr() != source_prompt.ref_code.data_ptr()
        assert cached_prompt.ref_spk_embedding.data_ptr() != source_prompt.ref_spk_embedding.data_ptr()


def test_qwen_rebuilds_changed_voice_and_clears_cache(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"first")
    model = make_model()

    assert not isinstance(generate(model, str(voice_path), "first transcript"), str)
    assert not isinstance(generate(model, str(voice_path), "second transcript"), str)
    assert len(model._voice_clone_cache) == 1

    voice_path.write_bytes(b"different file contents")
    assert not isinstance(generate(model, str(voice_path), "second transcript"), str)

    assert len(model._model.create_calls) == 3
    assert len(model._voice_clone_cache) == 1

    model.clear_voice()
    assert model._voice_info is None
    assert model._voice_clone_cache == {}

    assert not isinstance(generate(model, str(voice_path), "second transcript"), str)
    model.kill()
    assert model._voice_clone_cache == {}
    assert model._model is None


@pytest.mark.parametrize("model_type", ["", "unknown", "base", "custom_voice", "voice_design"])
def test_qwen_instruction_capability_uses_saved_checkpoint_type(model_type):
    project = Project(tts_model_type="qwen3tts_local")
    project.set_model_setting("qwen3tts_local", "model_type", model_type)

    assert Qwen3Model.can_instructions(project) is (model_type in ("custom_voice", "voice_design"))


@pytest.mark.parametrize("saved_type", ["base", "custom_voice", "voice_design"])
@pytest.mark.parametrize("runtime_type", ["unknown", "base", "custom_voice", "voice_design"])
def test_qwen_instruction_capability_prefers_loaded_checkpoint(saved_type, runtime_type):
    project = Project(tts_model_type="qwen3tts_local")
    project.set_model_setting("qwen3tts_local", "model_type", saved_type)
    model = make_model()
    model._model.model = SimpleNamespace(tts_model_type=runtime_type)

    assert Qwen3Model.can_instructions(project, model) is (runtime_type in ("custom_voice", "voice_design"))


@pytest.mark.parametrize("model_type", ["", "unknown", "base", "custom_voice", "voice_design"])
def test_qwen_voice_clone_capability_uses_saved_checkpoint_type(model_type):
    project = Project(tts_model_type="qwen3tts_local")
    project.set_model_setting("qwen3tts_local", "model_type", model_type)

    assert Qwen3Model.can_voice_clone(project) is (model_type not in ("custom_voice", "voice_design"))


@pytest.mark.parametrize("saved_type", ["base", "custom_voice", "voice_design"])
@pytest.mark.parametrize("runtime_type", ["unknown", "base", "custom_voice", "voice_design"])
def test_qwen_voice_clone_capability_prefers_loaded_checkpoint(saved_type, runtime_type):
    project = Project(tts_model_type="qwen3tts_local")
    project.set_model_setting("qwen3tts_local", "model_type", saved_type)
    model = make_model()
    model._model.model = SimpleNamespace(tts_model_type=runtime_type)

    assert Qwen3Model.can_voice_clone(project, model) is (runtime_type not in ("custom_voice", "voice_design"))


@pytest.mark.parametrize("model_type", ["custom_voice", "voice_design"])
@pytest.mark.parametrize("missing_instructions", [{}, {"qwen3tts_sglomni": "Not local"}, {"qwen3tts_local": ""}])
def test_qwen_project_generation_routes_selected_voice_instructions_to_library(
    monkeypatch, model_type, missing_instructions
):
    project = Project(tts_model_type="qwen3tts_local", voice_references=[
        Voice(presets={"qwen3tts_local": "Ryan"}, instructions={"qwen3tts_local": "Speak warmly"}),
        Voice(presets={"qwen3tts_local": "Vivian"}, instructions={"qwen3tts_local": "Speak sternly"}),
        Voice(presets={"qwen3tts_local": "Ryan"}, instructions=missing_instructions),
    ])
    project.set_model_setting("qwen3tts_local", "model_type", model_type)
    project.set_model_setting("qwen3tts_local", "seed", 7)
    model = make_model()
    wrapper = model._model
    wrapper.model = SimpleNamespace(tts_model_type=model_type)
    wrapper.get_supported_speakers = lambda: ["Vivian", "Ryan"]
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return [np.zeros(8, dtype=np.float32) for _ in kwargs["text"]], 24_000

    # Exercise the real project adapter and instruction batching down to the
    # fake library entry point, rather than replacing model generation methods.
    setattr(wrapper, f"generate_{model_type}", generate)
    original_get = Project.get_model_setting

    def get_setting(self, model_id, name):
        assert name != "instructions", "There must be no global instruction fallback"
        return original_get(self, model_id, name)

    monkeypatch.setattr(Project, "get_model_setting", get_setting)
    prompts = ["First sentence.", "Second sentence."]
    for index in (0, 1, 2, 3):
        result = model.generate_using_project(project, prompts, voice_selection_index=index)
        assert not isinstance(result, str)
        assert len(result) == len(prompts)

    empty = None if model_type == "custom_voice" else ["", ""]
    assert [call["instruct"] for call in calls] == [
        ["Speak warmly", "Speak warmly"],
        ["Speak sternly", "Speak sternly"],
        empty,
        ["Speak warmly", "Speak warmly"],
    ]
    assert all(call["text"] == prompts for call in calls)
    assert all(call["language"] == ["english", "english"] for call in calls)
    if model_type == "custom_voice":
        # Each voice's own speaker
        assert [call["speaker"] for call in calls] == [
            [speaker, speaker] for speaker in ("Ryan", "Vivian", "Ryan", "Ryan")
        ]


@pytest.mark.parametrize("speakers, preset, expected", [
    (["Vivian", "Ryan"], "", "No speaker set for this voice"),
    (["Vivian", "Ryan"], "Nobody", "Invalid speaker id for this model: Nobody"),
    (["Vivian"], "", None),  # single-speaker checkpoints always use their only speaker
])
def test_qwen_custom_voice_generation_needs_a_valid_voice_speaker(speakers, preset, expected):
    project = Project(tts_model_type="qwen3tts_local",
                      voice_references=[Voice(presets={"qwen3tts_local": preset} if preset else {})])
    project.set_model_setting("qwen3tts_local", "model_type", "custom_voice")
    model = make_model()
    wrapper = model._model
    wrapper.model = SimpleNamespace(tts_model_type="custom_voice")
    wrapper.get_supported_speakers = lambda: speakers
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return [np.zeros(8, dtype=np.float32) for _ in kwargs["text"]], 24_000

    wrapper.generate_custom_voice = generate
    result = model.generate_using_project(project, ["Hello."])
    if expected is None:
        assert not isinstance(result, str) and calls[0]["speaker"] == ["Vivian"]
    else:
        assert result == expected and calls == []


def test_qwen_base_project_generation_still_routes_selected_clone_not_instructions(tmp_path, monkeypatch):
    voices = [
        Voice("a.flac", "Words A", instructions={"qwen3tts_local": "Unused A"}),
        Voice("b.flac", "Words B", instructions={"qwen3tts_local": "Unused B"}),
    ]
    project = Project(dir_path=str(tmp_path), tts_model_type="qwen3tts_local", voice_references=voices)
    project.set_model_setting("qwen3tts_local", "model_type", "base")
    project.set_model_setting("qwen3tts_local", "seed", 7)
    for voice in voices:
        (tmp_path / voice.file_name).write_bytes(b"voice")
    model = make_model()
    model._model.model = SimpleNamespace(tts_model_type="base")
    original_get = Project.get_model_setting

    def get_setting(self, model_id, name):
        assert name != "instructions", "Base must not read global instructions"
        return original_get(self, model_id, name)

    monkeypatch.setattr(Project, "get_model_setting", get_setting)
    for index in (0, 1, 0):
        result = model.generate_using_project(project, ["A sentence."], voice_selection_index=index)
        assert not isinstance(result, str)
    assert model._model.create_calls == [
        (str(tmp_path / "a.flac"), "Words A"),
        (str(tmp_path / "b.flac"), "Words B"),
    ]


@pytest.mark.parametrize("instructions", ["", "Speak warmly"])
def test_qwen_voice_design_warnings_do_not_read_model_global_instructions(monkeypatch, instructions):
    project = Project(tts_model_type="qwen3tts_local", voice_references=[
        Voice(instructions={"qwen3tts_local": instructions}),
    ])
    project.set_model_setting("qwen3tts_local", "model_type", "voice_design")
    model = make_model()
    original_get = Project.get_model_setting

    def get_setting(self, model_id, name):
        assert name != "instructions", "Warnings must not read global instructions"
        return original_get(self, model_id, name)

    monkeypatch.setattr(Project, "get_model_setting", get_setting)
    assert model.get_warning_issues(project) == [
        'Language value passed to model: english (project language code = "en")']
    project.language_code = "xx"
    assert model.get_warning_issues(project) == [
        'Language value passed to model: auto (project language code = "xx"; no matching Qwen3-TTS language)']
