"""Breeze TTS 2 local (`breeze_tts_2_local`): catalog, wiring, menus and the generation call (fake library)."""
import contextlib
import types
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.breeze_base_model import BreezeBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType

MODEL_ID = "breeze_tts_2_local"


def make_project(tmp_path, voice_references=None) -> Project:
    # The reference-code cache stats the voice file
    for reference in voice_references or []:
        if reference.get("file_name"):
            (tmp_path / reference["file_name"]).write_bytes(b"x")
    project = Project.model_validate({
        "dir_path": str(tmp_path),
        "tts_model_type": MODEL_ID,
        "voice_references": voice_references or [],
    })
    return project


def test_breeze_spec_and_tts_registry(monkeypatch):
    info = TtsModelType.require_by_id(MODEL_ID).value

    assert info.backend_kind is TtsBackendKind.LOCAL
    assert info.local_module_test == "breeze_tts"
    assert info.local_module_exclude == []
    assert info.local_torch_devices == [DeviceType.CUDA]
    assert info.file_tag == TtsModelType.require_by_id("breeze_tts_2_audiocpp").value.file_tag == "breeze2"
    assert info.default_output_sample_rate == 24_000
    assert not info.requires_voice
    assert not info.can_stream
    assert info.can_instructions
    assert info.ui["instructions_message"] == TtsModelType.require_by_id("breeze_tts_2_audiocpp").value.ui["instructions_message"]
    assert info.requirements_file_name == "requirements-breeze.txt"
    # Own settings, not shared with the audio.cpp entry
    assert REGISTRY.voice_binding(MODEL_ID).name == "file_name"
    assert REGISTRY.transcript_binding(MODEL_ID).name == "transcript"
    assert REGISTRY.voice_binding(MODEL_ID).group == ""

    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id(MODEL_ID))
    assert Tts.get_class_for_type(TtsModelType.require_by_id(MODEL_ID)) is BreezeBaseModel
    assert Tts._MODEL_REGISTRY[TtsModelType.require_by_id(MODEL_ID)][2] == "_breeze"


def test_breeze_requires_reference_transcript(tmp_path):
    project = make_project(tmp_path, [{"file_name": "voice.flac", "transcript": ""}])
    assert Tts.requires_reference_transcript(project)


def test_breeze_model_menu_items(monkeypatch, tmp_path):
    state = cast(State, SimpleNamespace(project=make_project(tmp_path)))
    captured = {}
    monkeypatch.setattr(ModelMenuShared, "menu_wrapper",
                        lambda state, items, **kwargs: captured.update(items=items(state)))
    ModelMenuShared.menu(state)
    labels = [strip_ansi_codes(get_string_from(state, item.label)) for item in captured["items"]]
    prefixes = ["Temperature", "Top-P", "Top_K", "Repetition penalty", "CFG", "Seed"]
    assert len(labels) == len(prefixes)
    assert all(label.startswith(prefix) for label, prefix in zip(labels, prefixes))


def test_breeze_sampling_defaults_and_overrides(tmp_path):
    project = make_project(tmp_path)
    sampling = BreezeBaseModel.resolve_sampling(project)
    assert (sampling.temperature, sampling.top_k, sampling.top_p, sampling.repetition_penalty, sampling.cfg) == \
        (0.9, 50, 1.0, 1.1, 4.0)
    for name, value in [("temperature", 0.7), ("top_k", 30), ("top_p", 0.8), ("repetition_penalty", 1.3), ("cfg", 2.5)]:
        project.set_model_setting(MODEL_ID, name, value)
    sampling = BreezeBaseModel.resolve_sampling(project)
    assert (sampling.temperature, sampling.top_k, sampling.top_p, sampling.repetition_penalty, sampling.cfg) == \
        (0.7, 30, 0.8, 1.3, 2.5)


# --- Generation (fake library; needs the `breeze_tts` venv) ---

class FakeBreezeLibrary:
    def __init__(self):
        self.seeds: list[int] = []
        self.prepare_calls: list[dict] = []
        self.generate_calls: list[dict] = []
        self.generation_configs: list[dict] = []
        self.encoded: list[str] = []
        self.events: list[tuple] = []
        self.bound = False
        self.generate_state: list[dict] = []


@pytest.fixture
def fake_model(monkeypatch):
    pytest.importorskip("breeze_tts")
    import torch
    from tts_audiobook_tool.tts_models import breeze_model as module

    lib = FakeBreezeLibrary()

    from breeze_tts.infer import templates

    def prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, **kwargs):
        lib.prepare_calls.append({"requests": requests, "template": template.name, **kwargs})
        # As upstream: the reference clip goes through the module-level encode hook once per request
        for request in requests:
            if request.get("ref_audio_path"):
                templates._encode_prompt_audio(audio_tokenizer, request["ref_audio_path"])
        return {"input_ids": "ids", "num_rows": len(requests), "attention_mask": torch.ones(len(requests), 100)}

    def encode_prompt_audio(audio_tokenizer, audio_path):
        lib.encoded.append(str(audio_path))
        return torch.zeros(10, 16, dtype=torch.int16)

    class FakeGraph:
        def __init__(self, weight):
            self.captured_weight = weight
            self.sampling: dict = {}

        def set_temperature(self, value): self.sampling["temperature"] = value
        def set_top_k(self, value): self.sampling["top_k"] = value
        def set_top_p(self, value): self.sampling["top_p"] = value

    class FakeBatchDepthDecoder:
        """As upstream: one graph per bucket; graph setup casts the codebooks head to fp32 in place."""
        def __init__(self, model, max_batch_size):
            self.model = model
            self.graph = None
            self.bucket_size = None
            self.captures = 0

        def prepare(self, batch_size):
            lib.events.append(("prepare", batch_size))
            bucket = 2 * (1 << (batch_size - 1).bit_length())
            if self.graph is not None and self.bucket_size == bucket:
                return
            head = self.model.depth_decoder.codebooks_head
            head.weight.data = head.weight.data.float()
            self.graph = FakeGraph(head.weight.data)
            self.bucket_size = bucket
            self.captures += 1

        @contextlib.contextmanager
        def bound(self, model):
            lib.bound = True
            try:
                yield
            finally:
                lib.bound = False

    class FakeModel:
        def __init__(self):
            self.depth_decoder = SimpleNamespace(
                codebooks_head=torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16))

        def generate(self, **kwargs):
            lib.generate_calls.append(kwargs)
            lib.generate_state.append(
                {"bound": lib.bound, "head_dtype": self.depth_decoder.codebooks_head.weight.dtype})
            # One row per request, each with its own (EOS-trimmed) length
            return [torch.zeros(1, 2400 * (row + 1)) for row in range(kwargs["num_rows"])]

    monkeypatch.setattr(module, "set_all_seeds", lambda seed: (lib.seeds.append(seed), lib.events.append(("seed", seed))))
    monkeypatch.setattr(module, "BatchDepthDecoder", FakeBatchDepthDecoder)
    monkeypatch.setattr(module, "prepare_inputs", prepare_inputs)
    monkeypatch.setattr(templates, "_encode_prompt_audio", encode_prompt_audio)
    monkeypatch.setattr(module, "update_generation_config_for_breeze",
                        lambda model, config: lib.generation_configs.append(config))
    model = module.BreezeModel.__new__(module.BreezeModel)
    model._model = FakeModel()
    model._tokenizer = object()
    model._audio_tokenizer = object()
    model._device_type = DeviceType.CUDA
    model._batch_depth_decoder = None
    model._graph_head_weight = None
    return model, lib


def test_breeze_generation_uses_pure_clone_template(fake_model, tmp_path):
    model, lib = fake_model
    project = make_project(tmp_path, [{"file_name": "voice.flac", "transcript": " Hello there. "}])
    project.set_model_setting(MODEL_ID, "seed", 1234)

    result = model.generate_using_project(project, ["One."])

    assert isinstance(result, list) and len(result) == 1
    assert result[0].sr == 24_000 and len(result[0].data) == 2400
    assert lib.seeds == [1234]
    request = lib.prepare_calls[0]["requests"][0]
    assert request["text"] == "One."
    assert request["speaker"] == "S0"
    assert request["ref_text"] == "Hello there."
    assert request["ref_audio_path"].endswith("voice.flac")
    assert "instruction" not in request
    assert lib.prepare_calls[0]["template"] == "ref_clone_tata"
    assert lib.prepare_calls[0]["guidance_scale"] == 1.0
    call = lib.generate_calls[0]
    assert call["output_audio"] is True
    assert call["max_new_tokens"] == BreezeBaseModel.MAX_NEW_TOKENS
    assert call["audio_tokenizer"] is model._audio_tokenizer
    assert [p.penalty for p in call["logits_processor"]] == [BreezeBaseModel.REPETITION_PENALTY_DEFAULT]


def test_breeze_generation_batches_prompts_in_one_call(fake_model, tmp_path):
    model, lib = fake_model
    project = make_project(tmp_path, [{"file_name": "voice.flac", "transcript": "Hi."}])
    project.set_model_setting(MODEL_ID, "seed", 1234)

    result = model.generate_using_project(project, ["One.", "Two.", "Three."])

    assert isinstance(result, list)
    # Output order follows prompt order; each row keeps its own length
    assert [len(sound.data) for sound in result] == [2400, 4800, 7200]
    assert lib.seeds == [1234]
    assert len(lib.prepare_calls) == len(lib.generate_calls) == 1
    requests = lib.prepare_calls[0]["requests"]
    assert [r["text"] for r in requests] == ["One.", "Two.", "Three."]
    assert len({r["id"] for r in requests}) == 3
    assert all(r["ref_text"] == "Hi." and r["ref_audio_path"].endswith("voice.flac") for r in requests)


def test_breeze_generation_batch_row_mismatch_is_an_error(fake_model, tmp_path):
    model, lib = fake_model
    model._model.generate = lambda **kwargs: lib.generate_calls.append(kwargs) or []
    project = make_project(tmp_path, [{"file_name": "voice.flac", "transcript": "Hi."}])

    result = model.generate_using_project(project, ["One.", "Two."])

    assert isinstance(result, str) and "mismatch" in result


def test_breeze_has_batch_size_setting():
    assert REGISTRY.orchestration_binding(MODEL_ID).name == "batch_size"
    assert TtsModelType.require_by_id(MODEL_ID).can_batch()
    assert Tts.can_batch(TtsModelType.require_by_id(MODEL_ID))


def test_breeze_generation_random_seed(fake_model, tmp_path):
    model, lib = fake_model
    project = make_project(tmp_path, [{"file_name": "voice.flac", "transcript": "Hi."}])
    project.set_model_setting(MODEL_ID, "seed", 1234)

    model.generate_using_project(project, ["One."], force_random_seed=True, max_random_seed=10)

    assert 0 <= lib.seeds[0] <= 10


def test_breeze_generation_sample_requires_transcript(fake_model, tmp_path):
    model, lib = fake_model
    result = model.generate_using_project(
        make_project(tmp_path, [{"file_name": "voice.flac", "transcript": "  "}]), ["One."])
    assert isinstance(result, str) and "transcript is required" in result
    assert lib.generate_calls == []


@pytest.mark.parametrize("voice_reference, template, guidance_scale", [
    # Sample only: pure clone, CFG not applicable
    ({"file_name": "voice.flac", "transcript": "Hi."}, "ref_clone_tata", 1.0),
    # Sample + instructions: voice direction, CFG
    ({"file_name": "voice.flac", "transcript": "Hi.", "instructions": {MODEL_ID: " Slowly. "}}, "ref_edit_tata", 4.0),
    # Instructions only: voice design, CFG
    ({"file_name": "", "transcript": "", "instructions": {MODEL_ID: "A calm narrator."}}, "tts_instruction", 4.0),
    # Neither: plain TTS (arbitrary voice)
    ({"file_name": "", "transcript": ""}, "tts_plain", 1.0),
    # Other models' instructions are ignored
    ({"file_name": "voice.flac", "transcript": "Hi.", "instructions": {"breeze_tts_2_audiocpp": "Loud."}},
     "ref_clone_tata", 1.0),
])
def test_breeze_generation_template_per_voice(fake_model, tmp_path, voice_reference, template, guidance_scale):
    model, lib = fake_model
    project = make_project(tmp_path, [voice_reference])

    result = model.generate_using_project(project, ["One.", "Two."])

    assert isinstance(result, list) and len(result) == 2
    call = lib.prepare_calls[0]
    assert call["template"] == template
    assert call["guidance_scale"] == guidance_scale
    for request in call["requests"]:
        assert ("ref_audio_path" in request) == ("ref_text" in request) == bool(voice_reference["file_name"])
        instructions = voice_reference.get("instructions", {}).get(MODEL_ID, "").strip()
        assert request.get("instruction", "") == instructions


def test_breeze_generation_applies_sampling_settings(fake_model, tmp_path):
    model, lib = fake_model
    project = make_project(tmp_path, [{"file_name": "", "transcript": "", "instructions": {MODEL_ID: "Calm."}}])
    for name, value in [("temperature", 0.7), ("top_k", 30), ("top_p", 0.8), ("repetition_penalty", 1.3), ("cfg", 2.5)]:
        project.set_model_setting(MODEL_ID, name, value)

    model.generate_using_project(project, ["One."])

    config = lib.generation_configs[0]
    assert (config["temperature"], config["top_k"], config["top_p"]) == (0.7, 30, 0.8)
    assert (config["depth_decoder_temperature"], config["depth_decoder_top_k"], config["depth_decoder_top_p"]) == \
        (0.7, 30, 0.8)
    assert lib.prepare_calls[0]["guidance_scale"] == 2.5
    assert [p.penalty for p in lib.generate_calls[0]["logits_processor"]] == [1.3]


def test_breeze_reference_codes_are_cached(fake_model, tmp_path):
    from breeze_tts.infer import templates
    model, lib = fake_model
    (tmp_path / "voice.flac").write_bytes(b"x")
    project = make_project(tmp_path, [{"file_name": "voice.flac", "transcript": "Hi."}])
    original = templates._encode_prompt_audio

    model.generate_using_project(project, ["One.", "Two."])
    model.generate_using_project(project, ["Three."])

    # 3 requests encoded through the hook, the real encoder ran once; hook restored afterwards
    assert lib.encoded == [str(tmp_path / "voice.flac")]
    assert templates._encode_prompt_audio is original


def test_breeze_cfg_uses_depth_graph_and_keeps_checkpoint_head(fake_model, tmp_path):
    import torch
    model, lib = fake_model
    project = make_project(tmp_path, [{"file_name": "", "transcript": "", "instructions": {MODEL_ID: "Calm."}}])
    project.set_model_setting(MODEL_ID, "seed", 5)
    project.set_model_setting(MODEL_ID, "temperature", 0.7)
    head = model._model.depth_decoder.codebooks_head

    for prompts in (["One."], ["Two."], ["Three.", "Four.", "Five."]):
        model.generate_using_project(project, prompts)
        # Eager paths always see the checkpoint dtype; the graph's fp32 copy stays referenced (its memory is replayed)
        assert head.weight.dtype == torch.bfloat16
        graph = model._batch_depth_decoder.graph
        assert model._graph_head_weight.dtype == torch.float32
        assert model._graph_head_weight.data_ptr() == graph.captured_weight.data_ptr()
        assert graph.sampling == {"temperature": 0.7, "top_k": 50, "top_p": 1.0}

    # Captured once per bucket (2 for batch 1, 8 for batch 3); re-seeded after each prepare (capture consumes RNG)
    assert model._batch_depth_decoder.captures == 2
    assert lib.events[-2:] == [("prepare", 3), ("seed", 5)]
    assert all(state == {"bound": True, "head_dtype": torch.bfloat16} for state in lib.generate_state)


def test_breeze_without_cfg_does_not_use_depth_graph(fake_model, tmp_path):
    model, lib = fake_model
    project = make_project(tmp_path, [{"file_name": "voice.flac", "transcript": "Hi."}])
    project.set_model_setting(MODEL_ID, "cfg", 4.0)

    model.generate_using_project(project, ["One."])

    assert model._batch_depth_decoder is None
    assert lib.generate_state[0]["bound"] is False


def test_breeze_truncation_warning(fake_model, tmp_path, capsys):
    import torch
    model, lib = fake_model
    frame = 24_000 // 12.5
    model._model.generate = lambda **kwargs: [torch.zeros(1, int(frame * BreezeBaseModel.MAX_NEW_TOKENS)),
                                              torch.zeros(1, int(frame * 10))]
    project = make_project(tmp_path, [{"file_name": "voice.flac", "transcript": "Hi."}])

    result = model.generate_using_project(project, ["One.", "Two."])

    assert isinstance(result, list) and len(result) == 2
    out = strip_ansi_codes(capsys.readouterr().out)
    assert "1500-frame limit" in out and "batch item 1/2" in out and "batch item 2/2" not in out
