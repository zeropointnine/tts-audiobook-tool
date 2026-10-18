"""Selected-voice instruction routing, independent of global model settings."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppBackendAdapter, AudioCppModelSupport
from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions

OMNI = "omnivoice_audiocpp"
BREEZE = "breeze_tts_2_audiocpp"
COSY = "cosyvoice3_audiocpp"


@pytest.mark.parametrize("model_id,mode,uses_instructions,uses_transcript", [
    (OMNI, None, True, True),
    (BREEZE, None, True, True),
    (COSY, "zero_shot", False, True),
    (COSY, "cross_lingual", False, False),
    (COSY, "instruct", True, False),
])
def test_selected_voice_routes_sample_transcript_and_model_specific_instructions(
        monkeypatch, tmp_path, model_id, mode, uses_instructions, uses_transcript):
    definition = load_audio_cpp_definitions().models[model_id]
    support = AudioCppModelSupport(definition)
    adapter = AudioCppBackendAdapter(definition, support, "exact-server-id")
    project = Project(dir_path=str(tmp_path), tts_model_type=model_id, voice_references=[
        {"file_name": "first.flac", "transcript": "first transcript",
         "instructions": {model_id: "First voice", "unrelated": "Wrong model"}},
        {"file_name": "second.flac", "transcript": "second transcript",
         "instructions": {model_id: "Second voice", "unrelated": "Wrong model"}},
        {"file_name": "third.flac", "transcript": "third transcript",
         "instructions": {"unrelated": "Must not fall back"}},
        {"file_name": "fourth.flac", "transcript": "fourth transcript",
         "instructions": {model_id: ""}},
    ])
    if mode is not None:
        project.set_model_setting(model_id, "template_name", mode)
    # Simulate legacy parameter storage without asking the settings registry to
    # accept removed declarations. The adapter must never even read these keys.
    original_get = Project.get_model_setting

    def get_setting(self, owner, name):
        assert name not in {"instruct", "instruction", "instructions"}
        return original_get(self, owner, name)

    monkeypatch.setattr(Project, "get_model_setting", get_setting)
    monkeypatch.setattr(AudioCppUtil, "make_voice_ref", lambda path: str(path))
    sent = []
    monkeypatch.setattr(AudioCppUtil, "generate", lambda url, payload, **kwargs:
                        sent.append(payload) or SimpleNamespace())

    for index in (0, 1, 2, 3, 5):
        assert not isinstance(adapter.generate_using_project(
            project, ["one", "two"], voice_selection_index=index), str)
    for offset, index in enumerate((0, 1, 2, 3, 5)):
        selected = project.voice_references[index % 4]
        for payload in sent[2 * offset:2 * offset + 2]:
            assert payload["model"] == "exact-server-id"
            assert payload["voice_ref"]["data"].endswith(selected.file_name)
            assert payload.get("reference_text", "") == (selected.transcript if uses_transcript else "")
            expected = selected.instructions.get(model_id, "") if uses_instructions else ""
            assert payload["options"].get("instruction", "") == expected
            assert ("instruction" in payload["options"]) is bool(expected)
    assert support.can_instructions(project) is uses_instructions


@pytest.mark.parametrize("model_id", [OMNI, BREEZE])
def test_sampleless_selected_voice_uses_its_own_instructions(monkeypatch, model_id):
    definition = load_audio_cpp_definitions().models[model_id]
    adapter = AudioCppBackendAdapter(definition, AudioCppModelSupport(definition), "server-id")
    project = Project(tts_model_type=model_id, voice_references=[
        {"file_name": "", "instructions": {model_id: "First"}},
        {"file_name": "", "instructions": {model_id: "Second"}},
        {"file_name": "", "instructions": {"omnivoice_local": "Wrong namespace"}},
    ])
    monkeypatch.setattr(AudioCppUtil, "make_voice_ref", lambda _: pytest.fail("sample-less voice accessed"))
    sent = []
    monkeypatch.setattr(AudioCppUtil, "generate", lambda url, payload, **kwargs:
                        sent.append(payload) or SimpleNamespace())
    for index in (0, 1, 2):
        assert not isinstance(adapter.generate_using_project(project, ["hi"], voice_selection_index=index), str)
    assert [payload["options"].get("instruction") for payload in sent] == ["First", "Second", None]
    assert all("voice_ref" not in payload and "reference_text" not in payload for payload in sent)


def test_instruction_declaration_disabled_ignores_voice_data(monkeypatch):
    definition = replace(load_audio_cpp_definitions().models[OMNI], voice_instructions=False)
    support = AudioCppModelSupport(definition)
    project = Project(tts_model_type=OMNI, voice_references=[
        {"file_name": "", "instructions": {OMNI: "Ignored"}},
    ])
    sent = []
    monkeypatch.setattr(AudioCppUtil, "generate", lambda url, payload, **kwargs:
                        sent.append(payload) or SimpleNamespace())
    assert not isinstance(AudioCppBackendAdapter(definition, support, "server-id").generate_using_project(project, ["hi"]), str)
    assert "instruction" not in sent[0]["options"]
    assert not support.can_instructions(project)
