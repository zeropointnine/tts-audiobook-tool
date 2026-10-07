from pathlib import Path
from types import SimpleNamespace

import pytest

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.server.server import get_blocking_issues_error
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.none_base_model import NoneBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def test_blocking_issues_surfaced_as_error_text(monkeypatch, tmp_path):
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("none"))
    project = Project.model_validate({"dir_path": str(tmp_path)})

    result = get_blocking_issues_error(project, None)

    assert result.startswith("TTS model is not ready for inference:\n")
    assert "Select a TTS model for this project" in result


def test_ready_model_returns_empty_string(monkeypatch, tmp_path):
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("none"))
    monkeypatch.setattr(
        NoneBaseModel,
        "get_blocking_issues",
        classmethod(lambda cls, project, instance: []),
    )
    project = Project.model_validate({"dir_path": str(tmp_path)})

    assert get_blocking_issues_error(project, None) == ""


def test_multiple_blocking_issues_all_wrapped(monkeypatch, tmp_path):
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("none"))
    issues = [
        ReadinessIssue("voice clone", "A voice clone sample is required"),
        ReadinessIssue("server", "The inference server is unreachable"),
    ]
    monkeypatch.setattr(
        NoneBaseModel,
        "get_blocking_issues",
        classmethod(lambda cls, project, instance: issues),
    )
    project = Project.model_validate({"dir_path": str(tmp_path)})

    result = get_blocking_issues_error(project, None)

    assert result.startswith("TTS model is not ready for inference:\n")
    assert "A voice clone sample is required" in result
    assert "The inference server is unreachable" in result


def test_server_readiness_includes_unavailable_project_binding():
    project = Project(tts_model_type="glm_local")
    issue = Tts.bind_project(project)
    assert issue is not None
    assert issue.verbose in get_blocking_issues_error(project, None)
    assert project.tts_model_type == "glm_local"


@pytest.mark.parametrize(
    "model_id, transcripts, should_exit, crop",
    [
        ("higgs_v3_sglomni", [""], True, None),
        ("higgs_v3_sglomni", ["   "], True, None),
        ("higgs_v3_sglomni", ["", "second transcript"], True, None),
        ("higgs_v3_sglomni", ["first transcript", ""], False, None),
        ("higgs_v3_sglomni", [], False, None),
        ("zonos2_sglomni", [""], False, None),
        # An active crop swaps in the cropped span's transcript: the startup
        # check must validate that transcript, not the original's.
        pytest.param(
            "higgs_v3_sglomni", ["original transcript"], True,
            {"crop_transcript": "", "crop_file": True},
            id="crop-empty-transcript-exits",
        ),
        pytest.param(
            "higgs_v3_sglomni", [""], False,
            {"crop_transcript": "cropped transcript", "crop_file": True},
            id="crop-transcript-satisfies-check",
        ),
        pytest.param(
            "higgs_v3_sglomni", [""], True,
            {"crop_transcript": "cropped transcript", "crop_file": False},
            id="missing-crop-file-falls-back-to-original",
        ),
    ],
)
def test_server_startup_requires_first_voice_transcript(
    monkeypatch, tmp_path, capsys, model_id, transcripts, should_exit, crop,
):
    from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
    from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
    from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
    from tts_audiobook_tool.prefs import Prefs
    from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
    from tts_audiobook_tool.server.server import Server

    project = Project.model_validate({
        "dir_path": str(tmp_path),
        "tts_model_type": model_id,
        "voice_references": [
            {"file_name": f"voice-{index}.wav", "transcript": transcript}
            for index, transcript in enumerate(transcripts)
        ],
    })
    if crop is not None and project.voice_references:
        entry = project.voice_references[0]
        entry.update({
            "crop_file_name": "a.flac",
            "crop_start": "0.0", "crop_end": "4.0",
            "crop_transcript": crop["crop_transcript"],
        })
        if crop["crop_file"]:
            crop_path = Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry))
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop_path.write_bytes(b"x")
    monkeypatch.setattr(Prefs, "load", lambda: SimpleNamespace(
        project_dir=str(tmp_path), remote_tts_url="",
    ))
    for utility in (AudioCppUtil, RemoteTtsDiscovery, SglOmniUtil):
        monkeypatch.setattr(utility, "set_base_url", lambda url: None)
    monkeypatch.setattr(ProjectLoadUtil, "load_using_dir_path", lambda *args, **kwargs: project)

    class ModelBindingReached(Exception):
        pass

    def bind_project(*args, **kwargs):
        raise ModelBindingReached

    monkeypatch.setattr(Tts, "bind_project", bind_project)
    if should_exit:
        with pytest.raises(SystemExit) as exc:
            Server()
        assert exc.value.code == 1
        output = capsys.readouterr().out
        assert "Voice clone is missing required accompanying transcript." in output
        assert "Run the interactive app" in output
        assert "first voice clone" in output
    else:
        # Stop before model/audio initialization: passing the guard is enough.
        with pytest.raises(ModelBindingReached):
            Server()
        assert "Voice clone is missing accompanying transcript." not in capsys.readouterr().out
