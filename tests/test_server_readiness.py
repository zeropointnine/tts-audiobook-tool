from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.server.server import get_blocking_issues_error
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.none_base_model import NoneBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def test_blocking_issues_surfaced_as_error_text(monkeypatch, tmp_path):
    monkeypatch.setattr(Tts, "_type", TtsModelType.NONE)
    project = Project.model_validate({"dir_path": str(tmp_path)})

    result = get_blocking_issues_error(project, None)

    assert result.startswith("TTS model is not ready for inference:\n")
    assert "A TTS model is required" in result


def test_ready_model_returns_empty_string(monkeypatch, tmp_path):
    monkeypatch.setattr(Tts, "_type", TtsModelType.NONE)
    monkeypatch.setattr(
        NoneBaseModel,
        "get_blocking_issues",
        classmethod(lambda cls, project, instance: []),
    )
    project = Project.model_validate({"dir_path": str(tmp_path)})

    assert get_blocking_issues_error(project, None) == ""


def test_multiple_blocking_issues_all_wrapped(monkeypatch, tmp_path):
    monkeypatch.setattr(Tts, "_type", TtsModelType.NONE)
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
