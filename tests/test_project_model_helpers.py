"""Project-facing helpers use saved selection, not the bound worker model."""
from types import SimpleNamespace

from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from project_settings_test_support import set_setting


def test_project_voice_and_batch_helpers_ignore_active_model(monkeypatch):
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("none"))
    project = Project(tts_model_type="mira_local")
    set_setting(project, "mira_voice_file_name", ["narrator_mira.flac"])
    set_setting(project, "mira_batch_size", 3)
    saves = []
    monkeypatch.setattr(Project, "save", lambda current: saves.append(current) or "")

    assert ProjectVoiceUtil.has_voice(project)
    assert ProjectVoiceUtil.get_voice_label(project) == "narrator"
    assert ProjectVoiceUtil.get_batch_size(project) == 3
    ProjectVoiceUtil.set_batch_size(project, 4)
    assert ProjectVoiceUtil.get_batch_size(project) == 4
    assert saves == [project]


def test_base_voice_display_count_uses_support_model_identity(monkeypatch):
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("none"))
    project = Project(tts_model_type="mira_local")
    set_setting(project, "mira_voice_file_name", ["mira-only.flac"])
    set_setting(project, "omnivoice_voice_file_name", ["one.flac", "two.flac"])

    support = Tts.get_model_support_for_type(TtsModelType.require_by_id("omnivoice_local"))
    info = support.get_voice_display_info(project, None)
    assert info is not None
    assert "+1 more" in info.value


def test_segment_file_tag_and_fallback_voice_use_same_explicit_model(monkeypatch):
    project = Project(tts_model_type="mira_local")
    chosen = TtsModelType.require_by_id("omnivoice_local")
    looked_up = []
    monkeypatch.setattr(Tts, "get_model_support_for_type", lambda model: (
        looked_up.append(model) or SimpleNamespace(get_voice_tag=lambda current: "omnivoice-narrator")))
    monkeypatch.setattr(SegmentTranscriptUtil, "make_generation_word_error_count", lambda result: 0)
    group = PhraseGroup([Phrase("Hello", Reason.SENTENCE)])

    filename = SoundSegmentUtil.make_file_name(
        index=0, phrase_group=group, project=project,
        tts_model_type=chosen.value, validation_result=SimpleNamespace(),
        is_real_time=False,
    )

    assert looked_up == [chosen]
    assert f"[{chosen.value.file_tag}] [omnivoice-narrator]" in filename


def test_segment_prompt_uses_project_selected_support(monkeypatch):
    project = Project(tts_model_type="mira_local")
    seen = []
    monkeypatch.setattr(Tts, "get_model_support", lambda current: (
        seen.append(current) or SimpleNamespace(prepare_text_for_inference=lambda current, text: "prepared " + text)))

    assert SegmentTranscriptUtil.make_inference_prompt(project, "Hello") == "prepared Hello"
    assert seen == [project]
