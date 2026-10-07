from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import Ansi


def make_project(model: TtsModelType = TtsModelType.require_by_id("glm_local")) -> Project:
    return Project(dir_path="", tts_model_type=model.id)


def set_book_max_words(project: Project, max_words: int) -> None:
    project.book.segmentation_settings = project.book.segmentation_settings._replace(
        max_words_per_segment=max_words
    )


def test_no_warning_when_at_or_below_reco_max():
    project = make_project()
    set_book_max_words(project, 60)
    assert Tts.get_model_support(project).get_max_words_exceed_warning(project) == ""
    set_book_max_words(project, 40)
    assert Tts.get_model_support(project).get_max_words_exceed_warning(project) == ""


def test_warning_uses_proper_name_when_no_disambiguator():
    project = make_project()
    set_book_max_words(project, 80)
    result = Tts.get_model_support(project).get_max_words_exceed_warning(project)
    assert result.startswith(f"{Ansi.ITALICS}Source text's max word length (80)")
    assert "exceeds GLM-TTS recommended model limit (60)" in result
    assert result.endswith(f"{Ansi.ITALICS}Output accuracy on longer prompts may be degraded")
    assert result.count("\n") == 1


def test_warning_prefers_disambiguator_name(monkeypatch):
    project = make_project()
    set_book_max_words(project, 80)
    cls = Tts.get_model_support(project)
    monkeypatch.setattr(
        cls, "get_max_words_range_reco",
        classmethod(lambda c, project, instance=None: (40, 40, "Chatterbox Multilingual V2")),
    )
    result = cls.get_max_words_exceed_warning(project)
    assert "exceeds Chatterbox Multilingual V2 recommended model limit (40)" in result
    assert "GLM-TTS" not in result


def test_no_warning_for_model_without_reco():
    project = make_project(TtsModelType.require_by_id("none"))
    set_book_max_words(project, 80)
    assert Tts.get_model_support(project).get_max_words_exceed_warning(project) == ""


def test_base_get_warning_issues_includes_max_words_warning(monkeypatch):
    project = make_project()
    set_book_max_words(project, 80)
    cls = Tts.get_model_support(project)
    monkeypatch.setattr(cls, "_get_standard_random_voice_reason", classmethod(lambda c, p: None))
    stub_cls = type("StubModel", (cls,), {
        "generate_using_project": lambda *args, **kwargs: None,
        "kill": lambda *args, **kwargs: None,
    })
    instance = object.__new__(stub_cls)
    warnings = instance.get_warning_issues(project)
    assert len(warnings) == 1
    assert "Source text's max word length" in warnings[0]
    assert "exceeds GLM-TTS recommended model limit (60)" in warnings[0]
