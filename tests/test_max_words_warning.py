from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import Ansi


def make_project() -> Project:
    return Project(dir_path="")


def preserve_tts_state():
    return {
        "had_tts_type": hasattr(Tts, "_type"),
        "tts_type": getattr(Tts, "_type", None),
    }


def restore_tts_state(saved) -> None:
    if saved["had_tts_type"]:
        Tts._type = saved["tts_type"]
    else:
        delattr(Tts, "_type")


def test_no_warning_when_at_or_below_reco_max():
    saved = preserve_tts_state()
    try:
        # GLM reco is (40, 60, "")
        Tts._type = TtsModelType.GLM
        project = make_project()
        project.applied_max_words = 60
        assert Tts.get_class().get_max_words_exceed_warning(project) == ""
        project.applied_max_words = 40
        assert Tts.get_class().get_max_words_exceed_warning(project) == ""
    finally:
        restore_tts_state(saved)


def test_warning_uses_proper_name_when_no_disambiguator():
    saved = preserve_tts_state()
    try:
        # GLM reco max is 60
        Tts._type = TtsModelType.GLM
        project = make_project()
        project.applied_max_words = 80

        result = Tts.get_class().get_max_words_exceed_warning(project)
        assert result.startswith(f"{Ansi.ITALICS}Source text's max word length (80)")
        assert "exceeds GLM-TTS recommended model limit (60)" in result
        assert result.endswith(f"{Ansi.ITALICS}Output accuracy may be degraded")
        assert result.count("\n") == 1
    finally:
        restore_tts_state(saved)


def test_warning_prefers_disambiguator_name(monkeypatch):
    saved = preserve_tts_state()
    try:
        Tts._type = TtsModelType.GLM
        project = make_project()
        project.applied_max_words = 80

        cls = Tts.get_class()
        monkeypatch.setattr(
            cls,
            "get_max_words_range_reco",
            classmethod(lambda c, project, instance=None: (40, 40, "Chatterbox Multilingual V2")),
        )

        result = cls.get_max_words_exceed_warning(project)
        assert "exceeds Chatterbox Multilingual V2 recommended model limit (40)" in result
        assert "GLM-TTS" not in result
    finally:
        restore_tts_state(saved)


def test_no_warning_for_model_without_reco():
    saved = preserve_tts_state()
    try:
        # NONE model reco is (0, 0, "")
        Tts._type = TtsModelType.NONE
        project = make_project()
        project.applied_max_words = 80
        assert Tts.get_class().get_max_words_exceed_warning(project) == ""
    finally:
        restore_tts_state(saved)


def test_base_get_warning_issues_includes_max_words_warning(monkeypatch):
    saved = preserve_tts_state()
    try:
        # GLM reco max is 60
        Tts._type = TtsModelType.GLM
        project = make_project()
        project.applied_max_words = 80

        cls = Tts.get_class()
        monkeypatch.setattr(
            cls,
            "_get_standard_random_voice_reason",
            classmethod(lambda c, p: None),
        )

        # Concrete stub subclass so the abstract instance method can be called
        stub_cls = type("StubModel", (cls,), {
            "generate_using_project": lambda *args, **kwargs: None,
            "kill": lambda *args, **kwargs: None,
        })
        instance = object.__new__(stub_cls)
        warnings = instance.get_warning_issues(project)
        assert len(warnings) == 1
        assert "Source text's max word length" in warnings[0]
        assert "exceeds GLM-TTS recommended model limit (60)" in warnings[0]
    finally:
        restore_tts_state(saved)
