import pytest

from tts_audiobook_tool import start as start_module
from tts_audiobook_tool.constants_hints import HINT_CHATTERBOX_PACKAGE_UPDATE
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def test_startup_blocks_incompatible_chatterbox_package(monkeypatch):
    printed = []
    monkeypatch.setattr(start_module.Tts, "get_type", lambda: TtsModelType.CHATTERBOX)
    monkeypatch.setattr(
        start_module.ChatterboxApiDetect,
        "has_required_v3_features",
        lambda: False,
    )
    monkeypatch.setattr(start_module.hints, "print_hint", printed.append)

    with pytest.raises(SystemExit) as exc:
        object.__new__(start_module.Start).exit_on_incompatible_chatterbox_package()

    assert exc.value.code == 1
    assert printed == [HINT_CHATTERBOX_PACKAGE_UPDATE]


def test_startup_accepts_compatible_chatterbox_package(monkeypatch):
    monkeypatch.setattr(start_module.Tts, "get_type", lambda: TtsModelType.CHATTERBOX)
    monkeypatch.setattr(
        start_module.ChatterboxApiDetect,
        "has_required_v3_features",
        lambda: True,
    )
    monkeypatch.setattr(
        start_module.hints,
        "print_hint",
        lambda _: pytest.fail("Compatibility hint should not be printed"),
    )

    object.__new__(start_module.Start).exit_on_incompatible_chatterbox_package()


def test_startup_does_not_probe_package_for_other_model(monkeypatch):
    monkeypatch.setattr(start_module.Tts, "get_type", lambda: TtsModelType.DOTS)
    monkeypatch.setattr(
        start_module.ChatterboxApiDetect,
        "has_required_v3_features",
        lambda: pytest.fail("Chatterbox package should not be inspected"),
    )

    object.__new__(start_module.Start).exit_on_incompatible_chatterbox_package()


def test_package_update_hint_uninstalls_before_requirements_install():
    uninstall_command = "pip uninstall chatterbox-tts"
    install_command = "pip install -r requirements-chatterbox.txt"
    hint_text = HINT_CHATTERBOX_PACKAGE_UPDATE.text

    assert uninstall_command in hint_text
    assert install_command in hint_text
    assert hint_text.index(uninstall_command) < hint_text.index(install_command)
