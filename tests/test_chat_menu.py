from types import SimpleNamespace

import pytest

from tts_audiobook_tool.app_types import VoiceSelectMode
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from tts_audiobook_tool import text_util
from tts_audiobook_tool.conversation.conversation_types import ChatInputMode
from tts_audiobook_tool.menus import chat_menu
from tts_audiobook_tool.menus.chat_menu import ChatMenu


def test_echo_override_is_orthogonal_to_chat_input_mode(monkeypatch) -> None:
    saves: list[bool] = []
    prefs = SimpleNamespace(
        chat_input_mode=ChatInputMode.MIC_IMMEDIATE,
        chat_echo_override=False,
        save=lambda: saves.append(True),
    )
    state = SimpleNamespace(prefs=prefs)
    captured: dict = {}
    echo_options: dict = {}

    def menu(**kwargs) -> None:
        captured.update(kwargs)

    def options_menu(**kwargs) -> None:
        echo_options.update(kwargs)

    monkeypatch.setattr(chat_menu.MenuUtil, "menu", menu)
    monkeypatch.setattr(chat_menu.MenuUtil, "options_menu", options_menu)
    monkeypatch.setattr(chat_menu, "print_feedback", lambda *_args: None)

    ChatMenu.input_mode_menu(state)

    items = captured["items"]
    labels = [text_util.strip_ansi_codes(item.label) for item in items]
    assert labels == [
        "Microphone, submit immediately after silence (default) (selected)",
        "Microphone, submit by pressing ENTER",
        "Text input",
        "Echo text only (no LLM) (currently: False default)",
    ]

    echo_item = items[3]
    echo_item.handler(state, echo_item)
    assert prefs.chat_echo_override is False
    assert echo_options["heading_text"] == "Echo text only (no LLM)"
    assert echo_options["current_value"] is False

    echo_options["on_select"](True)
    assert prefs.chat_echo_override is True
    assert prefs.chat_input_mode is ChatInputMode.MIC_IMMEDIATE
    assert ChatMenu.get_chat_input_mode_label_value(state) == "microphone, echo only"

    text_item = items[2]
    text_item.handler(state, text_item)
    assert prefs.chat_input_mode is ChatInputMode.TEXT
    assert prefs.chat_echo_override is True
    assert ChatMenu.get_chat_input_mode_label_value(state) == "text, echo only"
    assert saves == [True, True]


@pytest.mark.parametrize(
    ("mode", "expected_scope"),
    [(VoiceSelectMode.DISABLED, [0]), (VoiceSelectMode.USER_DEFINED, [0]),
     (VoiceSelectMode.AUTO_ADVANCE, [0, 1])],
)
@pytest.mark.parametrize("valid", [True, False])
def test_chat_preflight_uses_literal_scope(monkeypatch, mode, expected_scope, valid) -> None:
    from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared

    state = SimpleNamespace(
        project=SimpleNamespace(
            get_tts_model_type=lambda: TtsModelType.require_by_id("qwen3tts_local"),
            voice_references=[Voice(), Voice()],
            voice_select_mode=mode,
            # Chat must not inherit the project's per-line voice assignment.
            phrase_groups=[SimpleNamespace(voice_index=1)],
        ),
        prefs=SimpleNamespace(chat_save=False),
    )
    scopes: list[list[int]] = []
    launches: list[object] = []

    def validate(_state, *, voice_indices):
        assert _state is state
        scopes.append(list(voice_indices))
        return valid

    def menu(passed_state, _heading, make_items, **_kwargs):
        item = make_items(passed_state)[0]
        item.handler(passed_state, item)

    monkeypatch.setattr(chat_menu.MenuUtil, "menu", menu)
    monkeypatch.setattr(chat_menu.Tts, "get_info", lambda _project: SimpleNamespace(can_stream=False))
    monkeypatch.setattr(VoiceMenuShared, "validate_voices", staticmethod(validate))
    monkeypatch.setattr(chat_menu, "run_conversation_app", launches.append)

    ChatMenu.menu(state)

    assert scopes == [expected_scope]
    assert launches == ([state] if valid else [])
