from types import SimpleNamespace
from typing import cast

from tts_audiobook_tool.constants_config import PREFS_DEFAULT_LLM_API_KEY_ENV_VAR
from tts_audiobook_tool.menus import llm_settings_menu
from tts_audiobook_tool.menus.llm_settings_menu import EXTRA_PARAMS_TARGETS, LlmSettingsMenu
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.state import State

CHAT, SPEAKER_ASSIGN = EXTRA_PARAMS_TARGETS


def make_state() -> State:
    prefs = Prefs(
        llm_chat_extra_params={"thinking": {"type": "disabled"}},
        llm_speaker_assign_extra_params={"thinking": {"type": "enabled"}},
    )
    return cast(State, SimpleNamespace(prefs=prefs))


def test_extra_params_prompt_sets_only_its_own_feature_value(monkeypatch) -> None:
    state = make_state()
    monkeypatch.setattr(llm_settings_menu.AskAdvanced, "ask", lambda message="", prefill="": '{"top_k": 3}')

    LlmSettingsMenu.llm_extra_params_menu(state, CHAT)

    assert state.prefs.llm_chat_extra_params == {"top_k": 3}
    assert state.prefs.llm_speaker_assign_extra_params == {"thinking": {"type": "enabled"}}
    assert Prefs.load().llm_chat_extra_params == {"top_k": 3}


def test_extra_params_prompt_clear_command_empties_only_its_own_feature_value(monkeypatch) -> None:
    # "/clear" (in place of the removed "Clear extra params" item) resets to {}
    state = make_state()
    monkeypatch.setattr(llm_settings_menu.AskAdvanced, "ask", lambda message="", prefill="": "/clear")

    LlmSettingsMenu.llm_extra_params_menu(state, SPEAKER_ASSIGN)

    assert state.prefs.llm_speaker_assign_extra_params == {}
    assert state.prefs.llm_chat_extra_params == {"thinking": {"type": "disabled"}}
    assert Prefs.load().llm_speaker_assign_extra_params == {}


def test_extra_params_prompt_prefills_current_value(monkeypatch) -> None:
    # The current value is offered for editing; accepting it unchanged is a no-op
    state = make_state()
    prefills: list[str] = []

    def fake_ask(message: str = "", prefill: str = "") -> str:
        prefills.append(prefill)
        return prefill

    monkeypatch.setattr(llm_settings_menu.AskAdvanced, "ask", fake_ask)

    LlmSettingsMenu.llm_extra_params_menu(state, CHAT)

    assert prefills == ['{"thinking": {"type": "disabled"}}']
    assert state.prefs.llm_chat_extra_params == {"thinking": {"type": "disabled"}}


def test_extra_params_prompt_empty_input_keeps_value(monkeypatch) -> None:
    state = make_state()
    monkeypatch.setattr(llm_settings_menu.AskAdvanced, "ask", lambda message="", prefill="": "")

    LlmSettingsMenu.llm_extra_params_menu(state, CHAT)

    assert state.prefs.llm_chat_extra_params == {"thinking": {"type": "disabled"}}


def test_model_name_prompt_sets_and_clears(monkeypatch) -> None:
    # The model name is set from the prompt, and "/clear" (in place of the
    # removed "Clear LLM model name" item) empties it
    state = cast(State, SimpleNamespace(prefs=Prefs(llm_model="old-model")))
    prefills: list[str] = []

    def answer(value: str):
        def fake_ask(message: str = "", prefill: str = "") -> str:
            prefills.append(prefill)
            return value
        return fake_ask

    monkeypatch.setattr(llm_settings_menu.AskAdvanced, "ask", answer("new-model"))
    LlmSettingsMenu.llm_model_menu(state)
    assert state.prefs.llm_model == "new-model"
    assert Prefs.load().llm_model == "new-model"

    monkeypatch.setattr(llm_settings_menu.AskAdvanced, "ask", answer("/clear"))
    LlmSettingsMenu.llm_model_menu(state)
    assert state.prefs.llm_model == ""
    assert Prefs.load().llm_model == ""
    assert prefills == ["old-model", "new-model"]


def test_url_prompt_sets_rejects_and_clears(monkeypatch) -> None:
    # A valid URL is saved, a non-http(s) one is rejected (keeping the old
    # value), and "/clear" (in place of the removed menu item) empties it
    state = cast(State, SimpleNamespace(prefs=Prefs(llm_url="https://old.example/v1/chat/completions")))
    monkeypatch.setattr(llm_settings_menu.ask, "ask_error", lambda *_args, **_kwargs: None)

    def answer(value: str) -> None:
        monkeypatch.setattr(llm_settings_menu.AskAdvanced, "ask", lambda message="", prefill="": value)
        LlmSettingsMenu.llm_url_menu(state)

    answer("https://new.example/v1/messages")
    assert Prefs.load().llm_url == "https://new.example/v1/messages"

    answer("new.example/v1/messages")
    assert state.prefs.llm_url == "https://new.example/v1/messages"

    answer("/clear")
    assert state.prefs.llm_url == ""
    assert Prefs.load().llm_url == ""


def test_api_key_env_var_prompt_sets_rejects_and_resets(monkeypatch) -> None:
    # A valid name is saved, an invalid one is rejected (keeping the old
    # value), and "/clear" (in place of the removed menu item) resets to the default
    state = cast(State, SimpleNamespace(prefs=Prefs()))
    monkeypatch.setattr(llm_settings_menu.ask, "ask_error", lambda *_args, **_kwargs: None)

    def answer(value: str) -> None:
        monkeypatch.setattr(llm_settings_menu.AskAdvanced, "ask", lambda message="", prefill="": value)
        LlmSettingsMenu.api_key_menu(state)

    answer("MY_PROVIDER_KEY")
    assert Prefs.load().llm_api_key_env_var == "MY_PROVIDER_KEY"

    answer("not a name")
    assert state.prefs.llm_api_key_env_var == "MY_PROVIDER_KEY"

    answer("/clear")
    assert state.prefs.llm_api_key_env_var == PREFS_DEFAULT_LLM_API_KEY_ENV_VAR
    assert Prefs.load().llm_api_key_env_var == PREFS_DEFAULT_LLM_API_KEY_ENV_VAR
