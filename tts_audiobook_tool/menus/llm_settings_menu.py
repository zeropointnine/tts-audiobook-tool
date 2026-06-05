import json

from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *


class LlmSettingsMenu:

    @staticmethod
    def menu(state: State) -> None:

        def item_maker(_: State) -> list[MenuItem]:
            prefs = state.prefs
            items: list[MenuItem] = []

            required = f"{COL_DIM}({COL_ERROR}required{COL_DIM})"

            if prefs.llm_extra_params:
                extra_params_label = ellipsize(json.dumps(prefs.llm_extra_params), 50)
                extra_params_label = make_menu_label("Extra params", extra_params_label)
            else:
                extra_params_label = make_menu_label_optional("Extra params")

            items.append(
                MenuItem(
                    lambda _: make_menu_label("LLM endpoint URL", ellipsize(prefs.llm_url, 50)) if prefs.llm_url else f"LLM endpoint URL {required}",
                    lambda _, __: LlmSettingsMenu.llm_url_menu(state)
                )
            )
            if prefs.llm_url:
                items.append(MenuItem("Clear LLM endpoint URL", lambda _, __: LlmSettingsMenu.clear_llm_url(state)))

            items.append(
                MenuItem(
                    lambda _: make_menu_label("API key", ellipsize(prefs.llm_api_key, 9)) if prefs.llm_api_key else f"API key {required}",
                    lambda _, __: LlmSettingsMenu.api_key_menu(state)
                )
            )
            if prefs.llm_api_key:
                items.append(MenuItem("Clear API key", lambda _, __: LlmSettingsMenu.clear_api_key(state)))

            items.append(
                MenuItem(
                    lambda _: make_menu_label("LLM model name", prefs.llm_model or "none"),
                    lambda _, __: LlmSettingsMenu.llm_model_menu(state)
                )
            )
            if prefs.llm_model:
                items.append(MenuItem("Clear LLM model name", lambda _, __: LlmSettingsMenu.clear_llm_model(state)))

            items.append(
                MenuItem(
                    extra_params_label,
                    lambda _, __: LlmSettingsMenu.llm_extra_params_menu(state)
                )
            )
            if prefs.llm_extra_params:
                items.append(MenuItem("Clear extra params", lambda _, __: LlmSettingsMenu.clear_llm_extra_params(state)))

            return items

        subheading = (
            "Configuration for the app's LLM integration, currently used for the\n"
            "\"LLM Chat\" feature.\n"
        )
        MenuUtil.menu(
            state,
            "LLM settings",
            item_maker,
            subheading=subheading
        )

    @staticmethod
    def llm_url_menu(state: State) -> None:

        def validator(value: str) -> str:
            if not value.startswith("http://") and not value.startswith("https://"):
                return "Must start with http:// or https://"
            return ""

        ask.ask_string_and_save(
            state.prefs,
            f"Enter LLM endpoint URL\n{COL_DIM}Eg, \"https://www.example.com/v1/chat/completions\" or \"https://www.example.com/v1/messages\"{COL_DEFAULT}",
            "llm_url",
            "Set LLM URL to:",
            validator=validator
        )

    @staticmethod
    def api_key_menu(state: State) -> None:

        def validator(value: str) -> str:
            if not value.strip():
                return "Value cannot be empty"
            return ""

        ask.ask_string_and_save(
            state.prefs,
            "Enter LLM token:",
            "llm_api_key",
            "Set LLM token to:",
            validator=validator
        )

    @staticmethod
    def clear_llm_url(state: State) -> None:
        state.prefs.llm_url = ""
        print_feedback("Cleared LLM endpoint URL")

    @staticmethod
    def clear_api_key(state: State) -> None:
        state.prefs.llm_api_key = ""
        print_feedback("Cleared LLM token")

    @staticmethod
    def llm_model_menu(state: State) -> None:

        def validator(value: str) -> str:
            if not value.strip():
                return "Value cannot be empty"
            return ""

        ask.ask_string_and_save(
            state.prefs,
            f"Enter LLM model name\n{COL_DIM}Eg, \"example-model-v1\"{COL_DEFAULT}:",
            "llm_model",
            "Set LLM model name to:",
            validator=validator
        )

    @staticmethod
    def clear_llm_model(state: State) -> None:
        state.prefs.llm_model = ""
        print_feedback("Cleared LLM model name")

    @staticmethod
    def llm_extra_params_menu(state: State) -> None:
        printt("Enter extra params as a JSON object:")
        printt(f"{COL_DIM}These values are merged into the LLM request payload as-is.")
        printt(f"{COL_DIM}Use this for provider-specific fields not covered by the app.")
        printt(f'{COL_DIM}For example, using DeepSeek: {COL_MEDIUM}{{"thinking": {{"type": "disabled"}}}}')
        printt()
        value = ask.ask(lower=False)
        if not value:
            return

        try:
            parsed = json.loads(value)
        except Exception:
            print_feedback("Bad JSON", is_error=True)
            return

        if not isinstance(parsed, dict):
            print_feedback("JSON value must be an object", is_error=True)
            return

        state.prefs.llm_extra_params = parsed
        print_feedback("Set LLM extra params to:", json.dumps(parsed, ensure_ascii=False))

    @staticmethod
    def clear_llm_extra_params(state: State) -> None:
        state.prefs.llm_extra_params = {}
        print_feedback("Cleared LLM extra params")
