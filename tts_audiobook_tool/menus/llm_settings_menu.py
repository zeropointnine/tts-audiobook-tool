import json
import re
from dataclasses import dataclass

from tts_audiobook_tool import ask
from tts_audiobook_tool.ask_advanced import AskAdvanced
from tts_audiobook_tool.constants_config import PREFS_DEFAULT_LLM_API_KEY_ENV_VAR
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *


@dataclass(frozen=True)
class ExtraParamsTarget:
    """A feature with its own LLM extra params pref."""
    feature_name: str
    prefs_attr: str
    example: str


EXTRA_PARAMS_TARGETS = (
    ExtraParamsTarget("LLM Chat", "llm_chat_extra_params", '{"thinking": {"type": "disabled"}}'),
    ExtraParamsTarget("Auto-identify characters", "llm_speaker_assign_extra_params", '{"thinking": {"type": "enabled"}}'),
)


class LlmSettingsMenu:

    @staticmethod
    def menu(state: State) -> None:

        def item_maker(_: State) -> list[MenuItem]:
            prefs = state.prefs
            items: list[MenuItem] = []

            required = f"{COL_DIM}({COL_ERROR}required{COL_DIM})"

            items.append(
                MenuItem(
                    lambda _: make_menu_label("LLM endpoint URL", ellipsize(prefs.llm_url, 50)) if prefs.llm_url else f"LLM endpoint URL {required}",
                    lambda _, __: LlmSettingsMenu.llm_url_menu(state)
                )
            )

            def api_key_label(_: State) -> str:
                label = make_menu_label("Environment variable with API key", prefs.llm_api_key_env_var)
                # Dim rather than an error: self-hosted servers may need no key
                status = ellipsize(prefs.llm_api_key, 9) if prefs.llm_api_key else "no value"
                return f"{label} {COL_DIM}({status}{COL_DIM})"

            items.append(
                MenuItem(
                    api_key_label,
                    lambda _, __: LlmSettingsMenu.api_key_menu(state)
                )
            )

            items.append(
                MenuItem(
                    lambda _: make_menu_label("LLM model name", prefs.llm_model or "none"),
                    lambda _, __: LlmSettingsMenu.llm_model_menu(state)
                )
            )

            for target in EXTRA_PARAMS_TARGETS:
                items.append(
                    MenuItem(
                        lambda _, target=target: LlmSettingsMenu.make_extra_params_label(state, target),
                        lambda _, __, target=target: LlmSettingsMenu.llm_extra_params_menu(state, target)
                    )
                )

            return items

        subheading = (
            "Configuration for the app's LLM integration, used by the \"LLM Chat\"\n"
            "and \"Auto-identify characters\" features."
        )
        MenuUtil.menu(
            state,
            "LLM settings",
            item_maker,
            subheading=subheading
        )

    @staticmethod
    def llm_url_menu(state: State) -> None:
        printt('Enter LLM endpoint URL (or "/clear" to clear):')
        printt(f'{COL_DIM}Eg, "https://www.example.com/v1/chat/completions" or "https://www.example.com/v1/messages"')
        printt()
        current = state.prefs.llm_url
        value = AskAdvanced.ask(prefill=current).strip()
        printt()
        if not value or value == current:
            return

        if value == "/clear":
            state.prefs.llm_url = ""
            state.prefs.save()
            print_feedback("Cleared LLM endpoint URL")
            return

        if not value.startswith("http://") and not value.startswith("https://"):
            ask.ask_error("Must start with http:// or https://")
            return

        state.prefs.llm_url = value
        state.prefs.save()
        print_feedback("Set LLM URL to:", value)

    @staticmethod
    def api_key_menu(state: State) -> None:
        printt("Enter the name of the environment variable which stores the value of your LLM API key")
        printt(f'{COL_DIM}Default is "{PREFS_DEFAULT_LLM_API_KEY_ENV_VAR}" (enter "/clear" to reset to default)')
        printt(f"{COL_DIM}Required by most hosted providers;")
        printt(f"{COL_DIM}may not be needed for some self-hosted servers.")
        printt()
        current = state.prefs.llm_api_key_env_var
        value = AskAdvanced.ask(prefill=current).strip()
        printt()
        if not value or value == current:
            return

        if value == "/clear":
            state.prefs.llm_api_key_env_var = PREFS_DEFAULT_LLM_API_KEY_ENV_VAR
            state.prefs.save()
            print_feedback("Reset API key environment variable to default:", PREFS_DEFAULT_LLM_API_KEY_ENV_VAR)
            return

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            ask.ask_error("Must be a valid environment variable name")
            return

        state.prefs.llm_api_key_env_var = value
        state.prefs.save()
        print_feedback("Set API key environment variable to:", value)

    @staticmethod
    def llm_model_menu(state: State) -> None:
        printt('Enter LLM model name (or "/clear" to clear):')
        printt(f'{COL_DIM}Eg, "example-model-v1"')
        printt(f"{COL_DIM}Required by most hosted providers;")
        printt(f"{COL_DIM}may be optional for some self-hosted servers.")
        printt()
        current = state.prefs.llm_model
        value = AskAdvanced.ask(prefill=current).strip()
        printt()
        if not value or value == current:
            return

        if value == "/clear":
            state.prefs.llm_model = ""
            state.prefs.save()
            print_feedback("Cleared LLM model name")
            return

        state.prefs.llm_model = value
        state.prefs.save()
        print_feedback("Set LLM model name to:", value)

    @staticmethod
    def make_extra_params_label(state: State, target: ExtraParamsTarget) -> str:
        label = f"Extra params ({target.feature_name})"
        params: dict = getattr(state.prefs, target.prefs_attr)
        if params:
            return make_menu_label(label, ellipsize(json.dumps(params), 50))
        return make_menu_label_optional(label)

    @staticmethod
    def llm_extra_params_menu(state: State, target: ExtraParamsTarget) -> None:
        printt('Enter extra params as a JSON object (or "/clear" to clear):')
        printt(f"{COL_DIM}These values are merged into the LLM request payload as-is.")
        printt(f"{COL_DIM}Use this for provider-specific fields not covered by the app.")
        printt(f"{COL_DIM}For example, using DeepSeek: {COL_MEDIUM}{target.example}")
        printt()
        params: dict = getattr(state.prefs, target.prefs_attr)
        current = json.dumps(params, ensure_ascii=False) if params else ""
        value = AskAdvanced.ask(prefill=current).strip()
        printt()
        if not value or value == current:
            return

        if value == "/clear":
            setattr(state.prefs, target.prefs_attr, {})
            state.prefs.save()
            print_feedback(f"Cleared extra params ({target.feature_name})")
            return

        try:
            parsed = json.loads(value)
        except Exception:
            ask.ask_error("Bad JSON")
            return

        if not isinstance(parsed, dict):
            ask.ask_error("JSON value must be an object")
            return

        setattr(state.prefs, target.prefs_attr, parsed)
        state.prefs.save()
        print_feedback(f"Set extra params ({target.feature_name}) to:", json.dumps(parsed, ensure_ascii=False))
