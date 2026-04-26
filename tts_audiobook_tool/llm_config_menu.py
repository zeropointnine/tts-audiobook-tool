import json

from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.constants_config import DEFAULT_LLM_CONVERSATION_SYSTEM_PROMPT
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *


class LlmConfigMenu:

    @staticmethod
    def menu(state: State) -> None:

        def item_maker(_: State) -> list[MenuItem]:
            prefs = state.prefs

            sys_prompt_label = make_menu_label("System prompt", ellipsize(prefs.llm_system_prompt, 50) or "none")

            if prefs.llm_extra_params:
                extra_params_label = ellipsize(json.dumps(prefs.llm_extra_params), 50)
                extra_params_label = make_menu_label("Extra params", extra_params_label)
            else:
                extra_params_label = make_menu_label_optional("Extra params")

            return [
                MenuItem(
                    lambda _: make_menu_label("LLM endpoint URL", ellipsize(prefs.llm_url or "none", 50)),
                    lambda _, __: LlmConfigMenu.llm_url_menu(state)
                ),
                MenuItem(
                    lambda _: make_menu_label("Access token", ellipsize(prefs.llm_token, 9) if prefs.llm_token else "none"),
                    lambda _, __: LlmConfigMenu.llm_token_menu(state)
                ),
                MenuItem(
                    lambda _: make_menu_label("LLM model name", prefs.llm_model or "none"),
                    lambda _, __: LlmConfigMenu.llm_model_menu(state)
                ),
                MenuItem(
                    sys_prompt_label,
                    lambda _, __: LlmConfigMenu.llm_system_prompt_options_menu(state)
                ),
                MenuItem(
                    extra_params_label,
                    lambda _, __: LlmConfigMenu.llm_extra_params_menu(state)
                )
            ]

        subheading = (
            "Configuration for the app's LLM integration, currently used for the\n"
            "\"Realtime Conversation\" feature.\n"
        )
        MenuUtil.menu(
            state,
            "LLM configuration:",
            item_maker,
            subheading=subheading
        )

    @staticmethod
    def llm_url_menu(state: State) -> None:

        def validator(value: str) -> str:
            if not value.startswith("http://") and not value.startswith("https://"):
                return "Must start with http:// or https://"
            return ""

        AskUtil.ask_string_and_save(
            state.prefs,
            f"Enter LLM endpoint URL\n{COL_DIM}Eg, \"https://www.example.com/v1/chat/completions\" or \"https://www.example.com/v1/messages\"{COL_DEFAULT}",
            "llm_url",
            "Set LLM URL to:",
            validator=validator
        )

    @staticmethod
    def llm_token_menu(state: State) -> None:

        def validator(value: str) -> str:
            if not value.strip():
                return "Value cannot be empty"
            return ""

        AskUtil.ask_string_and_save(
            state.prefs,
            "Enter LLM token:",
            "llm_token",
            "Set LLM token to:",
            validator=validator
        )

    @staticmethod
    def llm_model_menu(state: State) -> None:

        def validator(value: str) -> str:
            if not value.strip():
                return "Value cannot be empty"
            return ""

        AskUtil.ask_string_and_save(
            state.prefs,
            f"Enter LLM model name\n{COL_DIM}Eg, \"example-model-v1\"{COL_DEFAULT}:",
            "llm_model",
            "Set LLM model name to:",
            validator=validator
        )

    @staticmethod
    def llm_system_prompt_menu(state: State) -> None:

        def validator(value: str) -> str:
            if not value.strip():
                return "Value cannot be empty"
            return ""

        AskUtil.ask_string_and_save(
            state.prefs,
            "Enter LLM system prompt:",
            "llm_system_prompt",
            "Set LLM system prompt to:",
            validator=validator
        )

    @staticmethod
    def llm_system_prompt_options_menu(state: State) -> None:

        def item_maker(_: State) -> list[MenuItem]:
            items: list[MenuItem] = [
                MenuItem(
                    "Edit system prompt",
                    lambda _, __: LlmConfigMenu.llm_system_prompt_menu(state)
                )
            ]

            default_label = "Use app default system prompt"
            if state.prefs.llm_system_prompt == DEFAULT_LLM_CONVERSATION_SYSTEM_PROMPT:
                default_label += f" {COL_ACCENT}(selected)"

            items.append(
                MenuItem(
                    default_label,
                    lambda _, __: LlmConfigMenu.use_default_system_prompt(state)
                )
            )

            if state.prefs.llm_system_prompt:
                items.append(
                    MenuItem(
                        "Clear system prompt",
                        lambda _, __: LlmConfigMenu.clear_system_prompt(state)
                    )
                )

            items.append(
                MenuItem(
                    "Print current system prompt",
                    lambda _, __: LlmConfigMenu.print_current_system_prompt(state)
                )
            )

            return items

        MenuUtil.menu(
            state,
            "LLM system prompt:",
            item_maker
        )

    @staticmethod
    def use_default_system_prompt(state: State) -> None:
        state.prefs.llm_system_prompt = DEFAULT_LLM_CONVERSATION_SYSTEM_PROMPT
        print_feedback("Set LLM system prompt to app default")

    @staticmethod
    def clear_system_prompt(state: State) -> None:
        state.prefs.llm_system_prompt = ""
        print_feedback("Cleared LLM system prompt")

    @staticmethod
    def print_current_system_prompt(state: State) -> None:
        system_prompt = state.prefs.llm_system_prompt or "None"
        s = f"{COL_DIM}Current LLM system prompt:\n"
        s += f"{COL_DEFAULT}{system_prompt}"
        printt(s)
        printt()
        AskUtil.ask_enter_to_continue()

    @staticmethod
    def llm_extra_params_menu(state: State) -> None:
        printt("Enter extra params as a JSON object:")
        printt(f"{COL_DIM}These values are merged directly into the request payload as-is.")
        printt(f"{COL_DIM}For OpenAI-style endpoints they get inserted alongside keys like \"model\", \"messages\", and \"temperature\".")
        printt(f"{COL_DIM}For Anthropic-style endpoints they get inserted alongside keys like \"model\", \"messages\", \"max_tokens\", and \"temperature\".")
        printt(f"{COL_DIM}Use this for provider-specific request fields such as reasoning or thinking options.{COL_DEFAULT}")
        printt()
        value = AskUtil.ask(lower=False)
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