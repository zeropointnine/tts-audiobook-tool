import os

from tts_audiobook_tool.conversation.chat_config import (
    get_chat_input_mode_label,
    get_resolved_system_prompt,
    get_system_prompt_label,
    get_system_prompt_preset_label,
)
from tts_audiobook_tool.conversation.conversation_types import ChatInputMode
from tts_audiobook_tool.textual.conversation_app import run_conversation_app
from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.constants import CHAT_SYSTEM_PROMPTS
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool import readiness
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *


class ChatMenu:
    @staticmethod
    def menu(state: State) -> None:

        def make_start_label(state: State) -> str:
            blocker = readiness.get_chat_blocker_text(state, verbose=False)
            label = "Start"
            if blocker:
                label += f" {COL_DIM}({COL_ERROR}{blocker}{COL_DIM})"
            return label

        subheading = (
            f"Interactive chat with a configured LLM where each assistant reply is\n"
            "generated as speech using the current TTS model and voice settings.\n"
        )

        def start_conversation(_: State, __: MenuItem) -> None:
            run_conversation_app(state)

        def make_items(_: State) -> list[MenuItem]:

            items = [
                MenuItem(make_start_label, start_conversation),
                MenuItem(
                    lambda _: make_menu_label(
                        "Input mode", ChatMenu.get_chat_input_mode_label_value(state)
                    ),
                    lambda _, __: ChatMenu.input_mode_menu(state)
                ),
                MenuItem(
                    lambda _: make_menu_label(
                        "LLM system prompt",
                        ChatMenu.get_system_prompt_label_value(state),
                    ),
                    lambda _, __: ChatMenu.system_prompt_menu(state),
                    superlabel="Options",
                ),
            ]

            if Tts.get_info().can_stream:
                items.append(
                    MenuItem(
                        lambda _: make_menu_label(
                            "Streaming", state.project.streaming_chat, True
                        ),
                        lambda _, __: ChatMenu.streaming_menu(state),
                    )
                )

            items.append(
                MenuItem(
                    lambda _: make_menu_label("Save output", state.prefs.chat_save),
                    lambda _, __: ChatMenu.save_menu(state),
                )
            )

            if state.prefs.chat_save:
                items.append(
                    MenuItem(
                        lambda _: make_menu_label(
                            "Save mic input as well", state.prefs.chat_save_mic
                        ),
                        lambda _, __: ChatMenu.save_mic_menu(state),
                    )
                )

            return items

        MenuUtil.menu(
            state,
            f"LLM voice chat",
            make_items,
            subheading=subheading,
            hint=HINT_LLM_CHAT,
            breadcrumb="LLM voice chat",
        )

    @staticmethod
    def input_mode_menu(state: State) -> None:

        def select_input_mode(_: State, item: MenuItem) -> None:
            value = item.data
            if not isinstance(value, ChatInputMode) or value == state.prefs.chat_input_mode:
                return
            state.prefs.chat_input_mode = value
            state.prefs.save()
            print_feedback(
                "Input mode set to:",
                ChatMenu.get_chat_input_mode_label_value(state),
            )

        mode_items = [
            (
                "Microphone, submit immediately after silence",
                ChatInputMode.MIC_IMMEDIATE,
            ),
            ("Microphone, submit by pressing ENTER", ChatInputMode.MIC_ENTER),
            ("Text input", ChatInputMode.TEXT),
        ]
        items: list[MenuItem] = []
        for label, mode in mode_items:
            if mode == PREFS_DEFAULT_CHAT_INPUT_MODE:
                label += f" {COL_DIM}(default)"
            if mode == state.prefs.chat_input_mode:
                label += f" {COL_ACCENT}(selected)"
            items.append(MenuItem(label, select_input_mode, data=mode))

        echo_label = (
            "Echo text only (no LLM) "
            + make_currently_string(
                state.prefs.chat_echo_override,
                default=PREFS_DEFAULT_CHAT_ECHO_OVERRIDE,
            )
        )
        items.append(
            MenuItem(
                echo_label,
                lambda _, __: ChatMenu.echo_override_menu(state),
                superlabel=" ", superlabel_no_blank_line=True
            )
        )

        MenuUtil.menu(
            state=state,
            heading="Input mode",
            items=items,
            one_shot=True,
            breadcrumb="Input mode",
        )

    @staticmethod
    def echo_override_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.prefs.chat_echo_override = value
            state.prefs.save()
            print_feedback("Echo text only (no LLM) set to:", value)

        MenuUtil.options_menu(
            state=state,
            heading_text="Echo text only (no LLM)",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.prefs.chat_echo_override,
            default_value=PREFS_DEFAULT_CHAT_ECHO_OVERRIDE,
            on_select=on_select,
            breadcrumb="Echo text only",
        )

    @staticmethod
    def streaming_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.streaming_chat = value
            state.project.save()
            print_feedback("Streaming set to:", state.project.streaming_chat)

        subheading = (
            f"{COL_DIM}Streaming outputs audio as it is generated, instead of\n"
            "waiting for the normal full generation path to finish. This gives\n"
            "minimal response latency, but skips the usual post-processing.\n"
        )

        MenuUtil.options_menu(
            state=state,
            heading_text="Streaming",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.project.streaming_chat,
            default_value=True,
            on_select=on_select,
            subheading=subheading,
            breadcrumb="Streaming",
        )

    @staticmethod
    def save_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.prefs.chat_save = value
            state.prefs.save()
            print_feedback("Set to:", state.prefs.chat_save)

        dir_path = os.path.join(state.project.dir_path, PROJECT_CHAT_OUTPUT_SUBDIR)
        if os.path.exists(dir_path):
            path_text = text_util.make_terminal_hyperlink(dir_path, is_file=True)
        else:
            path_text = dir_path
        subheading = (
            f"Saves sound segment FLAC files generated while using LLM Chat will be saved to\n"
            f"{path_text}\n"
        )

        MenuUtil.options_menu(
            state=state,
            heading_text="Save output to files",
            subheading=subheading,
            labels=["True", "False"],
            values=[True, False],
            current_value=state.prefs.chat_save,
            default_value=PROJECT_DEFAULT_CHAT_SAVE,
            on_select=on_select,
            breadcrumb="Save output",
        )

    @staticmethod
    def save_mic_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.prefs.chat_save_mic = value
            state.prefs.save()
            print_feedback("Set to:", state.prefs.chat_save_mic)

        MenuUtil.options_menu(
            state=state,
            heading_text="Save mic input as well",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.prefs.chat_save_mic,
            default_value=PROJECT_DEFAULT_CHAT_SAVE_MIC,
            on_select=on_select,
            breadcrumb="Save mic input",
        )

    # ---

    @staticmethod
    def llm_system_prompt_menu(state: State) -> None:

        def validator(value: str) -> str:
            if not value.strip():
                return "Value cannot be empty"
            return ""

        printt("Enter LLM system prompt:")
        value = ask.ask_input(lower=False)
        if not value:
            return
        err = validator(value)
        if err:
            ask.ask_error(err)
            return
        state.prefs.llm_system_prompt = value
        state.prefs.system_prompt_preset = ""
        state.prefs.save()
        print_feedback("Set LLM system prompt to:", value)

    @staticmethod
    def system_prompt_menu(state: State) -> None:

        def item_maker(_: State) -> list[MenuItem]:
            edit_label = make_menu_label(
                "Edit custom system prompt",
                ellipsize(state.prefs.llm_system_prompt, 40) or "none",
            )

            items: list[MenuItem] = [
                MenuItem(
                    lambda _: make_menu_label(
                        "System prompt preset",
                        ChatMenu.get_system_prompt_preset_label_value(state),
                    ),
                    lambda _, __: ChatMenu.system_prompt_preset_menu(state),
                ),
                MenuItem(
                    edit_label, lambda _, __: ChatMenu.llm_system_prompt_menu(state)
                ),
            ]

            if state.prefs.llm_system_prompt or state.prefs.system_prompt_preset:
                items.append(
                    MenuItem(
                        "Clear system prompt",
                        lambda _, __: ChatMenu.clear_system_prompt(state),
                    )
                )

            items.append(
                MenuItem(
                    "Print current system prompt",
                    lambda _, __: ChatMenu.print_current_system_prompt(state),
                    superlabel=" ", superlabel_no_blank_line=True,
                )
            )

            return items

        MenuUtil.menu(state, "System prompt", item_maker)

    @staticmethod
    def system_prompt_preset_menu(state: State) -> None:

        def on_select(value: str) -> None:
            state.prefs.system_prompt_preset = value
            state.prefs.save()
            print_feedback(
                "Set LLM system prompt preset to:",
                ChatMenu.get_system_prompt_preset_label_value(state),
            )

        labels = ["None"] + [label for _, label in CHAT_SYSTEM_PROMPTS]
        values = [""] + [file_name for file_name, _ in CHAT_SYSTEM_PROMPTS]

        MenuUtil.options_menu(
            state=state,
            heading_text="System prompt preset",
            labels=labels,
            values=values,
            current_value=state.prefs.system_prompt_preset,
            default_value=CHAT_SYSTEM_PROMPTS[0][0],
            on_select=on_select,
            breadcrumb="System prompt preset",
        )

    @staticmethod
    def clear_system_prompt(state: State) -> None:
        state.prefs.llm_system_prompt = ""
        state.prefs.system_prompt_preset = ""
        state.prefs.save()
        print_feedback("Cleared LLM system prompt")

    @staticmethod
    def print_current_system_prompt(state: State) -> None:
        system_prompt = ChatMenu.get_resolved_system_prompt(state) or "None"
        s = f"{COL_DIM}Current LLM system prompt:\n\n"
        s += f"{COL_DEFAULT}{system_prompt}"
        printt(s)
        printt()
        ask.ask_enter_to_continue()

    @staticmethod
    def get_system_prompt_label_value(state: State) -> str:
        return get_system_prompt_label(state)

    @staticmethod
    def get_chat_input_mode_label_value(state: State) -> str:
        label = get_chat_input_mode_label(state.prefs.chat_input_mode)
        if state.prefs.chat_echo_override:
            label += ", echo only"
        return label

    @staticmethod
    def get_system_prompt_preset_label_value(state: State) -> str:
        return get_system_prompt_preset_label(state.prefs.system_prompt_preset)

    @staticmethod
    def get_resolved_system_prompt(state: State) -> str:
        return get_resolved_system_prompt(state)
