from tts_audiobook_tool.conversation.conversation import ConversationStatic
from tts_audiobook_tool import ask, text_util
from tts_audiobook_tool.constants import ASSETS_DIR_NAME, CHAT_SYSTEM_PROMPTS, package_dir
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
            f"{COL_DIM}Mic input is transcribed and sent to the configured LLM, and the text\n"
            "response is generated as speech using the current TTS model settings.\n"
        )

        def make_items(_: State) -> list[MenuItem]:

            items = [
                MenuItem(make_start_label, lambda _, __: ConversationStatic.start(state)),

                MenuItem(
                    lambda _: make_menu_label("LLM system prompt", ChatMenu.get_system_prompt_label_value(state)),
                    lambda _, __: ChatMenu.system_prompt_menu(state),
                    superlabel="Options"
                ),

                MenuItem(
                    lambda _: make_menu_label(
                        "Submit prompt immediately after transcription",
                        state.prefs.conversation_stt_immediate,
                        False
                    ),
                    lambda _, __: ChatMenu.conversation_stt_immediate_menu(state),
                )
            ]

            if Tts.get_info().can_stream:
                items.append(
                    MenuItem(
                        lambda _: make_menu_label("Streaming", state.project.streaming_chat, True),
                        lambda _, __: ChatMenu.streaming_menu(state),
                    )
                )

            items.append(
                MenuItem(
                    lambda _: make_menu_label("Save output", state.prefs.chat_save),
                    lambda _, __: ChatMenu.save_menu(state)
                )
            )

            if state.prefs.chat_save:
                items.append(
                    MenuItem(
                        lambda _: make_menu_label("Save mic input as well", state.prefs.chat_save_mic),
                        lambda _, __: ChatMenu.save_mic_menu(state)
                    )
                )

            return items

        MenuUtil.menu(
            state,
            f"LLM voice chat {COL_DIM}(experimental){COL_DEFAULT}",
            make_items,
            subheading=subheading,
            hint=HINT_LLM_CHAT,
            breadcrumb="LLM voice chat",
        )

    @staticmethod
    def conversation_stt_immediate_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.prefs.conversation_stt_immediate = value
            print_feedback(
                "Submit prompt immediately after transcription set to:",
                state.prefs.conversation_stt_immediate,
            )

        MenuUtil.options_menu(
            state=state,
            heading_text="Submit prompt immediately after transcription",
            labels=["True", "False"],
            values=[True, False],
            current_value=state.prefs.conversation_stt_immediate,
            default_value=False,
            on_select=on_select,
            breadcrumb="Submit prompt immediately",
        )

    @staticmethod
    def streaming_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            state.project.streaming_chat = value
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
            print_feedback("Set to:", state.prefs.chat_save)

        dir_path = os.path.join(state.project.dir_path, PROJECT_CHAT_OUTPUT_SUBDIR)
        if os.path.exists(dir_path):
            subheading = f"Saves FLAC files to {text_util.make_terminal_hyperlink(dir_path, is_file=True)}\n"
        else:
            subheading = f"FLAC files will be saved to {text_util.make_terminal_hyperlink(dir_path)}\n"

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
        value = ask.ask(lower=False)
        if not value:
            return
        err = validator(value)
        if err:
            print_feedback(err, is_error=True)
            return
        state.prefs.llm_system_prompt = value
        state.prefs.system_prompt_preset = ""
        print_feedback("Set LLM system prompt to:", value)

    @staticmethod
    def system_prompt_menu(state: State) -> None:

        def item_maker(_: State) -> list[MenuItem]:
            edit_label = make_menu_label(
                "Edit custom system prompt",
                ellipsize(state.prefs.llm_system_prompt, 40) or "none"
            )

            items: list[MenuItem] = [
                MenuItem(
                    lambda _: make_menu_label(
                        "System prompt preset",
                        ChatMenu.get_system_prompt_preset_label_value(state)
                    ),
                    lambda _, __: ChatMenu.system_prompt_preset_menu(state)
                ),
                MenuItem(
                    edit_label,
                    lambda _, __: ChatMenu.llm_system_prompt_menu(state)
                )
            ]

            if state.prefs.llm_system_prompt or state.prefs.system_prompt_preset:
                items.append(
                    MenuItem(
                        "Clear system prompt",
                        lambda _, __: ChatMenu.clear_system_prompt(state)
                    )
                )

            items.append(
                MenuItem(
                    "Print current system prompt",
                    lambda _, __: ChatMenu.print_current_system_prompt(state),
                    superlabel=" ", superlabel_no_blank_line=True
                )
            )

            return items

        MenuUtil.menu(
            state,
            "System prompt",
            item_maker
        )

    @staticmethod
    def system_prompt_preset_menu(state: State) -> None:

        def on_select(value: str) -> None:
            state.prefs.system_prompt_preset = value
            print_feedback("Set LLM system prompt preset to:", ChatMenu.get_system_prompt_preset_label_value(state))

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
        prefs = state.prefs
        if prefs.system_prompt_preset:
            return ChatMenu.get_system_prompt_preset_label_value(state)
        return ellipsize(prefs.llm_system_prompt, 50) or "none"

    @staticmethod
    def get_system_prompt_preset_label_value(state: State) -> str:
        preset = state.prefs.system_prompt_preset
        if not preset:
            return "none"
        for file_name, label in CHAT_SYSTEM_PROMPTS:
            if file_name == preset:
                return label
        return preset

    @staticmethod
    def get_resolved_system_prompt(state: State) -> str:
        if state.prefs.system_prompt_preset:
            file_name = state.prefs.system_prompt_preset
            file_path = os.path.join(package_dir, ASSETS_DIR_NAME, file_name)
            return text_util.load_text_file(file_path)
        return state.prefs.llm_system_prompt.strip()
