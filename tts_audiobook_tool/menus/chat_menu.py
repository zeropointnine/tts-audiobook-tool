from tts_audiobook_tool.conversation.conversation import ConversationStatic
from tts_audiobook_tool.constants_hints import *
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.prereqs_util import PrereqUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *


class ChatMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_start_label(state: State) -> str:
            prereq = PrereqUtil.get_chat_prereq_error_string(state, verbose=False)
            label = "Start"
            if prereq:
                label += f" {COL_DIM}({COL_ERROR}{prereq}{COL_DIM})"
            return label

        subheading = (
            f"{COL_DIM}Mic input is transcribed and sent to the configured LLM, and the text\n"
            "response is generated as speech using the current TTS model settings.\n"
        )

        has_more_than_one_option = Tts.get_info().can_stream

        items = [
            MenuItem(make_start_label, lambda _, __: ConversationStatic.start(state)),
            
            MenuItem(
                lambda _: make_menu_label(
                    "Submit prompt immediately after transcription",
                    state.prefs.conversation_stt_immediate,
                    False
                ),
                lambda _, __: ChatMenu.conversation_stt_immediate_menu(state),
                superlabel="Options" if has_more_than_one_option else ""
            )
        ]

        if Tts.get_info().can_stream:
            items.append(
                MenuItem(
                    lambda _: make_menu_label("Streaming", state.project.streaming_chat, True),
                    lambda _, __: ChatMenu.streaming_menu(state),
                )
            )

        MenuUtil.menu(
            state,
            f"LLM voice chat {COL_DIM}(experimental){COL_DEFAULT}",
            items,
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