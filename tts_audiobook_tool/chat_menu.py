from tts_audiobook_tool.conversation.conversation import Conversation, ConversationStatic
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.prereqs_util import PrereqUtil
from tts_audiobook_tool.state import State
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
            f"{COL_DIM}Realtime back-and-forth chat with LLM using microphone input.\n"
            "Your speech is transcribed into prompts, with spoken responses\n"
            "generated using the app's TTS pipeline.\n"
        )

        items = [
            MenuItem(make_start_label, lambda _, __: ConversationStatic.start(state)),
            
            MenuItem(
                lambda _: make_menu_label(
                    "Submit prompt immediately after transcription",
                    state.prefs.conversation_stt_immediate,
                    False,
                ),
                lambda _, __: ChatMenu.conversation_stt_immediate_menu(state),
            )
            # TODO: Add "Options" superlabel when more than one option item exists
        ]

        MenuUtil.menu(
            state,
            f"LLM voice chat {COL_DIM}(experimental){COL_DEFAULT}",
            items,
            subheading=subheading,
            hint=HINT_LLM_CHAT
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
        )