from tts_audiobook_tool.conversation.conversation import Conversation, ConversationStatic
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *


class ConversationMenu:

    @staticmethod
    def menu(state: State) -> None:
        subheading = (
            f"{COL_DIM}Speak into your microphone to build prompts for a realtime back-and-forth\n"
            "conversation with an LLM, with spoken responses generated through the app's\n"
            "TTS pipeline.\n\n"
            "Works best when the TTS model inferences faster than 100% realtime speed\n"
            "(or ideally much faster).\n"
        )
        if not Conversation.has_llm_config(state):
            subheading += f"\n{COL_DEFAULT}LLM settings must first be configured from the Options menu.\n"

        items = [
            MenuItem("Start", lambda _, __: ConversationStatic.start(state)),
            MenuItem(
                lambda _: make_menu_label(
                    "Submit prompt immediately after transcription",
                    state.prefs.conversation_stt_immediate,
                    False,
                ),
                lambda _, __: ConversationMenu.conversation_stt_immediate_menu(state),
            ),
        ]

        MenuUtil.menu(
            state,
            f"Realtime LLM Chat {COL_DIM}(experimental){COL_DEFAULT}",
            items,
            subheading=subheading,
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