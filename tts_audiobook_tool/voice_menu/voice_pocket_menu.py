from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.voice_menu import VoiceMenuShared


class VoicePocketMenu:

    @staticmethod
    def menu(state: State) -> None:

        def make_language_label(_) -> str:
            lang = state.project.pocket_model_code
            if lang:
                label = make_currently_string(lang)
            else:
                label = make_currently_string(PocketBaseModel.DEFAULT_LANGUAGE)
            return f"Pocket model {label}"

        def make_predefined_voice_label(_: State) -> str:
            name = state.project.pocket_predefined_voice
            currently = make_currently_string(name) if name else ""
            return f"Select predefined voice {currently}".strip()

        def make_voice_file_label(_: State) -> str:
            value = state.project.pocket_voice_file_name
            if value:
                currently = make_currently_string(ellipsize_path_for_menu(value.removesuffix("_pocket.flac")))
            else:
                currently = ""
            return f"Select voice clone sample {currently}".strip()

        def make_items(_: State) -> list[MenuItem]:

            items = []

            items.append(
                MenuItem(
                    make_voice_file_label,
                    lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.POCKET)
                )
            )
            items.append(
                MenuItem(
                    make_predefined_voice_label,
                    lambda _, __: predefined(state),
                )
            )
            if state.project.pocket_voice_file_name or state.project.pocket_predefined_voice:
                items.append(VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.POCKET))

            items.append(MenuItem(make_language_label, lambda _, __: ask_language(state)))

            item = VoiceMenuShared.make_temperature_item(
                state=state,
                attr="pocket_temperature",
                default_value=PocketBaseModel.DEFAULT_TEMPERATURE,
                min_value=PocketBaseModel.TEMPERATURE_MIN,
                max_value=PocketBaseModel.TEMPERATURE_MAX
            )
            item.superlabel = VOICE_ADVANCED_SUPERLABEL
            items.append(item)

            items.append(
                VoiceMenuShared.make_seed_item(state, "pocket_seed")
            )

            return items

        VoiceMenuShared.menu_wrapper(state, make_items)

# ---

def predefined(state: State) -> None:
    voices = PocketBaseModel.PREDEFINED_VOICES

    current = state.project.pocket_predefined_voice or None

    def on_select(voice: str) -> None:
        with state.project.batch():
            state.project.pocket_predefined_voice = voice
            state.project.pocket_voice_file_name = ""

    MenuUtil.options_menu(
        state=state,
        heading_text="Select predefined voice",
        labels=voices,
        values=voices,
        current_value=current,
        default_value=None,
        on_select=on_select,
    )

def ask_language(state: State) -> None:

    languages = PocketBaseModel.LANGUAGES
    default = PocketBaseModel.DEFAULT_LANGUAGE

    def on_select(lang: str) -> None:

        previous = state.project.pocket_model_code
        if lang == previous:
            return

        # Validation step arguably non-essential since we're using a controlled, hard-list, but yea
        printt(f"{COL_DIM}{Ansi.ITALICS}Validating... ")
        printt()
        state.project.pocket_model_code = lang
        state.project.save()
        Tts.set_model_params_using_project(state.project)
        Tts.clear_tts_model()
        try:
            Tts.get_pocket()
        except Exception as e:
            state.project.pocket_model_code = previous
            state.project.save()
            Tts.set_model_params_using_project(state.project)
            AskUtil.ask_error(f"\n{make_error_string(e)}")
        printt(f"{COL_DIM}{Ansi.ITALICS}OK ")
        printt()

    MenuUtil.options_menu(
        state=state,
        heading_text="Select Pocket TTS model",
        labels=languages,
        values=languages,
        subheading="Ensure your voice clone audio language matches the model you select below\n",
        current_value=state.project.pocket_model_code or default,
        default_value=default,
        on_select=on_select,
    )
