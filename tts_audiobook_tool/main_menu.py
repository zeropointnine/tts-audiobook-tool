from tts_audiobook_tool.concat_submenu import ConcatSubmenu
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.options_submenu import OptionsSubmenu
from tts_audiobook_tool.generate_submenu import GenerateSubmenu
from tts_audiobook_tool.project_submenu import ProjectSubmenu
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.text_submenu import TextSubmenu
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.state import State

# TODO some import above is triggering unsightly warning from "jieba", either directly or transitively

class MainMenu:
    """
    Main menu and misc submenus
    """

    @staticmethod
    def menu(state: State, did_reset=False) -> None:

        MainMenu._print_menu(state, did_reset)

        SigIntHandler().set("menu") # not great
        hotkey = ask_hotkey()
        if SigIntHandler().did_interrupt:
            SigIntHandler().clear()
            MainMenu.quit()

        if not hotkey:
            return
        MainMenu._handle_menu_hotkey(hotkey, state)

    @staticmethod
    def _print_menu(state: State, did_reset: bool):

        # Title
        s = f"{COL_DIM}(active model: {COL_ACCENT}{Tts.get_type().value.ui['proper_name']}{COL_DIM})"
        print_heading(f"{APP_NAME} {s}")

        # Dir check
        if did_reset:
            printt(f"{COL_ERROR}Directory {state.prefs.project_dir} not found.\nCleared project settings.\n")

        num_generated = state.project.sound_segments.num_generated()

        # Project
        s = f"{make_hotkey_string('P')} Project directory "
        if not state.prefs.project_dir:
            s += f"{COL_ERROR}(required){COL_DEFAULT}"

        else:
            s += make_currently_string(state.prefs.project_dir)
        printt(s)

        # Voice
        if state.prefs.project_dir and Tts.get_type() != TtsModelInfos.NONE:
            voice_label = state.project.get_voice_label()
            if not state.project.can_voice:
                current = f"{COL_ERROR}(required){COL_DEFAULT} "
            else:
                current = make_currently_string(voice_label, label="current voice clone: ")
            printt(f"{make_hotkey_string('V')} Voice clone and model settings {current}")

        # Text
        if state.prefs.project_dir:
            if state.project.text_segments:
                lines = "line" if len(state.project.text_segments) == 1 else "lines"
                value = f"{len(state.project.text_segments)} {lines}"
                current = make_currently_string(value)
            else:
                current = f"{COL_ERROR}(required){COL_DEFAULT}"
            s = f"{make_hotkey_string('T')} Text {current}"
            printt(s)

        # Generate audio
        if state.prefs.project_dir and Tts.get_type() != TtsModelInfos.NONE:
            s = f"{make_hotkey_string('G')} Generate audio"
            if not state.project.can_voice and not state.project.text_segments:
                s2 = f"{COL_ERROR} (requires text and voice sample)"
            elif not state.project.can_voice:
                s2 = f"{COL_ERROR} (requires voice sample)"
            elif not state.project.text_segments:
                s2 = f"{COL_ERROR} (requires text)"
            else:
                s2 = f" {COL_DIM}({COL_ACCENT}{num_generated}{COL_DIM} of {COL_ACCENT}{len(state.project.text_segments)}{COL_DIM} lines complete)"
            printt(s + s2)

        # Concat
        if state.prefs.project_dir:
            s = f"{make_hotkey_string('C')} Concatenate audio segments to create audiobook file"
            if num_generated == 0:
                s += f" {COL_ERROR}(must first generate audio)"
            printt(s)

        # Options
        printt(f"{make_hotkey_string('O')} Options/Tools")

        # Quit
        printt(f"{make_hotkey_string('Q')} Quit")

        printt()


    @staticmethod
    def _handle_menu_hotkey(hotkey: str, state: State) -> None:

        # Preempt potential directory scan
        if hotkey == "q":
            MainMenu.quit()

        num_generated = state.project.sound_segments.num_generated()

        match hotkey:
            case "p":
                ProjectSubmenu.submenu(state)
            case "v":
                if state.prefs.project_dir:
                    match Tts.get_type():
                        case TtsModelInfos.OUTE:
                            from tts_audiobook_tool.voice_oute_submenu import VoiceOuteSubmenu
                            VoiceOuteSubmenu.submenu(state.project)
                        case TtsModelInfos.CHATTERBOX:
                            from tts_audiobook_tool.voice_chatterbox_submenu import VoiceChatterboxSubmenu
                            VoiceChatterboxSubmenu.submenu(state.project)
                        case TtsModelInfos.FISH:
                            from tts_audiobook_tool.voice_fish_submenu import VoiceFishSubmenu
                            VoiceFishSubmenu.submenu(state.project)
                        case TtsModelInfos.HIGGS:
                            from tts_audiobook_tool.voice_higgs_submenu import VoiceHiggsSubmenu
                            VoiceHiggsSubmenu.submenu(state.project)
                        case TtsModelInfos.VIBEVOICE:
                            from tts_audiobook_tool.voice_vibevoice_submenu import VoiceVibeVoiceSubmenu
                            VoiceVibeVoiceSubmenu.submenu(state.project)
                        case TtsModelInfos.INDEXTTS2:
                            from tts_audiobook_tool.voice_indextts2_submenu import VoiceIndexTts2Submenu
                            VoiceIndexTts2Submenu.submenu(state.project)
            case "t":
                if not state.prefs.project_dir:
                    return
                if not state.project.text_segments:
                    TextSubmenu.set_text_submenu(state, "Set text:")
                else:
                    TextSubmenu.submenu(state)
            case "g":
                if state.project.can_generate_audio:
                    GenerateSubmenu.submenu(state)
            case "c":
                if not state.prefs.project_dir or num_generated == 0:
                    return
                ConcatSubmenu.submenu(state)
            case "o":
                OptionsSubmenu.submenu(state)


    @staticmethod
    def quit():
        printt("State saved. Exiting.")
        exit(0)
