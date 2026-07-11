from __future__ import annotations

import signal
import sys
import threading
import time

from tts_audiobook_tool import ask
from tts_audiobook_tool import app_support
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.conversation.conversation_internals import PromptBuilder, ResponseSession, Ui
from tts_audiobook_tool.app_support import app_hint_util, app_memory
from tts_audiobook_tool import readiness
from tts_audiobook_tool.util import *

from tts_audiobook_tool.system_support.ansi import Ansi
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.conversation.realtime_transcriber import RealtimeTranscriber
from tts_audiobook_tool.conversation.console_session import ConsoleSession
from tts_audiobook_tool.conversation.conversation_types import ChunkingConfig, QueuedStream
from tts_audiobook_tool.conversation.sound_input_device_util import SoundInputDeviceInfo
from tts_audiobook_tool.sound.sound_device_stream import SoundDeviceStream


class Conversation:
    """
    "Realtime LLM Chat"

    Orchestrates the full lifecycle: preflight checks, mic capture via
    RealtimeTranscriber, prompt assembly (PromptBuilder), LLM streaming and
    TTS playback (ResponseSession), and clean teardown of the terminal/audio
    state on exit or Ctrl-C.
    """

    def __init__(
        self,
        state: State,
        phrase_stt_enabled: bool = True,
        stt_immediate: bool | None = None,
    ) -> None:
        # phrase_stt_enabled: runs a secondary STT pass over the TTS output audio to
        # produce phrase-level timing segments (via make_phrase_spoken_segments). When
        # disabled, each response chunk is treated as a single unsegmented span. The
        # extra STT pass adds latency and a failure path — if it returns no segments the
        # code falls back gracefully, but bad timing data can cause subtle sync issues.
        self.state = state
        self.phrase_stt_enabled = phrase_stt_enabled
        self.chat_input_mode = state.prefs.chat_input_mode
        self.is_text_input = self.chat_input_mode == CHAT_INPUT_MODE_TEXT
        if stt_immediate is None:
            self.stt_immediate = self.chat_input_mode == CHAT_INPUT_MODE_MIC_IMMEDIATE
        else:
            self.stt_immediate = stt_immediate

    def start(self) -> None:

        if not Conversation._run_preflight_checks(self.state):
            if self.state.prefs.menu_clears_screen:
                ask.ask_enter_to_continue()
            return

        self.print_various()

        self.init_session_state()
        # Conversation is a top-level session. Do not inherit continuation
        # history from earlier app flows or previous chat sessions.
        Tts.clear_continuation()
        Tts.reset_voice_rotation_index()
        try:
            self.install_runtime()
            self.run_main_loop()
        except KeyboardInterrupt:
            self.handle_top_level_interrupt()
        finally:
            self.teardown()

    def print_various(self) -> None:

        # Warnings
        warnings = Tts.get_instance().get_warning_issues(self.state.project)
        if warnings:
            s = "\n".join(warnings)
            printt(f"{COL_DIM_ITALICS}{s}")
            printt()

        if self.is_text_input:
            printt(f"Type a prompt and press {make_hotkey_string('Enter', outer_color=COL_DIM)} to submit")
            printt()
        else:
            # Mic info
            printt(f"{COL_DIM_ITALICS}Using sound input device:")
            printt(f"{COL_DIM_ITALICS}  {SoundInputDeviceInfo.get_input_device_description()}")
            printt()

            # Instructions
            if not self.stt_immediate:
                printt(f"Speak into the microphone to build your prompt:")
                printt(f"  {make_hotkey_string('Left/Right', outer_color=COL_DIM)} - select transcribed phrase")
                printt(f"  {make_hotkey_string('Delete', outer_color=COL_DIM)} - delete selected phrase")
                printt(f"  {make_hotkey_string('Enter', outer_color=COL_DIM)} - submit phrase")
                printt()
        printt(f"  {make_hotkey_string('Ctrl-C', outer_color=COL_DIM)} to interrupt audio or exit")
        printt()

    @staticmethod
    def _run_preflight_checks(state: State) -> bool:
        """
        Runs preflight checks
        Prints feedback messages along the way (errors, warnings, and info).
        Returns True for success
        """

        # Get blocking issues
        errors = readiness.get_chat_blockers(state)
        if errors:
            lines = [item.verbose for item in errors]
            s = "\n".join(lines)
            print_feedback(s, is_error=True)
            return False

        can_continue = app_hint_util.show_pre_inference_hints(state.prefs, state.project)
        if not can_continue:
            return False

        # Warm up models
        warm_up_result = ModelManager.warm_up_models(state, skip_yamnet=True)
        if warm_up_result.should_stop:
            app_support.print_warm_up_result_stop(warm_up_result)
            if warm_up_result.error:
                app_memory.gc_ram_vram()
            return False

        # Must check for TTS model blockers again because instance is guaranteed to exist now
        model_errors = Tts.get_class().get_blocking_issues(state.project, Tts.get_instance_if_exists()) 
        if model_errors:
            model_error_string = readiness.format_issues(model_errors, verbose=True)
            print_feedback(model_error_string, is_error=True)
            return False

        return True

    def init_session_state(self) -> None:
        state = self.state
        from tts_audiobook_tool.menus.chat_menu import ChatMenu
        system_prompt = ChatMenu.get_resolved_system_prompt(state)
        self.llm = LlmSession(
            api_endpoint_url=state.prefs.llm_url,
            token=state.prefs.llm_api_key,
            model=state.prefs.llm_model,
            system_prompt=system_prompt,
            extra_params=state.prefs.llm_extra_params,
            verbose=False,
        )
        self.prefs = state.prefs
        self.project = state.project
        self.chunking_config = ChunkingConfig(language_code=state.project.language_code)

        self.real_stdout, self.real_stderr = sys.stdout, sys.stderr
        self.fd2_redirect_lock = threading.Lock()
        self.ui = Ui(self.real_stdout)
        self.ctrl_c_requested = threading.Event()
        self.console: ConsoleSession | None = None
        self.prompt_builder: PromptBuilder | None = None
        self.util: RealtimeTranscriber | None = None
        if not self.is_text_input:
            self.console = ConsoleSession.create(real_stdout=self.real_stdout, real_stderr=self.real_stderr)
            self.prompt_builder = PromptBuilder(
                ui=self.ui,
                console=self.console,
                ctrl_c_requested=self.ctrl_c_requested,
                stt_immediate=self.stt_immediate,
            )
            silence_duration_s = (
                CHAT_SILENCE_THRESHOLD_IMMEDIATE
                if self.stt_immediate
                else CHAT_SILENCE_THRESHOLD_CHUNKED
            )
            self.util = RealtimeTranscriber(
                prefs=self.prefs,
                on_transcription=self.prompt_builder.on_transcription,
                silence_duration_s=silence_duration_s,
            )

        # Note, when streaming, output sound device samplerate is that of the
        # native samplerate of the TTS engine because we skip any post-processing.
        # Otherwise, use the app default samplerate for the general-purpose
        # post-processing chain.
        use_streaming_tts = Tts.get_info().can_stream and self.state.project.streaming_chat
        sr = Tts.get_info().sample_rate if use_streaming_tts else APP_SAMPLE_RATE
        self.sound_stream = SoundDeviceStream(sr)

        self.session: ResponseSession | None = None
        self.in_response = False
        self.exiting = False
        self.old_input_sigint = None
        self.console_restored = False
        self.capture_stopped = False

    def install_runtime(self) -> None:
        self.old_input_sigint = signal.signal(signal.SIGINT, self.on_sigint)
        self.ui.start()
        # Route every print on every thread (audio callback, whisper,
        # third-party libs, ...) through the UI queue so nothing races
        # with the cursor-relative render model in the UI worker.
        sys.stdout = QueuedStream(self.real_stdout, self.ui.queue)
        sys.stderr = QueuedStream(self.real_stderr, self.ui.queue)
        if self.console is not None:
            self.console.start()
            self.real_stdout.write(Ansi.CURSOR_HIDE)
            self.real_stdout.flush()
        if self.util is not None:
            self.util.start()
        self.sound_stream.start()
        self.ui.println()

    def on_sigint(self, *_: object) -> None:
        self.ctrl_c_requested.set()
        if self.exiting:
            return
        if self.in_response or self.is_text_input:
            raise KeyboardInterrupt

    def run_main_loop(self) -> None:
        while True:
            assembled = self.build_prompt()
            self.run_response_turn(assembled)

    def build_prompt(self) -> str:
        if self.is_text_input:
            return self.build_text_prompt()
        if self.prompt_builder is None:
            raise RuntimeError("Missing microphone prompt builder")
        return self.prompt_builder.build()

    def build_text_prompt(self) -> str:
        self.ui.wait_idle()
        self.real_stdout.write(Ansi.CURSOR_SHOW)
        self.real_stdout.write("> ")
        self.real_stdout.flush()
        try:
            text = input().strip()
        except EOFError:
            raise KeyboardInterrupt
        finally:
            self.real_stdout.write(Ansi.CURSOR_HIDE)
            self.real_stdout.flush()
        self.real_stdout.write("\n")
        self.real_stdout.flush()
        return text

    def run_response_turn(self, assembled: str) -> None:
        if self.util is not None:
            self.util.pause()
        user_input_sound = None
        if self.prefs.chat_save_mic and self.prompt_builder is not None:
            mic_audio = self.prompt_builder.take_finalized_mic_audio()
            if mic_audio is not None and mic_audio.size > 0:
                user_input_sound = Sound(mic_audio, WHISPER_SAMPLERATE)
        self.session = ResponseSession(
            ui=self.ui,
            llm=self.llm,
            state=self.state,
            chunking_config=self.chunking_config,
            stt_variant=self.prefs.stt_variant,
            stt_config=self.prefs.stt_config,
            real_stderr=self.real_stderr,
            fd2_redirect_lock=self.fd2_redirect_lock,
            ctrl_c_requested=self.ctrl_c_requested,
            phrase_stt_enabled=self.phrase_stt_enabled,
            sound_stream=self.sound_stream,
        )
        if user_input_sound is not None:
            self.session.user_input_sound = user_input_sound
            self.session.save_chat_mic_input_if_needed(assembled)
        self.in_response = True
        try:
            self.session.run(assembled, user_input_sound=user_input_sound)
            # Keep mic input paused briefly after playback so room/output
            # tail audio does not immediately get re-captured and
            # transcribed as the next user prompt. Flush both before and
            # after the settle window to discard buffered STT state.
            if self.util is not None:
                self.util.flush()
            settle_s = max(0.25, min(0.6, self.sound_stream.output_latency + 0.1))
            time.sleep(settle_s)
            if self.util is not None:
                self.util.flush()
        finally:
            self.in_response = False
            if self.util is not None:
                self.util.resume()
        if self.prompt_builder is not None:
            self.prompt_builder.resume()

    def handle_top_level_interrupt(self) -> None:
        self.exiting = True
        if self.in_response:
            # Ctrl-C raised after ResponseSession already cleaned up its
            # own interrupt (or during the post-playback settle). Swallow
            # it and exit start() cleanly via the finally block.
            self.ctrl_c_requested.clear()
            return
        # The realtime STT worker may currently be inside a blocking
        # transcribe() call, so stopping capture can take a noticeable
        # amount of time while its worker thread unwinds. Restore the
        # console and show exit feedback first so Ctrl-C feels immediate.
        self.ui.commit_render(1)
        self.ui.wait_idle()
        self.restore_console()
        print_feedback("Exiting LLM Chat")
        print("", end="", flush=True)
        self.stop_capture()

    def teardown(self) -> None:
        self.exiting = True
        # Restore stdio first so any further prints (from util.stop,
        # sound_stream.shut_down, etc.) bypass the queue and don't risk
        # deadlocking ui.wait_idle() on never-processed task_done.
        self.restore_console()
        model = Tts.get_instance_if_exists()
        if model is not None:
            model.clear_stream_state()
            model.clear_continuation()
        self.ui.stop()
        self.stop_capture()
        self.sound_stream.shut_down()

    def restore_console(self) -> None:
        if self.console_restored:
            return
        self.console_restored = True

        # Put stdio/tty back before any final user-facing prints so the
        # parent shell/terminal can immediately resume normal prompting.
        sys.stdout, sys.stderr = self.real_stdout, self.real_stderr
        try:
            self.real_stdout.write(Ansi.CURSOR_SHOW)
            self.real_stdout.flush()
        except Exception:
            pass
        if self.old_input_sigint is not None:
            signal.signal(signal.SIGINT, self.old_input_sigint)
        if self.console is not None:
            self.console.restore()

    def stop_capture(self) -> None:
        if self.capture_stopped:
            return
        self.capture_stopped = True
        if self.util is None:
            return
        try:
            self.util.stop()
        except Exception:
            pass


class ConversationStatic:

    @staticmethod
    def start(
        state: State,
        phrase_stt_enabled: bool = True,
        stt_immediate: bool | None = None,
    ) -> None:
        Conversation(
            state,
            phrase_stt_enabled=phrase_stt_enabled,
            stt_immediate=stt_immediate,
        ).start()
