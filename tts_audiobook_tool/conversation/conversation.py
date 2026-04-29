from __future__ import annotations

import signal
import sys
import threading
import time

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.conversation.conversation_internals import PromptBuilder, ResponseSession, Ui
from tts_audiobook_tool.util import *

from tts_audiobook_tool.ansi import Ansi
from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.models_util import ModelsUtil
from tts_audiobook_tool.llm_util import LlmUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.whisper_realtime_util import WhisperRealTimeUtil
from tts_audiobook_tool.conversation.console_session import ConsoleSession
from tts_audiobook_tool.conversation.conversation_types import ChunkingConfig, QueuedStream
from tts_audiobook_tool.conversation.sound_input_device_util import SoundInputDeviceInfo
from tts_audiobook_tool.sound_device_stream import SoundDeviceStream


class Conversation:
    """
    Manages a single interactive voice-to-LLM conversation session.

    Orchestrates the full lifecycle: preflight checks, mic capture via
    WhisperRealTimeUtil, prompt assembly (PromptBuilder), LLM streaming and
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
        self.stt_immediate = state.prefs.conversation_stt_immediate if stt_immediate is None else stt_immediate

    @staticmethod
    def has_llm_config(state: State) -> bool:
        return bool(
            state.prefs.llm_url.strip()
            and state.prefs.api_key.strip()
            and state.prefs.llm_model.strip()
        )

    @staticmethod
    def _run_preflight_checks(state: State) -> bool:
        """
        Runs preflight checks
        Prints feedback messages along the way (errors, warnings, and info).
        Returns True for success
        """
        mic_err = SoundInputDeviceInfo.get_check_error()
        if mic_err:
            print_feedback(mic_err, is_error=True)
            return False

        missing: list[str] = []
        if not state.prefs.llm_url.strip():
            missing.append("LLM endpoint URL")
        if not state.prefs.api_key.strip():
            missing.append("LLM token")
        if not state.prefs.llm_model.strip():
            missing.append("LLM model name")
        if not Conversation.has_llm_config(state):
            print_feedback(
                "LLM Conversation Tool requires the following preferences to be set:\n- " + "\n- ".join(missing),
                is_error=True,
            )
            return False

        if state.prefs.stt_variant == SttVariant.DISABLED:
            print_feedback("Speech-to-text must be enabled (see Options menu)", is_error=True)
            return False

        did_interrupt = ModelsUtil.warm_up_models(state, never_yamnet=True)
        if did_interrupt:
            print_feedback("\nCancelled")
            return False

        prereq_errors = Tts.get_class().get_prereq_errors(
            state.project,
            Tts.get_instance_if_exists(),
            short_format=False,
        )
        if prereq_errors:
            print_feedback("\n\n".join(prereq_errors), is_error=True)
            return False

        warnings = Tts.get_instance().get_prereq_warnings(state.project)
        if warnings:
            print_feedback(Ansi.ITALICS + "\n".join(warnings), no_preformat=True)

        return True

    def start(self) -> None:
        self.print_intro()
        if not Conversation._run_preflight_checks(self.state):
            return
        self.print_input_device_info()
        self.init_session_state()
        try:
            self.install_runtime()
            self.run_main_loop()
        except KeyboardInterrupt:
            self.handle_top_level_interrupt()
        finally:
            self.teardown()

    def print_intro(self) -> None:
        print_heading("Realtime LLM Chat Tool")
        printt(f"Speak into the microphone to build your prompt.")
        printt()
        printt(f"Hotkeys:")
        printt(f"  {make_hotkey_string('Ctrl-C', outer_color=COL_DIM)} to interrupt audio or exit")
        printt()

    def print_input_device_info(self) -> None:
        printt(f"{COL_DIM}{Ansi.ITALICS}Using sound input device:")
        printt(f"{COL_DIM}{SoundInputDeviceInfo.get_input_device_description()}")

    def init_session_state(self) -> None:
        state = self.state
        system_prompt = DEFAULT_LLM_CONVERSATION_SYSTEM_PROMPT if state.prefs.llm_system_prompt_default else state.prefs.llm_system_prompt.strip()
        self.llm = LlmUtil(
            api_endpoint_url=state.prefs.llm_url,
            token=state.prefs.api_key,
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
        self.console = ConsoleSession.create(real_stdout=self.real_stdout, real_stderr=self.real_stderr)
        self.ctrl_c_requested = threading.Event()
        self.prompt_builder = PromptBuilder(
            ui=self.ui,
            console=self.console,
            ctrl_c_requested=self.ctrl_c_requested,
            stt_immediate=self.stt_immediate,
        )
        self.util = WhisperRealTimeUtil(
            prefs=self.prefs,
            on_transcription=self.prompt_builder.on_transcription,
        )

        self.sound_stream = SoundDeviceStream(APP_SAMPLE_RATE)
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
        self.console.start()
        self.real_stdout.write(Ansi.CURSOR_HIDE)
        self.real_stdout.flush()
        self.util.start()
        self.sound_stream.start()
        self.ui.println()

    def on_sigint(self, *_: object) -> None:
        self.ctrl_c_requested.set()
        if self.exiting:
            return
        if self.in_response:
            raise KeyboardInterrupt

    def run_main_loop(self) -> None:
        while True:
            assembled = self.prompt_builder.build()
            self.run_response_turn(assembled)

    def run_response_turn(self, assembled: str) -> None:
        self.util.pause()
        self.session = ResponseSession(
            ui=self.ui,
            llm=self.llm,
            project=self.project,
            chunking_config=self.chunking_config,
            stt_variant=self.prefs.stt_variant,
            stt_config=self.prefs.stt_config,
            real_stderr=self.real_stderr,
            fd2_redirect_lock=self.fd2_redirect_lock,
            ctrl_c_requested=self.ctrl_c_requested,
            phrase_stt_enabled=self.phrase_stt_enabled,
            sound_stream=self.sound_stream,
        )
        self.in_response = True
        try:
            self.session.run(assembled)
            # Keep mic input paused briefly after playback so room/output
            # tail audio does not immediately get re-captured and
            # transcribed as the next user prompt. Flush both before and
            # after the settle window to discard buffered STT state.
            self.util.flush()
            settle_s = max(0.25, min(0.6, self.sound_stream.output_latency + 0.1))
            time.sleep(settle_s)
            self.util.flush()
        finally:
            self.in_response = False
            self.util.resume()
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
        print_feedback("Exited")
        print("", end="", flush=True)
        self.stop_capture()

    def teardown(self) -> None:
        self.exiting = True
        # Restore stdio first so any further prints (from util.stop,
        # sound_stream.shut_down, etc.) bypass the queue and don't risk
        # deadlocking ui.wait_idle() on never-processed task_done.
        self.restore_console()
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
        self.console.restore()

    def stop_capture(self) -> None:
        if self.capture_stopped:
            return
        self.capture_stopped = True
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
