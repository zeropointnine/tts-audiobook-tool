from __future__ import annotations

import queue
from typing import Callable

from tts_audiobook_tool.app_types import Sound, SttConfig, SttVariant
from tts_audiobook_tool.conversation.conversation_internals import (
    ConversationStreamingTts,
    _ResponseEngine,
)
from tts_audiobook_tool.conversation.conversation_types import (
    ChunkingConfig,
    ResponseResult,
)
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.l import L
from tts_audiobook_tool.sound.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.state import State
from tts_audiobook_tool import text_util


class _NotificationSink:
    """UI-neutral sink which records explicit runtime notices only."""

    def __init__(self, on_error: Callable[[str], None] | None) -> None:
        self.on_error = on_error
        self.errors: queue.SimpleQueue[str] = queue.SimpleQueue()

    def println(self, text: str = "") -> None:
        plain = text_util.strip_ansi_codes(text).strip()
        if not plain:
            return
        self.errors.put(plain)
        if self.on_error is not None:
            self.on_error(plain)


class ResponseSession(_ResponseEngine):
    """One response turn with snapshots and cooperative thread-safe interruption.

    The audio/chunking algorithms remain shared with the established response
    implementation, while terminal rendering is replaced by structured state.
    """

    def __init__(
        self,
        *,
        llm: LlmSession | None,
        state: State,
        chunking_config: ChunkingConfig,
        stt_variant: SttVariant,
        stt_config: SttConfig,
        sound_stream: SoundDeviceStream,
        phrase_stt_enabled: bool = True,
        echo_override: bool = False,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._notification_sink = _NotificationSink(on_error)
        super().__init__(
            ui=self._notification_sink,
            llm=llm,
            state=state,
            chunking_config=chunking_config,
            stt_variant=stt_variant,
            stt_config=stt_config,
            sound_stream=sound_stream,
            phrase_stt_enabled=phrase_stt_enabled,
            echo_override=echo_override,
        )

    def run(  # type: ignore[reportIncompatibleMethodOverride]
        self, assembled: str, user_input_sound: Sound | None = None
    ) -> ResponseResult:
        super().run(assembled, user_input_sound=user_input_sound)
        snapshot = self.snapshot()
        return ResponseResult(
            text=self.render_response_text(),
            interrupted=snapshot.interrupted,
        )

    def request_interrupt(self) -> None:
        """Log the request and delegate cancellation to the engine's public API."""
        L.i(
            "[chat] response: interrupt requested "
            "(stopping HTTP and TTS; buffered playback drains until the turn ends)"
        )
        super().request_interrupt()

    def take_errors(self) -> list[str]:
        errors: list[str] = []
        while True:
            try:
                errors.append(self._notification_sink.errors.get_nowait())
            except queue.Empty:
                return errors


__all__ = ["ConversationStreamingTts", "ResponseSession"]
