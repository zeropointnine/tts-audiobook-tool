import threading
from unittest.mock import patch

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil


class FakeOutputStream:
    release_event = threading.Event()

    def __init__(self, **kwargs):
        self.finished_callback = kwargs["finished_callback"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def reset_playback_state() -> None:
    FakeOutputStream.release_event.clear()
    PlaySoundUtil.stop_sound_async()
    with PlaySoundUtil._playback_state_lock:
        PlaySoundUtil._current_playback_thread = None
        PlaySoundUtil._stop_playback_event = None
        PlaySoundUtil._current_sound_id = ""


def test_play_sound_async_returns_current_id_until_playback_finishes() -> None:
    reset_playback_state()

    class CompletingOutputStream(FakeOutputStream):
        def __enter__(self):
            def finish():
                self.release_event.wait(timeout=1)
                self.finished_callback()

            threading.Thread(target=finish, daemon=True).start()
            return self

    sound = Sound(np.zeros(10, dtype=np.float32), 48000)
    with patch("tts_audiobook_tool.sound.play_sound_util.sd.OutputStream", CompletingOutputStream):
        sound_id = PlaySoundUtil.play_sound_async(sound)
        assert sound_id
        assert PlaySoundUtil.current_sound_id() == sound_id

        CompletingOutputStream.release_event.set()
        thread = PlaySoundUtil._current_playback_thread
        assert thread is not None
        thread.join(timeout=1)

    assert PlaySoundUtil.current_sound_id() == ""


def test_play_sound_file_async_returns_id_and_blank_error() -> None:
    reset_playback_state()
    sound = Sound(np.zeros(10, dtype=np.float32), 48000)

    with (
        patch.object(PlaySoundUtil, "play_sound_async", return_value="sound-id"),
        patch("tts_audiobook_tool.sound.play_sound_util.SoundFileUtil.load", return_value=sound),
    ):
        assert PlaySoundUtil.play_sound_file_async("sound.wav") == ("sound-id", "")


def test_play_sound_file_async_returns_blank_id_and_load_error() -> None:
    with patch("tts_audiobook_tool.sound.play_sound_util.SoundFileUtil.load", return_value="load failed"):
        assert PlaySoundUtil.play_sound_file_async("sound.wav") == ("", "load failed")
