import subprocess
from unittest.mock import patch

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import APP_SAMPLE_RATE, FFMPEG_COMMAND
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil


def test_load_range_seeks_and_caps_decode_to_one_minute() -> None:
    samples = np.array([0.25, -0.5], dtype=np.float32)
    with patch(
        "tts_audiobook_tool.sound.sound_file_util.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, samples.tobytes(), b""),
    ) as run:
        result = SoundFileUtil.load_range("book.m4b", 123.25, 500.5)
    assert isinstance(result, Sound)
    assert result.sr == APP_SAMPLE_RATE
    np.testing.assert_array_equal(result.data, samples)
    command = run.call_args.args[0]
    assert command[0] == FFMPEG_COMMAND
    assert command[command.index("-ss") + 1] == "123.25"
    assert command[command.index("-i") + 1] == "book.m4b"
    assert command[command.index("-t") + 1] == "60.0"
    assert command[command.index("-ar") + 1] == str(APP_SAMPLE_RATE)
    assert "-map" not in command  # Follow the output file's default audio track.
    assert all(option in command for option in ("-vn", "-sn", "-dn"))
    assert command[-1] == "pipe:1"
    assert run.call_args.kwargs == {"capture_output": True, "check": True}


def test_load_range_short_interval_and_empty_audio() -> None:
    with patch(
        "tts_audiobook_tool.sound.sound_file_util.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, b"", b""),
    ) as run:
        assert SoundFileUtil.load_range("book.mp3", 2.5, 3.75) == (
            "No audio was found in the selected interval"
        )
    assert run.call_args.args[0][run.call_args.args[0].index("-t") + 1] == "1.25"


def test_load_range_reports_decoder_errors_and_skips_invalid_intervals() -> None:
    with patch("tts_audiobook_tool.sound.sound_file_util.subprocess.run") as run:
        assert isinstance(SoundFileUtil.load_range("book.mp3", 1, 1), str)
        assert isinstance(SoundFileUtil.load_range("book.mp3", -1, 5), str)
        run.assert_not_called()
        run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"missing file\n")
        assert SoundFileUtil.load_range("missing.mp3", 1, 2) == "missing file"
        run.side_effect = FileNotFoundError("ffmpeg not found")
        assert "ffmpeg not found" in SoundFileUtil.load_range("book.mp3", 1, 2)
