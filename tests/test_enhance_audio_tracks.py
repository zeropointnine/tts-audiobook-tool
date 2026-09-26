"""Use the same default audio track for enhancement and review."""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import FFMPEG_COMMAND
from tts_audiobook_tool.enhance.enhance_alignment import _stream_audio_with_overlap
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil


def dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    frequencies = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    return float(frequencies[np.abs(np.fft.rfft(samples)).argmax()])


def test_enhance_uses_default_audio_track_for_transcription_output_and_preview(tmp_path: Path) -> None:
    if shutil.which(FFMPEG_COMMAND) is None:
        pytest.skip("FFmpeg is unavailable")

    source = tmp_path / "dual.mp4"
    subprocess.run(
        [
            FFMPEG_COMMAND, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=16000:duration=1",
            "-map", "0:a", "-map", "1:a", "-c:a", "aac",
            # The default narration is mono; the first (nondefault) track is stereo.
            "-ac:a:0", "2", "-ac:a:1", "1",
            "-disposition:a:0", "0", "-disposition:a:1", "default",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    transcription_audio = next(_stream_audio_with_overlap(str(source)))
    assert dominant_frequency(transcription_audio, 16000) == pytest.approx(880, abs=5)

    output = tmp_path / "enhanced.m4a"
    assert SoundFileUtil.transcode_to_aac_at(str(source), str(output)) == ""
    output_audio = next(_stream_audio_with_overlap(str(output)))
    assert dominant_frequency(output_audio, 16000) == pytest.approx(880, abs=5)

    preview = SoundFileUtil.load_range(str(source), 0, 1)
    assert isinstance(preview, Sound)
    assert dominant_frequency(preview.data, preview.sr) == pytest.approx(880, abs=5)
