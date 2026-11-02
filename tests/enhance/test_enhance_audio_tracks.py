"""Use the same default audio track for enhancement and review."""

from pathlib import Path

import numpy as np
import pytest

from audio_fixture_support import AudioTrack, write_aac_tracks
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.enhance.enhance_alignment import _stream_audio_with_overlap
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil


def dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    frequencies = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    return float(frequencies[np.abs(np.fft.rfft(samples)).argmax()])


@pytest.mark.requires_ffmpeg_exe(reason="enhance decoding, transcoding, and preview still use FFmpeg (Phase 3)")
def test_enhance_uses_default_audio_track_for_transcription_output_and_preview(tmp_path: Path) -> None:
    source = tmp_path / "dual.mp4"
    # The default narration is mono; the first (nondefault) track is stereo.
    write_aac_tracks(source, [AudioTrack(440, channels=2), AudioTrack(880, channels=1, default=True)])

    transcription_audio = next(_stream_audio_with_overlap(str(source)))
    assert dominant_frequency(transcription_audio, 16000) == pytest.approx(880, abs=5)

    output = tmp_path / "enhanced.m4a"
    assert SoundFileUtil.transcode_to_aac_at(str(source), str(output)) == ""
    output_audio = next(_stream_audio_with_overlap(str(output)))
    assert dominant_frequency(output_audio, 16000) == pytest.approx(880, abs=5)

    preview = SoundFileUtil.load_range(str(source), 0, 1)
    assert isinstance(preview, Sound)
    assert dominant_frequency(preview.data, preview.sr) == pytest.approx(880, abs=5)
