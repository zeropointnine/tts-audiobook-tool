from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tts_audiobook_tool.app_types import HighShelfEq, Sound
from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline


@pytest.mark.parametrize("reason", [Reason.SPACE_BREAK, Reason.SECTION_BREAK])
def test_concatenate_does_not_append_break_effect_to_final_segment(reason: Reason) -> None:
    sound = Sound(np.zeros(48000, dtype=np.float32), 48000)
    phrase = Phrase("Final phrase", reason)
    process = MagicMock()

    with patch.object(ConcatUtil, "init_ffmpeg_stream", return_value=process), \
         patch.object(ConcatUtil, "close_ffmpeg_stream"), \
         patch.object(ConcatUtil, "add_audio_to_ffmpeg_stream"), \
         patch.object(SoundPipeline, "make_concat_rendered_sound_segment", return_value=sound), \
         patch.object(SoundPipeline, "append_pause_or_section_effect", return_value=sound) as append_mock:
        result = ConcatUtil.concatenate_sound_segments(
            dest_path="output.flac",
            phrases_and_paths=[(phrase, "segment.flac", False)],
            use_break_sound_effect=True,
            high_shelf=HighShelfEq.DISABLED,
            reason_pauses=ReasonPauseTypes.NORMAL.value,
            print_progress=False,
        )

    assert result == [sound.duration]
    append_mock.assert_called_once_with(
        sound,
        reason=reason,
        reason_pauses=ReasonPauseTypes.NORMAL.value,
        use_break_sound_effect=False,
        is_first_in_section=False,
        pause_duration_override=None,
    )
