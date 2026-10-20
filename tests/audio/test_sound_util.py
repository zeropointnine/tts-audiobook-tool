import math

import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.sound.sound_util import SoundUtil


@pytest.mark.parametrize("sr", [16000, 24000, 44100, 48000])
@pytest.mark.parametrize("direction", [-math.inf, math.inf])
def test_sample_conversion_ignores_float_drift(sr, direction):
    seconds = 0.1
    drifted = math.nextafter(seconds, direction)
    assert SoundUtil.seconds_to_sample_index(drifted, sr) == round(seconds * sr)


@pytest.mark.parametrize("samples,expected", [(3.25, 3), (3.75, 4), (4.25, 4), (4.75, 5)])
def test_sample_conversion_rounds_to_nearest_sample(samples, expected):
    assert SoundUtil.seconds_to_sample_index(samples / 24000, 24000) == expected


@pytest.mark.parametrize("sr", [16000, 24000, 44100, 48000])
def test_trim_uses_rounded_boundaries_and_copies_data(sr):
    sound = Sound(np.arange(sr, dtype=np.float32), sr)
    start = math.nextafter(0.1, -math.inf)
    end = math.nextafter(0.3, -math.inf)
    trimmed = SoundUtil.trim(sound, start, end)
    np.testing.assert_array_equal(trimmed.data, sound.data[round(0.1 * sr):round(0.3 * sr)])
    assert not np.shares_memory(trimmed.data, sound.data)


def test_trim_defaults_and_end_clamping():
    sound = Sound(np.arange(10, dtype=np.float32), 10)
    np.testing.assert_array_equal(SoundUtil.trim(sound, None, None).data, sound.data)
    np.testing.assert_array_equal(SoundUtil.trim(sound, 0.3, 2.0).data, sound.data[3:])


@pytest.mark.parametrize("start,end", [(-0.1, 0.5), (0.5, 0.5), (0.6, 0.5), (1.0, 2.0)])
def test_trim_rejects_invalid_sample_ranges(start, end):
    sound = Sound(np.arange(10, dtype=np.float32), 10)
    with pytest.raises(ValueError, match="Invalid trim range"):
        SoundUtil.trim(sound, start, end)
