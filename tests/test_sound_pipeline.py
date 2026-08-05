import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline


def test_should_append_break_sound_effect_skips_first_section_space_break() -> None:
    assert not SoundPipeline.should_append_break_sound_effect(
        Reason.SPACE_BREAK,
        use_break_sound_effect=True,
        is_first_in_section=True,
    )


def test_should_append_break_sound_effect_keeps_non_initial_space_break() -> None:
    assert SoundPipeline.should_append_break_sound_effect(
        Reason.SPACE_BREAK,
        use_break_sound_effect=True,
        is_first_in_section=False,
    )


def test_should_append_break_sound_effect_keeps_section_break() -> None:
    assert SoundPipeline.should_append_break_sound_effect(
        Reason.SECTION_BREAK,
        use_break_sound_effect=True,
        is_first_in_section=True,
    )


def test_append_pause_or_section_effect_uses_selected_pause_policy() -> None:
    sr = 48000
    sound = Sound(np.zeros(sr, dtype=np.float32), sr)

    normal_result = SoundPipeline.append_pause_or_section_effect(
        sound,
        reason=Reason.SPACE_BREAK,
        reason_pauses=ReasonPauseTypes.NORMAL.value,
        use_break_sound_effect=True,
        is_first_in_section=True,
    )
    shorter_result = SoundPipeline.append_pause_or_section_effect(
        sound,
        reason=Reason.SPACE_BREAK,
        reason_pauses=ReasonPauseTypes.SHORTER.value,
        use_break_sound_effect=True,
        is_first_in_section=True,
    )

    normal_pause = ReasonPauseTypes.NORMAL.value.get_pause_for(Reason.SPACE_BREAK)
    shorter_pause = ReasonPauseTypes.SHORTER.value.get_pause_for(Reason.SPACE_BREAK)
    assert normal_result.duration == sound.duration + normal_pause
    assert shorter_result.duration == sound.duration + shorter_pause
    assert shorter_result.duration < normal_result.duration
