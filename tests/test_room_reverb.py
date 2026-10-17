"""
Room reverb: the streaming processor, the per-voice `room_reverb` setting (and
project-wide `room_reverb_stereo`), and its concat wiring.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tts_audiobook_tool.app_types import HighShelfEq, RoomReverb, RoomReverbStereo, Sound
from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.sound.noise_bed import NoiseBed
from tts_audiobook_tool.sound.room_reverb import (
    ROOM_DESIGNS, RoomReverbProcessor, StreamingReverb, make_room_ir, make_stereo_irs, split_ir,
)
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline

SR = 48_000


def energy(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(np.sum(x * x))


def noise(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal(round(seconds * SR))).astype(np.float32)


class TestImpulseResponse:

    @pytest.mark.parametrize("design", list(ROOM_DESIGNS))
    def test_unit_energy_and_deterministic(self, design: str) -> None:
        # Unit energy is what makes a preset's DRR mean what it says; a fixed
        # seed keeps re-exports identical
        a = make_room_ir(ROOM_DESIGNS[design], SR)
        b = make_room_ir(ROOM_DESIGNS[design], SR)
        assert energy(a) == pytest.approx(1.0, rel=1e-9)
        np.testing.assert_array_equal(a, b)

    def test_nothing_arrives_before_the_predelay(self) -> None:
        p = ROOM_DESIGNS["small"]
        ir = make_room_ir(p, SR)
        # Band-limit filters smear a little, so allow a sliver of energy early
        assert energy(ir[:round(p.predelay_ms / 1000 * SR * 0.9)]) < 1e-6

    def test_every_preset_names_a_known_design(self) -> None:
        for preset in RoomReverb:
            assert (preset.design is None) == (preset is RoomReverb.OFF)
            if preset.design is not None:
                assert preset.design in ROOM_DESIGNS
                assert preset.drr_db is not None


class TestStreamingReverb:

    def test_chunked_output_equals_one_long_call(self) -> None:
        # The tail carries across calls, so segment-sized chunks leave no seams
        ir = make_room_ir(ROOM_DESIGNS["small"], SR)
        x = noise(1.0)
        whole = StreamingReverb(ir, 12.0).process(x)
        r = StreamingReverb(ir, 12.0)
        chunked = np.concatenate([r.process(x[a:b]) for a, b in ((0, 300), (300, 301), (301, 9000), (9000, len(x)))])
        np.testing.assert_allclose(chunked, whole, atol=1e-12)

    def test_drr_sets_the_reverb_level(self) -> None:
        # For an impulse, dry energy over wet energy is the DRR (here the IR
        # fits inside the call, so no tail is dropped)
        ir = make_room_ir(ROOM_DESIGNS["small"], SR)
        impulse = np.zeros(len(ir) + 10)
        impulse[0] = 1.0
        out = StreamingReverb(ir, 20.0).process(impulse)
        wet = out - impulse
        assert 10 * np.log10(1.0 / energy(wet)) == pytest.approx(20.0, abs=0.01)

    def test_samples_past_send_end_stay_dry(self) -> None:
        # The send only covers the narration; what follows (eg a break effect)
        # gets the earlier tail added but adds no reverb of its own
        ir = make_room_ir(ROOM_DESIGNS["small"], SR)
        speech, effect = noise(0.5, 1), noise(0.5, 2)
        with_effect = StreamingReverb(ir, 10.0).process(np.concatenate([speech, effect]), send_end=len(speech))
        silent = StreamingReverb(ir, 10.0).process(np.concatenate([speech, np.zeros_like(effect)]))
        np.testing.assert_allclose(with_effect[len(speech):] - effect, silent[len(speech):], atol=1e-7)


class TestProcessor:

    def test_same_length_and_tail_reaches_into_following_chunk(self) -> None:
        reverb = RoomReverbProcessor(SR, "medium", 10.0)
        speech = reverb.apply(Sound(noise(0.5), SR))
        pause = reverb.apply(Sound(np.zeros(SR // 2, dtype=np.float32), SR))
        assert len(speech.data) == SR // 2 and len(pause.data) == SR // 2
        assert speech.data.dtype == np.float32
        assert energy(pause.data[:SR // 10]) > 0

    def test_is_last_fades_the_reverb_to_zero_but_not_the_dry_signal(self) -> None:
        reverb = RoomReverbProcessor(SR, "medium", 0.0)
        dry = noise(0.5)
        out = reverb.apply(Sound(dry, SR), is_last=True).data
        assert out[-1] == pytest.approx(dry[-1], abs=1e-6)
        assert not np.allclose(out[-1000:-500], dry[-1000:-500])

    def test_mismatched_sample_rate_is_left_alone(self) -> None:
        sound = Sound(noise(0.1), 24_000)
        assert RoomReverbProcessor(SR, "small", 20.0).apply(sound) is sound


# ---

EFFECT_LEVEL = 0.05
PAUSE_S = 0.4
EFFECT_S = 0.2


def run_concat(
        phrases: list[Phrase], sounds: list[np.ndarray], room_reverb: RoomReverb, effect_level: float = EFFECT_LEVEL,
        room_reverb_stereo: RoomReverbStereo = RoomReverbStereo.MONO, init_stream: MagicMock | None = None,
        voice_pans: list[float] | None = None, stereo_required: bool = False,
        paths: list[str] | None = None, voice_reverbs: list[RoomReverb] | None = None,
) -> list[np.ndarray]:
    """
    Runs the concat flow with real pauses (0.4 s, or a constant 0.2 s "effect"
    for break effects); returns the arrays streamed to ffmpeg.
    """
    streamed: list[np.ndarray] = []
    effect = np.full(round(EFFECT_S * SR), effect_level, dtype=np.float32)

    def fake_append(sound_in, reason, reason_pauses, break_effect, pause_duration_override):
        if break_effect is not None:
            return Sound(np.concatenate([sound_in.data, effect]), SR)
        return Sound(np.concatenate([sound_in.data, np.zeros(round(PAUSE_S * SR), dtype=np.float32)]), SR)

    renders = iter(Sound(s, SR) for s in sounds)
    init_stream = init_stream or MagicMock()
    with patch.object(ConcatUtil, "init_ffmpeg_stream", init_stream), \
         patch.object(ConcatUtil, "close_ffmpeg_stream"), \
         patch.object(ConcatUtil, "add_audio_to_ffmpeg_stream",
                      side_effect=lambda process, data: streamed.append(data.copy())), \
         patch.object(ConcatUtil, "PSEUDO_SILENCE_COMPENSATION_ENABLED", False), \
         patch.object(SoundPipeline, "make_concat_rendered_sound_segment",
                      side_effect=lambda *args, **kwargs: next(renders)), \
         patch.object(SoundPipeline, "append_pause_or_section_effect", side_effect=fake_append):
        ConcatUtil.concatenate_sound_segments(
            dest_path="output.flac",
            phrases_and_paths=[
                (phrase, paths[i] if paths is not None else f"segment{i}.flac")
                for i, phrase in enumerate(phrases)
            ],
            use_break_sound_effect=True,
            high_shelf=HighShelfEq.DISABLED,
            reason_pauses=ReasonPauseTypes.NORMAL.value,
            print_progress=False,
            room_reverb=room_reverb,
            room_reverb_stereo=room_reverb_stereo,
            voice_pans=voice_pans,
            voice_reverbs=voice_reverbs,
            stereo_required=stereo_required,
        )
    return streamed


class TestConcatFlow:

    PHRASES = [Phrase("One.", Reason.SENTENCE), Phrase("Two.", Reason.SECTION_BREAK),
               Phrase("Three.", Reason.SENTENCE)]

    def test_durations_unchanged_and_tail_decays_into_pauses(self) -> None:
        sounds = [noise(0.6, i) for i in range(3)]
        streamed = run_concat(self.PHRASES, sounds, RoomReverb.MEDIUM_DRR21)
        assert [len(x) for x in streamed] == [round(0.6 * SR) + round(PAUSE_S * SR),
                                              round(0.6 * SR) + round(EFFECT_S * SR),
                                              round(0.6 * SR) + round(PAUSE_S * SR)]
        pause = streamed[0][len(sounds[0]):]
        assert energy(pause[:SR // 20]) > 0

    def test_break_effect_is_not_reverberated(self) -> None:
        # Changing the effect changes the output by exactly the effect: none of
        # it feeds the reverb
        sounds = [noise(0.6, i) for i in range(3)]
        a = run_concat(self.PHRASES, sounds, RoomReverb.SMALL_DRR17, effect_level=EFFECT_LEVEL)
        b = run_concat(self.PHRASES, sounds, RoomReverb.SMALL_DRR17, effect_level=0.0)
        diff = np.concatenate(a) - np.concatenate(b)
        effect_start = sum(len(x) for x in a[:1]) + len(sounds[1])
        np.testing.assert_allclose(diff[effect_start:effect_start + round(EFFECT_S * SR)], EFFECT_LEVEL, atol=1e-6)
        assert np.max(np.abs(diff[effect_start + round(EFFECT_S * SR):])) < 1e-6

    def test_off_adds_nothing(self) -> None:
        sounds = [noise(0.5, i) for i in range(2)]
        phrases = [Phrase(t, Reason.SENTENCE) for t in ("One.", "Two.")]
        streamed = run_concat(phrases, sounds, RoomReverb.OFF)
        for sound, out in zip(sounds, streamed):
            assert np.array_equal(out[:len(sound)], sound)
            assert energy(out[len(sound):]) == 0.0


class TestConcatPerVoiceRooms:
    """Each segment's narration feeds its own voice's room; all rooms keep decaying."""

    PHRASES = [Phrase("One.", Reason.SENTENCE), Phrase("Two.", Reason.SENTENCE)]

    @staticmethod
    def segment(sound: np.ndarray) -> np.ndarray:
        return np.concatenate([sound, np.zeros(round(PAUSE_S * SR), dtype=np.float32)])

    def test_output_is_dry_plus_each_room_fed_only_its_voices_narration(self) -> None:
        sounds = [noise(0.5, i) for i in range(2)]
        streamed = run_concat(self.PHRASES, sounds, RoomReverb.OFF,
                              voice_reverbs=[RoomReverb.SMALL_DRR22, RoomReverb.MEDIUM_DRR21])
        # Reverb is linear: build the expected mix from two independent rooms
        room_a = RoomReverbProcessor(SR, "small", 22.0)
        room_b = RoomReverbProcessor(SR, "medium", 21.0)
        seg0, seg1 = (Sound(self.segment(x), SR) for x in sounds)
        expected0 = seg0.data + room_a.process_wet(seg0, send_end=len(sounds[0]))
        expected1 = (seg1.data
                     + room_a.process_wet(seg1, send_end=0, is_last=True)
                     + room_b.process_wet(seg1, send_end=len(sounds[1]), is_last=True))
        np.testing.assert_allclose(streamed[0], expected0, atol=1e-6)
        np.testing.assert_allclose(streamed[1], expected1, atol=1e-6)

    def test_previous_voices_tail_continues_under_a_dry_voice(self) -> None:
        sounds = [noise(0.5, i) for i in range(2)]
        streamed = run_concat(self.PHRASES, sounds, RoomReverb.OFF,
                              voice_reverbs=[RoomReverb.MEDIUM_DRR21, RoomReverb.OFF])
        tail_under_voice_two = streamed[1][:SR // 20] - sounds[1][:SR // 20]
        assert energy(tail_under_voice_two) > 0

    def test_dry_voices_narration_feeds_no_room(self) -> None:
        # Changing the dry voice's audio changes the output by exactly that audio
        first = noise(0.5, 0)
        outs = [run_concat(self.PHRASES, [first, noise(0.5, seed)], RoomReverb.OFF,
                           voice_reverbs=[RoomReverb.MEDIUM_DRR21, RoomReverb.OFF])
                for seed in (1, 2)]
        wet = [out[1] - self.segment(noise(0.5, seed)) for out, seed in zip(outs, (1, 2))]
        np.testing.assert_allclose(wet[0], wet[1], atol=1e-6)
        # ...and before any voice with reverb speaks, nothing is added at all
        streamed = run_concat(self.PHRASES, [first, noise(0.5, 1)], RoomReverb.OFF,
                              voice_reverbs=[RoomReverb.OFF, RoomReverb.MEDIUM_DRR21])
        np.testing.assert_array_equal(streamed[0], self.segment(first))

    def test_voices_with_the_same_preset_share_one_room(self) -> None:
        sounds = [noise(0.5, i) for i in range(2)]
        shared = run_concat(self.PHRASES, sounds, RoomReverb.OFF,
                            voice_reverbs=[RoomReverb.SMALL_DRR17, RoomReverb.SMALL_DRR17])
        single = run_concat(self.PHRASES, sounds, RoomReverb.SMALL_DRR17)
        for a, b in zip(shared, single):
            np.testing.assert_array_equal(a, b)

    def test_any_voice_with_reverb_makes_the_output_stereo(self) -> None:
        sounds = [noise(0.5, i) for i in range(2)]
        init = MagicMock()
        streamed = run_concat(self.PHRASES, sounds, RoomReverb.OFF, init_stream=init,
                              room_reverb_stereo=RoomReverbStereo.LATE_W100,
                              voice_reverbs=[RoomReverb.OFF, RoomReverb.SMALL_DRR22])
        assert init.call_args.kwargs["channels"] == 2
        # The dry first voice is centered, the same in both channels
        np.testing.assert_array_equal(streamed[0][:, 0], streamed[0][:, 1])
        assert not np.allclose(streamed[1][len(sounds[1]):, 0], streamed[1][len(sounds[1]):, 1])

    def test_misaligned_voice_reverbs_rejected(self) -> None:
        with pytest.raises(ValueError, match="voice_reverbs"):
            run_concat(self.PHRASES, [noise(0.5)], RoomReverb.OFF, voice_reverbs=[RoomReverb.OFF])


class TestConcatVoicePositioning:

    def test_pans_without_reverb_and_keeps_effect_centered(self) -> None:
        from tts_audiobook_tool.sound.voice_spatial import center_unity_pan_gains

        phrases = [Phrase("One.", Reason.SECTION_BREAK), Phrase("Two.", Reason.SENTENCE)]
        sounds = [noise(0.3, i) for i in range(2)]
        init = MagicMock()
        streamed = run_concat(phrases, sounds, RoomReverb.OFF,
                              voice_pans=[0.25, 0.0], init_stream=init)
        assert init.call_args.kwargs["channels"] == 2
        gains = center_unity_pan_gains(0.25)
        np.testing.assert_allclose(streamed[0][:len(sounds[0])],
                                   sounds[0][:, None] * gains, atol=1e-7)
        np.testing.assert_allclose(streamed[0][len(sounds[0]):], EFFECT_LEVEL)
        np.testing.assert_array_equal(streamed[1][:len(sounds[1]), 0], sounds[1])
        np.testing.assert_array_equal(streamed[1][:len(sounds[1]), 1], sounds[1])

    def test_project_requirement_keeps_centered_range_stereo(self) -> None:
        init = MagicMock()
        dry = noise(0.3)
        streamed = run_concat([Phrase("One.", Reason.SENTENCE)], [dry], RoomReverb.OFF,
                              voice_pans=[0.0], stereo_required=True, init_stream=init)
        assert init.call_args.kwargs["channels"] == 2
        np.testing.assert_array_equal(streamed[0][:len(dry)], np.stack([dry, dry], axis=1))

    def test_voice_pan_stays_aligned_across_missing_files_and_lookahead(self) -> None:
        from tts_audiobook_tool.sound.voice_spatial import center_unity_pan_gains

        phrases = [Phrase(str(i), Reason.SENTENCE) for i in range(4)]
        sounds = [noise(0.3, i) for i in range(2)]
        streamed = run_concat(phrases, sounds, RoomReverb.OFF,
                              paths=["", "one.flac", "", "two.flac"],
                              voice_pans=[0.25, 0.0, 0.0, 0.25])
        np.testing.assert_array_equal(streamed[0][:len(sounds[0]), 0], sounds[0])
        np.testing.assert_allclose(streamed[1][:len(sounds[1])],
                                   sounds[1][:, None] * center_unity_pan_gains(0.25), atol=1e-7)

    def test_pan_changes_only_direct_speech_not_shared_room(self) -> None:
        phrases = [Phrase("One.", Reason.SENTENCE), Phrase("Two.", Reason.SENTENCE)]
        sounds = [noise(0.3, i) for i in range(2)]
        common = dict(room_reverb_stereo=RoomReverbStereo.LATE_W100, stereo_required=True)
        centered = run_concat(phrases, sounds, RoomReverb.SMALL_DRR22,
                              voice_pans=[0.0, 0.0], **common)
        panned = run_concat(phrases, sounds, RoomReverb.SMALL_DRR22,
                           voice_pans=[0.25, 0.0], **common)
        np.testing.assert_array_equal(centered[0][len(sounds[0]):], panned[0][len(sounds[0]):])
        np.testing.assert_array_equal(centered[1], panned[1])

    def test_stereo_for_pan_does_not_enable_room_spread(self) -> None:
        dry = noise(0.3)
        out = run_concat([Phrase("One.", Reason.SENTENCE)], [dry], RoomReverb.SMALL_DRR22,
                         voice_pans=[0.25])[0]
        np.testing.assert_array_equal(out[len(dry):, 0], out[len(dry):, 1])
        assert np.any(out[len(dry):, 0] != 0)


class TestStereo:

    def test_split_parts_sum_to_the_ir(self) -> None:
        p = ROOM_DESIGNS["small"]
        ir = make_room_ir(p, SR)
        early, late = split_ir(ir, p, SR)
        np.testing.assert_allclose(early + late, ir, atol=1e-15)

    @pytest.mark.parametrize("spread_early", [False, True])
    def test_mono_fold_down_is_the_mono_reverb_never_wetter(self, spread_early: bool) -> None:
        # (L + R) / 2 cancels the decorrelated part: what's left is the mono
        # IR with only the spread part(s) lowered, by 3 dB at full width
        p = ROOM_DESIGNS["medium"]
        mono = make_room_ir(p, SR)
        ir_l, ir_r = make_stereo_irs(p, SR, 1.0, spread_early)
        early, late = split_ir(mono, p, SR)
        c = np.cos(np.pi / 4)
        expected = (c * early if spread_early else early) + c * late
        np.testing.assert_allclose((ir_l + ir_r) / 2, expected, atol=1e-12)

    def test_full_width_decorrelates_the_spread_parts(self) -> None:
        p = ROOM_DESIGNS["medium"]
        ir_l, ir_r = make_stereo_irs(p, SR, 1.0, spread_early=True)
        corr = np.sum(ir_l * ir_r) / np.sqrt(energy(ir_l) * energy(ir_r))
        # Two random renderings of the room: uncorrelated only statistically
        assert abs(corr) < 0.05
        # Each ear keeps about the mono reverb's energy (unit), so the
        # stereo version isn't simply wetter
        for ir in (ir_l, ir_r):
            assert 10 * np.log10(energy(ir)) == pytest.approx(0.0, abs=0.5)

    def test_wide_keeps_early_reflections_identical(self) -> None:
        p = ROOM_DESIGNS["small"]
        ir_l, ir_r = make_stereo_irs(p, SR, 1.0, spread_early=False)
        early_end = round((p.predelay_ms + p.er_window_ms - 5) / 1000 * SR)
        np.testing.assert_allclose(ir_l[:early_end], ir_r[:early_end], atol=1e-12)
        assert not np.allclose(ir_l[early_end + SR // 20:], ir_r[early_end + SR // 20:])

    def test_processor_outputs_stereo_with_a_centered_voice(self) -> None:
        reverb = RoomReverbProcessor(SR, "medium", 10.0, width=1.0, spread_early=True)
        dry = noise(0.5)
        out = reverb.apply(Sound(dry, SR), is_last=True).data
        assert out.shape == (len(dry), 2) and reverb.channels == 2
        # Dry voice in both channels; the reverb differs between them
        assert not np.allclose(out[:, 0], out[:, 1])
        np.testing.assert_allclose(out[-1], [dry[-1], dry[-1]], atol=1e-6)

    def test_noise_bed_adds_the_same_bed_to_both_channels(self) -> None:
        stereo = np.zeros((SR // 2, 2), dtype=np.float32)
        out = NoiseBed(SR, -40.0).apply(Sound(stereo, SR)).data
        assert out.shape == stereo.shape
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        assert energy(out[:, 0]) > 0

    def test_concat_streams_stereo_only_with_reverb_on(self) -> None:
        sounds = [noise(0.5, i) for i in range(2)]
        phrases = [Phrase(t, Reason.SENTENCE) for t in ("One.", "Two.")]

        init = MagicMock()
        streamed = run_concat(phrases, sounds, RoomReverb.SMALL_DRR22,
                              room_reverb_stereo=RoomReverbStereo.LATE_W100, init_stream=init)
        assert init.call_args.kwargs["channels"] == 2
        assert all(x.ndim == 2 and x.shape[1] == 2 for x in streamed)

        # The stereo setting does nothing while reverb is off
        init = MagicMock()
        streamed = run_concat(phrases, sounds, RoomReverb.OFF,
                              room_reverb_stereo=RoomReverbStereo.ALL_W100, init_stream=init)
        assert init.call_args.kwargs["channels"] == 1
        assert all(x.ndim == 1 for x in streamed)


# ---

def normalized(d: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    ProjectSerializationUtil.normalize_loaded_project_dict(d, warnings=warnings if warnings is not None else [])
    return d


class TestSetting:
    """`room_reverb` is per voice (project spec v5); `room_reverb_stereo` is project-wide."""

    def test_new_voice_defaults_to_off(self) -> None:
        assert RoomReverb.get_default() is RoomReverb.OFF
        assert Voice("a.flac").room_reverb is RoomReverb.OFF
        assert Project().get_voice_chat_room_reverb() is RoomReverb.OFF

    def test_missing_voice_key_loads_as_off(self) -> None:
        d = normalized({"version": 5, "voice_references": [{"file_name": "a.flac"}]})
        assert d["voice_references"][0].room_reverb is RoomReverb.OFF
        assert "room_reverb" not in d

    def test_unknown_voice_id_falls_back_to_off(self) -> None:
        d = normalized({"voice_references": [{"file_name": "a.flac", "room_reverb": "cathedral"}]})
        assert d["voice_references"][0].room_reverb is RoomReverb.OFF

    def test_ids_are_unique(self) -> None:
        ids = [item.id for item in RoomReverb]
        assert len(ids) == len(set(ids))

    def test_round_trips_through_json_dict(self) -> None:
        project = Project()
        project.voice_references = [Voice("a.flac", room_reverb=RoomReverb.MEDIUM_DRR21), Voice("b.flac")]
        project.room_reverb_stereo = RoomReverbStereo.ALL_W100
        saved = ProjectSerializationUtil.to_project_json_dict(project)
        assert "room_reverb" not in saved
        # Always written, including "off"
        assert [ref["room_reverb"] for ref in saved["voice_references"]] == ["medium_drr21", "off"]
        assert saved["room_reverb_stereo"] == "all_w100"
        loaded = Project.model_validate(saved)
        assert [voice.room_reverb for voice in loaded.voice_references] == [RoomReverb.MEDIUM_DRR21, RoomReverb.OFF]
        assert loaded.room_reverb_stereo is RoomReverbStereo.ALL_W100

    def test_v4_project_setting_moves_onto_every_voice(self) -> None:
        d = normalized({
            "version": 4, "room_reverb": "small_drr17", "room_reverb_stereo": "late_w100",
            "voice_references": [{"file_name": "a.flac"}, {"file_name": "b.flac"}],
        })
        assert [voice.room_reverb for voice in d["voice_references"]] == [RoomReverb.SMALL_DRR17] * 2
        assert "room_reverb" not in d
        assert d["room_reverb_stereo"] is RoomReverbStereo.LATE_W100

    def test_effective_reverb_is_voice_zeros(self) -> None:
        project = Project(tts_model_type="chatterbox_local")
        project.voice_references = [Voice("a.flac", room_reverb=RoomReverb.SMALL_DRR22),
                                    Voice("b.flac", room_reverb=RoomReverb.MEDIUM_DRR21)]
        assert project.get_voice_chat_room_reverb() is RoomReverb.SMALL_DRR22
        project.voice_references = []
        assert project.get_voice_chat_room_reverb() is RoomReverb.OFF

    def test_no_reverb_while_pocket_preset_is_used(self) -> None:
        project = Project(tts_model_type="pocket_local")
        project.voice_references = [Voice("a.flac", room_reverb=RoomReverb.SMALL_DRR22)]
        assert project.get_voice_chat_room_reverb() is RoomReverb.SMALL_DRR22
        project.set_model_setting("pocket_local", "predefined_voice", "alba")
        assert project.get_voice_chat_room_reverb() is RoomReverb.OFF

    def test_stereo_defaults_to_mono_silently(self) -> None:
        warnings: list[str] = []
        assert Project().room_reverb_stereo is RoomReverbStereo.MONO
        assert normalized({}, warnings)["room_reverb_stereo"] is RoomReverbStereo.MONO
        assert normalized({"room_reverb_stereo": "bogus"})["room_reverb_stereo"] is RoomReverbStereo.MONO
        assert not [w for w in warnings if "room_reverb_stereo" in w]
