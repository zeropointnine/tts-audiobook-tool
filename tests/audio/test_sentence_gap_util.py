from dataclasses import replace

import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from sentence_gap_test_support import SR, Utterance, make_floor, make_speech
from tts_audiobook_tool.sound.sentence_gap_util import (
    BoundaryGapResult,
    GapStatus,
    LoudnessEnvelope,
    SentenceBoundary,
    SentenceGapInsertion,
    SentenceGapParams,
    SentenceGapUtil,
)

PARAMS = SentenceGapParams(min_gap_s=0.4)


def run(u: Utterance, params: SentenceGapParams = PARAMS):
    return SentenceGapUtil.extend_sentence_gaps(u.sound, u.timed_phrases, u.words, params)


class TestExtend:

    def test_short_gap_is_extended_inside_the_true_gap(self) -> None:
        # A 150 ms gap with accurate timings is lengthened to the 400 ms target,
        # and the silence is inserted within the original gap, not in a word.
        u = Utterance(gap_s=0.15)
        report = run(u)

        assert len(report.results) == 1
        r = report.results[0]
        assert r.status is GapStatus.EXTEND, r.summary()
        assert r.insert_at is not None
        assert u.true_gap_start < r.insert_at < u.true_gap_end
        assert report.sound.duration == pytest.approx(u.sound.duration + r.insert_seconds, abs=1 / SR)
        assert report.sound.duration > u.sound.duration + 0.2

    def test_extended_audio_measures_as_long_enough_when_reanalyzed(self) -> None:
        # Idempotence: analyzing the output (with the boundary times mapped through
        # the insertion) sees a gap at the target length and asks for nothing more.
        u = Utterance(gap_s=0.15)
        report = run(u)
        boundary = report.results[0].boundary
        mapped = replace(
            boundary,
            lo=SentenceGapUtil.map_time(report.insertions, boundary.lo),
            hi=SentenceGapUtil.map_time(report.insertions, boundary.hi),
        )
        again = SentenceGapUtil.analyze(report.sound, [mapped], PARAMS)[0]

        assert again.status is GapStatus.LONG_ENOUGH, again.summary()
        assert again.gap_width is not None
        assert again.gap_width >= PARAMS.min_gap_s - 0.02

    def test_gap_with_noise_floor_is_extended_too(self) -> None:
        # Real TTS gaps are rarely digital zero; a -70 dB hiss floor still works.
        u = Utterance(gap_s=0.15, floor_level=0.0003)
        r = run(u).results[0]

        assert r.status is GapStatus.EXTEND, r.summary()
        assert r.insert_at is not None
        assert u.true_gap_start < r.insert_at < u.true_gap_end

    def test_gap_already_long_enough_is_left_alone(self) -> None:
        # No insertion, and the very same Sound object comes back.
        u = Utterance(gap_s=0.6)
        report = run(u)

        assert report.results[0].status is GapStatus.LONG_ENOUGH
        assert report.insertions == []
        assert report.sound is u.sound

    @pytest.mark.parametrize("end_shift,start_shift", [
        (0.0, 0.0),
        (+0.12, -0.12),   # Whisper smears the words into the gap
        (-0.10, +0.10),   # Whisper pulls the words back from the gap
        (+0.15, +0.15),   # both timings late
        (-0.15, -0.15),   # both timings early
    ])
    def test_imprecise_word_timings_still_find_the_gap(self, end_shift: float, start_shift: float) -> None:
        # Word timings only choose the search window; offsets of the size Whisper
        # produces must not move the insertion out of the real gap.
        u = Utterance(gap_s=0.15, whisper_end_shift=end_shift, whisper_start_shift=start_shift)
        r = run(u).results[0]

        assert r.status is GapStatus.EXTEND, r.summary()
        assert r.insert_at is not None
        assert u.true_gap_start < r.insert_at < u.true_gap_end

    @pytest.mark.parametrize("neighbor", ["before", "after"])
    @pytest.mark.parametrize("word_gap_s", [0.12, 0.5])
    @pytest.mark.parametrize("sentence_gap_s", [0.04, 0.02, 0.0])
    def test_unrelated_word_pause_is_not_claimed(
            self, neighbor: str, word_gap_s: float, sentence_gap_s: float,
    ) -> None:
        # A wider word pause on either side of a short sentence boundary must
        # neither win the width ranking nor substitute for a missing sentence gap.
        parts: list[np.ndarray] = []
        words: list[dict] = []
        t = 0.0
        word_gap = (0.0, 0.0)
        sentence_gap = (0.0, 0.0)
        for k, (word, duration, pause) in enumerate([
            ("That", 0.3, 0.0),
            ("is", 0.2, word_gap_s if neighbor == "before" else 0.0),
            ("it.", 0.06, sentence_gap_s),
            ("Go", 0.06, word_gap_s if neighbor == "after" else 0.0),
            ("on.", 0.3, 0.0),
        ]):
            parts.append(make_speech(duration, seed=k))
            words.append({"word": word, "start": t, "end": t + duration})
            t += duration
            if k == 2:
                sentence_gap = (t, t + pause)
            elif (k == 1 and neighbor == "before") or (k == 3 and neighbor == "after"):
                word_gap = (t, t + pause)
            parts.append(make_floor(pause))
            t += pause
        sound = Sound(np.concatenate(parts), SR)
        boundary_time = words[3]["start"]
        phrases = [
            TimedPhrase("That is it. ", 0.0, boundary_time, Reason.SENTENCE),
            TimedPhrase("Go on.", boundary_time, sound.duration, Reason.SENTENCE),
        ]

        report = SentenceGapUtil.extend_sentence_gaps(sound, phrases, words, PARAMS)
        result = report.results[0]

        if sentence_gap_s >= PARAMS.min_run_s:
            assert result.status is GapStatus.EXTEND, result.summary()
            assert len(report.insertions) == 1
            insertion = report.insertions[0]
            assert sentence_gap[0] < insertion.position < sentence_gap[1]
            assert sentence_gap_s + insertion.duration == pytest.approx(PARAMS.min_gap_s, abs=0.01)
        else:
            assert result.status is GapStatus.SKIP, result.summary()
            assert report.insertions == []
            assert report.sound is sound
        start, end = word_gap
        mapped_start = SentenceGapUtil.map_time(report.insertions, start)
        mapped_end = SentenceGapUtil.map_time(report.insertions, end)
        assert mapped_end - mapped_start == pytest.approx(word_gap_s)
        assert np.array_equal(
            report.sound.data[round(mapped_start * SR):round(mapped_end * SR)],
            sound.data[round(start * SR):round(end * SR)],
        )

    def test_closure_inside_previous_word_does_not_beat_a_brief_real_pause(self) -> None:
        # A stop consonant's closure inside the word before the boundary, ~0.15 s
        # before Whisper's reported word end, is wider than the brief real pause
        # just after that end. The narrow early search margin must keep it out
        # of the window, so the real pause is extended rather than the word split.
        closure = (0.55, 0.605)
        pause = (0.805, 0.84)
        data = np.concatenate([
            make_speech(closure[0], seed=1), make_floor(closure[1] - closure[0]),
            make_speech(pause[0] - closure[1], seed=2), make_floor(pause[1] - pause[0]),
            make_speech(0.5, seed=3),
        ])
        boundary = SentenceBoundary(phrase_index=0, lo=0.76, hi=0.9, prev_word="a", next_word="b")

        r = SentenceGapUtil.analyze(Sound(data, SR), [boundary], PARAMS)[0]

        assert r.status is GapStatus.EXTEND, r.summary()
        assert r.insert_at is not None
        assert pause[0] < r.insert_at < pause[1]

    def test_timings_off_by_more_than_the_window_are_skipped_not_misplaced(self) -> None:
        # When Whisper is badly wrong the search window misses the gap entirely.
        # Whatever happens, silence must never be inserted outside the true gap.
        u = Utterance(gap_s=0.15, whisper_end_shift=-0.9, whisper_start_shift=-0.9)
        report = run(u)
        r = report.results[0]

        if r.status is GapStatus.EXTEND:
            assert r.insert_at is not None
            assert u.true_gap_start < r.insert_at < u.true_gap_end
        else:
            assert report.sound is u.sound


class TestSkips:

    def test_no_gap_at_all_is_skipped(self) -> None:
        # Sentence boundary with no pause in the audio: nothing to extend, and
        # no silence may be fabricated mid-speech.
        u = Utterance(gap_s=0.0)
        report = run(u)

        assert report.results[0].status is GapStatus.SKIP
        assert report.sound is u.sound

    def test_only_word_dips_in_window_is_skipped(self) -> None:
        # A 25 ms soft dip is what a word boundary looks like; it's too short to
        # be taken for the sentence gap.
        u = Utterance(gap_s=0.025)
        r = run(u).results[0]

        assert r.status is GapStatus.SKIP
        assert r.reason in ("gap_too_short", "no_valley")

    def test_two_equal_gaps_in_window_is_ambiguous(self) -> None:
        # If two comparable pauses sit in the search window, we can't know which
        # is the sentence boundary, so we leave the audio alone.
        speech_a = make_speech(0.4, seed=1)
        speech_b = make_speech(0.15, seed=2)
        gap = make_floor(0.12)
        data = np.concatenate([make_speech(0.5), gap, speech_b, gap, speech_a])
        sound = Sound(data, SR)
        boundary = SentenceBoundary(phrase_index=0, lo=0.55, hi=0.7, prev_word="a", next_word="b")

        r = SentenceGapUtil.analyze(sound, [boundary], PARAMS)[0]

        assert r.status is GapStatus.SKIP
        assert r.reason == "ambiguous"

    def test_gap_touching_clip_edge_is_skipped(self) -> None:
        # A quiet run that runs off the end of the clip is trailing silence, not a
        # sentence gap with speech on both sides.
        data = np.concatenate([make_speech(0.5), make_floor(0.3)])
        boundary = SentenceBoundary(phrase_index=0, lo=0.45, hi=0.55)

        r = SentenceGapUtil.analyze(Sound(data, SR), [boundary], PARAMS)[0]

        assert r.status is GapStatus.SKIP

    def test_extension_is_confined_to_the_search_window(self) -> None:
        # The gap is well away from where the words claim the boundary is.
        data = np.concatenate([make_speech(1.0), make_floor(0.15), make_speech(1.0)])
        boundary = SentenceBoundary(phrase_index=0, lo=0.2, hi=0.3)

        r = SentenceGapUtil.analyze(Sound(data, SR), [boundary], PARAMS)[0]

        assert r.status is GapStatus.SKIP


class TestFindBoundaries:

    def make_inputs(self):
        words = [
            {"word": " Hello", "start": 0.0, "end": 0.3},
            {"word": " there,", "start": 0.3, "end": 0.6},
            {"word": " friend.", "start": 0.6, "end": 1.0},
            {"word": " How", "start": 1.3, "end": 1.5},
            {"word": " are", "start": 1.5, "end": 1.7},
            {"word": " you?", "start": 1.7, "end": 2.0},
            {"word": " Fine.", "start": 2.4, "end": 2.8},
        ]
        timed = [
            TimedPhrase("Hello there, ", 0.0, 0.6, reason=Reason.PHRASE),
            TimedPhrase("friend. ", 0.6, 1.3, reason=Reason.SENTENCE),
            TimedPhrase("How are you? ", 1.3, 2.4, reason=Reason.PARAGRAPH),
            TimedPhrase("Fine.", 2.4, 2.8, reason=Reason.SENTENCE),
        ]
        return timed, words

    def test_only_boundaries_whose_reason_is_a_sentence_or_stronger(self) -> None:
        # The comma boundary (Reason.PHRASE) is not a sentence gap; the SENTENCE and
        # PARAGRAPH boundaries are. The reasons are the ones recorded in the sidecar,
        # so no text-based guessing is involved.
        timed, words = self.make_inputs()
        boundaries = SentenceGapUtil.find_boundaries(timed, words)

        assert [b.phrase_index for b in boundaries] == [1, 2]
        assert [b.reason for b in boundaries] == [Reason.SENTENCE, Reason.PARAGRAPH]
        assert not any(b.skip_reason for b in boundaries)

    def test_reason_wins_over_how_the_text_looks(self) -> None:
        # A phrase ending in a period whose Reason is PHRASE (an ellipsis-style
        # fragment, say) is not a sentence boundary.
        timed, words = self.make_inputs()
        timed[1].reason = Reason.PHRASE

        boundaries = SentenceGapUtil.find_boundaries(timed, words)

        assert [b.phrase_index for b in boundaries] == [2]

    def test_min_reason_param_widens_or_narrows_the_boundaries_considered(self) -> None:
        timed, words = self.make_inputs()

        every = SentenceGapUtil.find_boundaries(timed, words, SentenceGapParams(min_reason=Reason.PHRASE))
        paragraphs = SentenceGapUtil.find_boundaries(timed, words, SentenceGapParams(min_reason=Reason.PARAGRAPH))

        assert [b.phrase_index for b in every] == [0, 1, 2]
        assert [b.phrase_index for b in paragraphs] == [2]

    def test_phrase_without_a_recorded_reason_is_skipped(self) -> None:
        # A sidecar from before reasons were saved: we can't tell what kind of
        # boundary this is, so it is reported as skipped, never acted on.
        timed, words = self.make_inputs()
        for t in timed:
            t.reason = None

        boundaries = SentenceGapUtil.find_boundaries(timed, words)

        assert [b.phrase_index for b in boundaries] == [0, 1, 2]
        assert all(b.skip_reason == "no_reason" for b in boundaries)

    def test_bracket_is_previous_word_end_to_next_word_start(self) -> None:
        timed, words = self.make_inputs()
        b = SentenceGapUtil.find_boundaries(timed, words)[0]

        assert (b.lo, b.hi) == (1.0, 1.3)
        assert b.prev_word.strip() == "friend."
        assert b.next_word.strip() == "How"

    def test_transcript_words_that_dont_match_the_source_are_flagged(self) -> None:
        # A misrecognized word beside the boundary means the force-alignment may be
        # a word off, so that boundary must not be acted on.
        timed, words = self.make_inputs()
        words[2] = {**words[2], "word": " ferrets."}

        b = SentenceGapUtil.find_boundaries(timed, words)[0]

        assert b.skip_reason == "word_mismatch"

    def test_boundary_with_no_matching_transcript_word_is_flagged(self) -> None:
        timed, words = self.make_inputs()
        timed[1].time_end = 1.1  # no word starts there

        b = SentenceGapUtil.find_boundaries(timed, words)[0]

        assert b.skip_reason == "no_word_at_boundary"

    def test_flagged_boundary_is_reported_as_skip(self) -> None:
        sound = Sound(make_speech(3.0), SR)
        boundary = SentenceBoundary(phrase_index=0, lo=1.0, hi=1.3, skip_reason="word_mismatch")

        r = SentenceGapUtil.analyze(sound, [boundary], PARAMS)[0]

        assert (r.status, r.reason) == (GapStatus.SKIP, "word_mismatch")


class TestApplyAndMapping:

    def test_map_time_shifts_only_times_after_the_insertion(self) -> None:
        insertions = [SentenceGapInsertion(1.0, 0.2), SentenceGapInsertion(3.0, 0.1)]

        assert SentenceGapUtil.map_time(insertions, 0.5) == pytest.approx(0.5)
        assert SentenceGapUtil.map_time(insertions, 1.0) == pytest.approx(1.0)
        assert SentenceGapUtil.map_time(insertions, 2.0) == pytest.approx(2.2)
        assert SentenceGapUtil.map_time(insertions, 4.0) == pytest.approx(4.3)

    def test_multiple_insertions_in_one_sound(self) -> None:
        # Two sentence boundaries in one clip are both extended; total length
        # grows by the sum, and the mapping accounts for both.
        data = np.concatenate([
            make_speech(0.5, 1), make_floor(0.15),
            make_speech(0.5, 2), make_floor(0.2),
            make_speech(0.5, 3),
        ])
        sound = Sound(data, SR)
        boundaries = [
            SentenceBoundary(0, lo=0.45, hi=0.7),
            SentenceBoundary(1, lo=1.1, hi=1.4),
        ]
        results = SentenceGapUtil.analyze(sound, boundaries, PARAMS)
        new_sound, insertions = SentenceGapUtil.apply(sound, results, PARAMS)

        assert [r.status for r in results] == [GapStatus.EXTEND, GapStatus.EXTEND]
        assert len(insertions) == 2
        assert new_sound.duration == pytest.approx(sound.duration + sum(i.duration for i in insertions), abs=2 / SR)
        assert SentenceGapUtil.map_time(insertions, sound.duration) == pytest.approx(new_sound.duration, abs=2 / SR)

    @pytest.mark.parametrize("first_gap,second_gap", [(0.15, 0.05), (0.05, 0.15), (0.1, 0.1)])
    def test_short_consecutive_sentences_extend_their_own_gaps(
            self, first_gap: float, second_gap: float,
    ) -> None:
        # Overlapping search windows must not make both boundaries extend the
        # wider pause, or make equally wide but separately owned pauses ambiguous.
        second_start = 0.5 + first_gap
        second_end = second_start + 0.1
        third_start = second_end + second_gap
        sound = Sound(np.concatenate([
            make_speech(0.5), make_floor(first_gap),
            make_speech(0.1), make_floor(second_gap), make_speech(0.5),
        ]), SR)
        words = [
            {"word": "Hello.", "start": 0.0, "end": 0.5},
            {"word": "Yes.", "start": second_start, "end": second_end},
            {"word": "Bye.", "start": third_start, "end": sound.duration},
        ]
        phrases = [
            TimedPhrase("Hello.", 0.0, second_start, Reason.SENTENCE),
            TimedPhrase("Yes.", second_start, third_start, Reason.SENTENCE),
            TimedPhrase("Bye.", third_start, sound.duration, Reason.SENTENCE),
        ]
        params = replace(PARAMS, min_gap_s=0.5)

        report = SentenceGapUtil.extend_sentence_gaps(sound, phrases, words, params)

        assert [r.status for r in report.results] == [GapStatus.EXTEND, GapStatus.EXTEND]
        assert len(report.insertions) == 2
        first, second = report.insertions
        assert 0.5 < first.position < second_start
        assert second_end < second.position < third_start
        assert first_gap + first.duration == pytest.approx(0.5, abs=0.01)
        assert second_gap + second.duration == pytest.approx(0.5, abs=0.01)

    def test_boundary_without_a_gap_does_not_reuse_its_neighbors_gap(self) -> None:
        # A short next sentence has no pause of its own; its widened window can
        # see the previous pause, but must not claim and extend it a second time.
        sound = Sound(np.concatenate([
            make_speech(0.5), make_floor(0.15), make_speech(0.6),
        ]), SR)
        boundaries = [SentenceBoundary(0, 0.5, 0.65), SentenceBoundary(1, 0.75, 0.8)]

        results = SentenceGapUtil.analyze(sound, boundaries, PARAMS)
        _, insertions = SentenceGapUtil.apply(sound, results, PARAMS)

        assert [r.status for r in results] == [GapStatus.EXTEND, GapStatus.SKIP]
        assert len(insertions) == 1
        assert 0.5 < insertions[0].position < 0.65

    def test_equally_close_boundaries_cannot_both_claim_a_gap(self) -> None:
        # Collapsed or overlapping STT brackets can give equally plausible
        # ownership. Leave the audio alone rather than picking an arbitrary owner.
        sound = Sound(np.concatenate([
            make_speech(0.5), make_floor(0.15), make_speech(0.5),
        ]), SR)
        boundaries = [SentenceBoundary(0, 0.5, 0.65), SentenceBoundary(1, 0.5, 0.65)]

        results = SentenceGapUtil.analyze(sound, boundaries, PARAMS)
        out, insertions = SentenceGapUtil.apply(sound, results, PARAMS)

        assert all(r.status is GapStatus.SKIP for r in results)
        assert out is sound
        assert insertions == []

    def test_overlapping_claims_from_different_thresholds_are_skipped(self, monkeypatch) -> None:
        # Per-window thresholds can measure different extents of one physical
        # pause. Even different insertion positions must not bypass the guard.
        sound = Sound(make_speech(1.5), SR)
        boundaries = [SentenceBoundary(0, 0.5, 0.6), SentenceBoundary(1, 0.7, 0.8)]
        claims = iter([
            BoundaryGapResult(boundaries[0], GapStatus.EXTEND,
                              gap_start=0.5, gap_end=0.7, insert_at=0.6, insert_seconds=0.2),
            BoundaryGapResult(boundaries[1], GapStatus.EXTEND,
                              gap_start=0.6, gap_end=0.8, insert_at=0.7, insert_seconds=0.2),
        ])
        monkeypatch.setattr(SentenceGapUtil, "_analyze_one", lambda *args: next(claims))

        results = SentenceGapUtil.analyze(sound, boundaries, PARAMS)
        out, insertions = SentenceGapUtil.apply(sound, results, PARAMS)

        assert all(r.status is GapStatus.SKIP and r.reason == "shared_gap" for r in results)
        assert all(r.insert_at is None and r.insert_seconds == 0.0 for r in results)
        assert out is sound
        assert insertions == []

    def test_untrusted_boundary_does_not_steal_a_gap(self) -> None:
        # A boundary already rejected by text alignment must not compete with a
        # trustworthy boundary for the same pause.
        sound = Sound(np.concatenate([
            make_speech(0.5), make_floor(0.15), make_speech(0.5),
        ]), SR)
        boundaries = [
            SentenceBoundary(0, 0.5, 0.65),
            SentenceBoundary(1, 0.5, 0.65, skip_reason="word_mismatch"),
        ]

        results = SentenceGapUtil.analyze(sound, boundaries, PARAMS)

        assert [r.status for r in results] == [GapStatus.EXTEND, GapStatus.SKIP]

    def test_audio_outside_the_gap_is_untouched(self) -> None:
        # Speech on either side of the insertion is bit-identical to the original
        # (the fade only touches samples that were already inside the quiet gap).
        u = Utterance(gap_s=0.15, floor_level=0.0003)
        report = run(u)
        ins = report.insertions[0]
        before = round(u.true_gap_start * SR)
        after_orig = round(u.true_gap_end * SR)
        after_new = after_orig + round(ins.duration * SR)

        assert np.array_equal(report.sound.data[:before], u.sound.data[:before])
        assert np.array_equal(report.sound.data[after_new:], u.sound.data[after_orig:])

    def test_inserted_region_is_digital_silence(self) -> None:
        u = Utterance(gap_s=0.15, floor_level=0.0003)
        report = run(u)
        ins = report.insertions[0]
        start = round(ins.position * SR)
        stop = start + round(ins.duration * SR)

        assert not np.any(report.sound.data[start:stop])

    def test_apply_does_nothing_without_extend_results(self) -> None:
        sound = Sound(make_speech(1.0), SR)

        out, insertions = SentenceGapUtil.apply(sound, [], PARAMS)

        assert out is sound and insertions == []

    def test_dtype_is_preserved(self) -> None:
        u = Utterance(gap_s=0.15)

        assert run(u).sound.data.dtype == u.sound.data.dtype


class TestLoudnessEnvelope:

    def test_digital_silence_is_far_below_speech(self) -> None:
        sound = Sound(np.concatenate([make_speech(0.5), make_floor(0.3), make_speech(0.5)]), SR)
        env = LoudnessEnvelope.compute(sound)

        gap_db = env.db[env.frame_at(0.65)]
        speech_db = env.db[env.frame_at(0.25)]

        assert speech_db - gap_db > 80
        assert env.duration == pytest.approx(sound.duration)

    def test_short_dip_is_resolved(self) -> None:
        # The envelope's 10 ms frames must be able to see a 40 ms gap; this is what
        # SilenceUtil.detect_silences' 30 ms frames cannot do.
        sound = Sound(np.concatenate([make_speech(0.5), make_floor(0.04), make_speech(0.5)]), SR)
        env = LoudnessEnvelope.compute(sound)

        assert env.db[env.frame_at(0.52)] < env.ref_db - 60
