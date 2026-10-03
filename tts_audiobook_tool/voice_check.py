"""
Voice similarity check for Qwen3-TTS voice clone generations.

Compares the speaker embedding of a generated segment with that of the
reference voice sample (the same sample the voice clone is made from).
A result that has passed validation but falls below the similarity threshold
is turned into a VoiceMismatchResult, which fails validation and causes the
line to be retried.

Retry schedule (attempt 1 is the first generation of a line):
- From attempt 2 on, a retry of a line whose previous attempt failed the voice
  check is generated with the Qwen3 temperature lowered by RETRY_TEMPERATURE_DELTA
  per attempt, down to the "min temperature" setting. The project's temperature is only
  changed in memory, for the duration of that generation.
- The similarity threshold depends on the duration of the generated segment
  (THRESHOLD_BY_DURATION; short segments have lower similarity values), plus
  the "threshold offset" setting.
- Once the minimum temperature has been reached, the similarity threshold is
  lowered by THRESHOLD_STEP for each further attempt.
- At most "max attempts" attempts are made while the voice check fails.

The settings (on/off, threshold offset, max attempts, min temperature) are stored in their own file in
the app user directory, so that they are available in both the main process
and the model worker process.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from typing import Iterator

import numpy as np

from tts_audiobook_tool.app_support import app_paths
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.validation_findings import ValidationFindings, ValidationInvalidReason
from tts_audiobook_tool.app_types.validation_result import SkippedResult, TranscriptResult, ValidationResult
from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM, COL_ERROR
from tts_audiobook_tool.util import make_error_string, printt


def _make_mismatch_message(
        similarity: float, threshold: float, kept_similarity: float | None, duration: float
) -> str:
    message = (
        f"{COL_ERROR}Voice mismatch {COL_DIM}(similarity={COL_DEFAULT}{similarity:.3f}"
        f"{COL_DIM}, threshold={COL_DEFAULT}{threshold:.3f}{COL_DIM} for {duration:.1f}s)"
    )
    if kept_similarity is not None:
        message += (
            f"{COL_DEFAULT}; earlier take is better {COL_DIM}(similarity={COL_DEFAULT}"
            f"{kept_similarity:.3f}{COL_DIM})"
        )
    return message


@dataclass
class VoiceMismatchResult(TranscriptResult):
    """
    The generated voice deviates from the reference voice sample
    (speaker embedding similarity below threshold).
    """

    similarity: float
    threshold: float
    # Set when an earlier take of the line had a higher similarity; this take
    # is then not saved (the earlier take's file is kept)
    kept_similarity: float | None = None

    def __post_init__(self):
        if self.findings.invalid_reason is None:
            self.findings.invalid_reason = ValidationInvalidReason.VOICE_MISMATCH

    @property
    def is_fail(self) -> bool:
        return self.findings.is_hard_invalid

    def get_ui_message(self) -> str:
        return _make_mismatch_message(self.similarity, self.threshold, self.kept_similarity, self.sound.duration)


@dataclass
class VoiceMismatchSkippedResult(ValidationResult):
    """
    Like VoiceMismatchResult, for a generation whose speech-to-text validation
    was skipped (eg, Whisper disabled), so that there is no transcript.
    """

    similarity: float
    threshold: float
    # Set when an earlier take of the line had a higher similarity; this take
    # is then not saved (the earlier take's file is kept)
    kept_similarity: float | None = None

    def __post_init__(self):
        if self.findings.invalid_reason is None:
            self.findings.invalid_reason = ValidationInvalidReason.VOICE_MISMATCH

    @property
    def is_fail(self) -> bool:
        return self.findings.is_hard_invalid

    def get_ui_message(self) -> str:
        return _make_mismatch_message(self.similarity, self.threshold, self.kept_similarity, self.sound.duration)


class VoiceCheck:

    # Cosine similarity threshold by duration of the generated segment:
    # (duration in seconds, exclusive upper bound; threshold), ascending.
    # Derived from listening tests on Qwen3-TTS-12Hz-1.7B-Base voice clone output.
    THRESHOLD_BY_DURATION = (
        (1.0, 0.947),
        (1.5, 0.960),
        (2.0, 0.965),
        (3.0, 0.969),
        (5.0, 0.973),
        (float("inf"), 0.978),
    )
    # Added to all values of THRESHOLD_BY_DURATION (setting)
    THRESHOLD_OFFSET_DEFAULT = 0.0
    THRESHOLD_OFFSET_MIN = -0.05
    THRESHOLD_OFFSET_MAX = 0.05
    # Once the minimum temperature has been reached, the threshold is lowered
    # by this amount for each further attempt
    THRESHOLD_STEP = 0.001
    # Max attempts (first generation included) while the voice check fails
    MAX_ATTEMPTS_DEFAULT = 10
    MAX_ATTEMPTS_MIN = 1
    MAX_ATTEMPTS_MAX = 30
    # From attempt 2 on, the temperature is lowered by this amount per attempt...
    RETRY_TEMPERATURE_DELTA = 0.1
    # ... down to this value (setting). Very low temperatures can make Qwen3
    # fail to end short prompts (runaway generation until GEN_TIMEOUT).
    RETRY_TEMPERATURE_MIN_DEFAULT = 0.3
    RETRY_TEMPERATURE_MIN_MIN = 0.1
    RETRY_TEMPERATURE_MIN_MAX = 1.0
    # Lower limit of the min temperature for short lines:
    # (number of letters/digits of the line, exclusive upper bound; min temperature), ascending
    RETRY_TEMPERATURE_MIN_BY_LENGTH = ((6, 0.5), (12, 0.4), (18, 0.3), (24, 0.2))

    # Required by Qwen3TTSForConditionalGeneration.extract_speaker_embedding()
    SPEAKER_SR = 24000

    # Generation length limit: seconds of audio = BASE + PER_LETTER * letters/digits
    # of the line (measured max in an audiobook: 0.30 s per letter for lines with
    # 24+ letters, 1.2 s for lines with less than 6 letters)
    MAX_AUDIO_SECONDS_BASE = 5.0
    MAX_AUDIO_SECONDS_PER_LETTER = 0.3
    # Qwen3-TTS-Tokenizer-12Hz: 12.5 codec frames per second
    CODEC_FRAMES_PER_SECOND = 12.5

    SETTINGS_FILE_NAME = "tts-audiobook-tool-voice-check.json"

    # Worker process state:
    # Indices of lines whose latest attempt failed the voice check
    _mismatch_indices: set[int] = set()
    # Threshold reduction per line index for the current batch (set by attempt_scope())
    _threshold_reductions: dict[int, float] = {}
    # Highest similarity of the saved voice mismatch take per line index,
    # while the line is being retried
    _best_mismatches: dict[int, float] = {}

    # --- Setting

    @staticmethod
    def get_settings_path() -> str:
        return os.path.join(app_paths.get_app_user_dir(), VoiceCheck.SETTINGS_FILE_NAME)

    @staticmethod
    def _load_settings() -> dict:
        try:
            with open(VoiceCheck.get_settings_path(), encoding="utf-8") as f:
                settings = json.load(f)
            return settings if isinstance(settings, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _save_setting(key: str, value) -> str:
        """ Returns error string or empty string on success """
        settings = VoiceCheck._load_settings()
        settings[key] = value
        try:
            with open(VoiceCheck.get_settings_path(), "w", encoding="utf-8") as f:
                json.dump(settings, f)
            return ""
        except OSError as e:
            return make_error_string(e)

    @staticmethod
    def is_enabled() -> bool:
        return VoiceCheck._load_settings().get("enabled", False) is True

    @staticmethod
    def set_enabled(value: bool) -> str:
        return VoiceCheck._save_setting("enabled", bool(value))

    @staticmethod
    def get_threshold_offset() -> float:
        value = VoiceCheck._load_settings().get("threshold_offset", None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and VoiceCheck.THRESHOLD_OFFSET_MIN <= value <= VoiceCheck.THRESHOLD_OFFSET_MAX:
            return float(value)
        return VoiceCheck.THRESHOLD_OFFSET_DEFAULT

    @staticmethod
    def set_threshold_offset(value: float) -> str:
        return VoiceCheck._save_setting("threshold_offset", float(value))

    @staticmethod
    def get_max_attempts() -> int:
        value = VoiceCheck._load_settings().get("max_attempts", None)
        if isinstance(value, int) and not isinstance(value, bool) \
                and VoiceCheck.MAX_ATTEMPTS_MIN <= value <= VoiceCheck.MAX_ATTEMPTS_MAX:
            return value
        return VoiceCheck.MAX_ATTEMPTS_DEFAULT

    @staticmethod
    def set_max_attempts(value: int) -> str:
        return VoiceCheck._save_setting("max_attempts", int(value))

    @staticmethod
    def get_min_temperature() -> float:
        value = VoiceCheck._load_settings().get("min_temperature", None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and VoiceCheck.RETRY_TEMPERATURE_MIN_MIN <= value <= VoiceCheck.RETRY_TEMPERATURE_MIN_MAX:
            return float(value)
        return VoiceCheck.RETRY_TEMPERATURE_MIN_DEFAULT

    @staticmethod
    def set_min_temperature(value: float) -> str:
        return VoiceCheck._save_setting("min_temperature", float(value))

    # --- Schedule

    @staticmethod
    def get_min_temperature_for_text(text: str, min_temperature: float) -> float:
        """
        Min temperature for a line: the setting, but not below the limit for
        short lines (counting letters and digits only).
        """
        length = sum(1 for char in text if char.isalnum())
        for max_length, limit in VoiceCheck.RETRY_TEMPERATURE_MIN_BY_LENGTH:
            if length < max_length:
                return max(min_temperature, limit)
        return min_temperature

    @staticmethod
    def _make_length_rule_string() -> str:
        """ Eg: "<6: 0.5, <12: 0.4, <18: 0.3, <24: 0.2" """
        return ", ".join(f"<{n}: {t:.1f}" for n, t in VoiceCheck.RETRY_TEMPERATURE_MIN_BY_LENGTH)

    @staticmethod
    def get_temperature(start_temperature: float, retry_count: int, min_temperature: float) -> float:
        """
        Temperature for an attempt (retry_count 0 is attempt 1).
        Never higher than `start_temperature`.
        """
        if retry_count <= 0:
            return start_temperature
        lowered = round(start_temperature - VoiceCheck.RETRY_TEMPERATURE_DELTA * retry_count, 3)
        return min(start_temperature, max(min_temperature, lowered))

    @staticmethod
    def get_duration_threshold(duration: float, offset: float) -> float:
        """ Threshold for a segment of the given duration (seconds), before any reduction """
        for max_duration, threshold in VoiceCheck.THRESHOLD_BY_DURATION:
            if duration < max_duration:
                return round(threshold + offset, 6)
        return round(VoiceCheck.THRESHOLD_BY_DURATION[-1][1] + offset, 6)

    @staticmethod
    def get_threshold_reduction(start_temperature: float, retry_count: int, min_temperature: float) -> float:
        """
        Amount by which the threshold is lowered for an attempt (retry_count 0 is attempt 1).
        The attempt that first uses the minimum temperature is not lowered;
        each further attempt lowers it by THRESHOLD_STEP.
        """
        for min_retry_count in range(retry_count + 1):
            temperature = VoiceCheck.get_temperature(start_temperature, min_retry_count, min_temperature)
            if temperature <= min_temperature + 1e-9:
                steps = retry_count - min_retry_count
                return round(VoiceCheck.THRESHOLD_STEP * steps, 6)
        return 0.0

    @staticmethod
    def get_max_retries(validation_result: ValidationResult, max_retries: int) -> int:
        """ Retry limit for a failed validation result """
        if isinstance(validation_result, (VoiceMismatchResult, VoiceMismatchSkippedResult)):
            return VoiceCheck.get_max_attempts() - 1
        return max_retries

    @staticmethod
    @contextmanager
    def attempt_scope(project, indices: list[int], retry_counts: list[int]) -> Iterator[None]:
        """
        Wraps the generation of one batch.

        Sets the similarity threshold reduction of each line for this attempt and, if the
        batch contains a retry of a line whose previous attempt failed the voice
        check, lowers the project's Qwen3 temperature (in memory only).
        Also limits the generation length according to the text length, so that
        a runaway generation (no end of speech) cannot run into GEN_TIMEOUT.
        The original temperature and generation length are restored on exit.
        """
        VoiceCheck._threshold_reductions = {}
        # A first attempt starts a new sequence for the line (eg, in a new run)
        for index, retry_count in zip(indices, retry_counts):
            if retry_count == 0:
                VoiceCheck._best_mismatches.pop(index, None)
        if not VoiceCheck.is_enabled():
            yield
            return

        original = project.qwen3_temperature
        start = original if original != -1 else VoiceCheck._get_default_temperature()

        min_temperature_setting = VoiceCheck.get_min_temperature()
        voice_retry_temperatures = []
        for index, retry_count in zip(indices, retry_counts):
            text = project.phrase_groups[index].as_flattened_phrase().text
            min_temperature = VoiceCheck.get_min_temperature_for_text(text, min_temperature_setting)
            VoiceCheck._threshold_reductions[index] = \
                VoiceCheck.get_threshold_reduction(start, retry_count, min_temperature)
            if retry_count > 0 and index in VoiceCheck._mismatch_indices:
                voice_retry_temperatures.append(VoiceCheck.get_temperature(start, retry_count, min_temperature))

        texts = [project.phrase_groups[index].as_flattened_phrase().text for index in indices]
        max_new_tokens = VoiceCheck.get_max_new_tokens(texts)
        is_generate_limited = VoiceCheck._limit_generate(max_new_tokens)

        if voice_retry_temperatures:
            project.qwen3_temperature = min(voice_retry_temperatures)
        try:
            yield
        finally:
            project.qwen3_temperature = original
            if is_generate_limited:
                VoiceCheck._unlimit_generate()

    @staticmethod
    def get_max_new_tokens(texts: list[str]) -> int:
        """
        Generation length limit (in codec frames) for a batch, from its longest
        text: MAX_AUDIO_SECONDS_BASE + MAX_AUDIO_SECONDS_PER_LETTER per letter/digit
        """
        letters = max((sum(1 for char in text if char.isalnum()) for text in texts), default=0)
        seconds = VoiceCheck.MAX_AUDIO_SECONDS_BASE + VoiceCheck.MAX_AUDIO_SECONDS_PER_LETTER * letters
        return int(math.ceil(seconds * VoiceCheck.CODEC_FRAMES_PER_SECOND))

    @staticmethod
    def _get_qwen3_generate_target():
        """ The object whose generate() Qwen3Model calls (Qwen3TTSForConditionalGeneration), or None """
        from tts_audiobook_tool.tts import Tts

        try:
            instance = Tts.get_instance_if_exists()
        except Exception:
            return None
        if instance is None or type(instance).__name__ != "Qwen3Model":
            return None
        wrapper = getattr(instance, "_model", None)
        return getattr(wrapper, "model", None)

    @staticmethod
    def _limit_generate(max_new_tokens: int) -> bool:
        """
        Makes generate() of the loaded Qwen3 model use `max_new_tokens` unless
        given explicitly (instance attribute shadowing the class method).
        Returns True if applied.
        """
        target = VoiceCheck._get_qwen3_generate_target()
        if target is None or "generate" in vars(target):
            return False
        original_generate = target.generate

        def generate(*args, **kwargs):
            kwargs.setdefault("max_new_tokens", max_new_tokens)
            return original_generate(*args, **kwargs)

        target.generate = generate
        return True

    @staticmethod
    def _unlimit_generate() -> None:
        target = VoiceCheck._get_qwen3_generate_target()
        if target is not None and "generate" in vars(target):
            del target.generate

    @staticmethod
    def _get_default_temperature() -> float:
        """ Temperature used by the loaded Qwen3 model when the project value is -1 """
        from tts_audiobook_tool.tts import Tts
        from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel

        instance = Tts.get_instance_if_exists()
        model = getattr(instance, "_model", None)
        defaults = getattr(model, "generate_defaults", None) or {}
        value = defaults.get("temperature", None)
        return float(value) if value is not None else Qwen3BaseModel.TEMPERATURE_FALLBACK_DEFAULT

    # --- Validation

    @staticmethod
    def apply(validation_result: ValidationResult, index: int) -> tuple[ValidationResult, str]:
        """
        Returns (validation_result, ui_line).

        `validation_result` is returned unchanged if the check is disabled, not
        applicable, or passes; else a VoiceMismatchResult is returned. If an
        earlier attempt of the line had a higher similarity, the result is
        marked with `kept_similarity` and must not be saved (see
        is_keeping_earlier_take()).
        `ui_line` shows the similarity of a passing result (to be appended to
        its validation message), else is empty.
        """
        VoiceCheck._mismatch_indices.discard(index)
        if not isinstance(validation_result, (TranscriptResult, SkippedResult)) or validation_result.is_fail:
            return validation_result, ""
        if not VoiceCheck.is_enabled():
            VoiceCheck._best_mismatches.pop(index, None)
            return validation_result, ""

        try:
            similarity = VoiceCheck.get_speaker_similarity(validation_result.sound)
        except Exception as e:
            printt(f"{COL_ERROR}Voice similarity check failed: {make_error_string(e)}")
            return validation_result, ""
        if similarity is None:
            VoiceCheck._best_mismatches.pop(index, None)
            return validation_result, ""

        duration = validation_result.sound.duration
        threshold = round(
            VoiceCheck.get_duration_threshold(duration, VoiceCheck.get_threshold_offset())
            - VoiceCheck._threshold_reductions.get(index, 0.0),
            6
        )
        if similarity >= threshold:
            VoiceCheck._best_mismatches.pop(index, None)
            ui_line = (
                f"\n{COL_DEFAULT}Voice similarity: {COL_DIM}{similarity:.3f} "
                f"(threshold {threshold:.3f} for {duration:.1f}s)"
            )
            return validation_result, ui_line

        result: ValidationResult
        if isinstance(validation_result, TranscriptResult):
            result = VoiceMismatchResult(
                sound=validation_result.sound,
                transcript_words=validation_result.transcript_words,
                similarity=similarity,
                threshold=threshold,
                findings=ValidationFindings(
                    transcript_errors=list(validation_result.findings.transcript_errors),
                    possible_truncation=validation_result.findings.possible_truncation,
                    invalid_reason=ValidationInvalidReason.VOICE_MISMATCH,
                ),
            )
        else:
            result = VoiceMismatchSkippedResult(
                sound=validation_result.sound,
                similarity=similarity,
                threshold=threshold,
                findings=ValidationFindings(invalid_reason=ValidationInvalidReason.VOICE_MISMATCH),
            )
        result.intra_sample_silence_trims = validation_result.intra_sample_silence_trims
        result.generated_start_trim_time = validation_result.generated_start_trim_time
        result.generated_end_trim_time = validation_result.generated_end_trim_time
        result.generated_trim_original_duration = validation_result.generated_trim_original_duration
        result.trailing_token_noise_trim_time = validation_result.trailing_token_noise_trim_time
        result.voice_tag = validation_result.voice_tag
        VoiceCheck._mismatch_indices.add(index)

        best_similarity = VoiceCheck._best_mismatches.get(index)
        if best_similarity is not None and best_similarity >= similarity:
            result.kept_similarity = best_similarity # type: ignore
            return result, ""
        VoiceCheck._best_mismatches[index] = similarity
        return result, ""

    @staticmethod
    def is_keeping_earlier_take(validation_result: ValidationResult) -> bool:
        """ True if the result must not be saved, because an earlier, better take is kept """
        return getattr(validation_result, "kept_similarity", None) is not None

    @staticmethod
    def get_speaker_similarity(sound: Sound) -> float | None:
        """
        Cosine similarity between the speaker embedding of `sound` and that of
        the voice sample used by the active Qwen3 model's most recent voice
        clone generation. Returns None when not applicable.
        """
        from tts_audiobook_tool.tts import Tts

        instance = Tts.get_instance_if_exists()
        if instance is None or type(instance).__name__ != "Qwen3Model":
            return None
        model = getattr(instance, "_model", None)
        voice_info = getattr(instance, "_voice_info", None)
        if model is None or not voice_info or not voice_info[0] or not voice_info[1]:
            return None

        import torch
        from tts_audiobook_tool.sound.sound_util import SoundUtil

        # Cached voice clone prompt of the current voice (same as used for generation)
        prompt = instance._get_or_create_voice_clone( # type: ignore
            source_path=voice_info[0],
            transcript=voice_info[1],
            factory=lambda: instance._create_voice_clone_prompt(voice_info), # type: ignore
        )

        sound = SoundUtil.resample_if_necessary(sound, VoiceCheck.SPEAKER_SR)
        audio = np.ascontiguousarray(sound.data, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if len(audio) == 0:
            return None

        with torch.inference_mode():
            embedding = model.model.extract_speaker_embedding(audio=audio, sr=VoiceCheck.SPEAKER_SR)
        a = embedding.float().cpu().reshape(-1)
        b = prompt.ref_spk_embedding.float().cpu().reshape(-1)
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))
