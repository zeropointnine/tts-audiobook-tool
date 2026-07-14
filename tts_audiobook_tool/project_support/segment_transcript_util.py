import json
import os
from pathlib import Path
from typing import Any

from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool.app_types.force_align_util import ForceAlignUtil
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.app_types.segment_transcript_data import SegmentTranscriptData
from tts_audiobook_tool.project_support.sound_segment_util import get_segment_stt_info_path
from tts_audiobook_tool.system_support import terminal
from tts_audiobook_tool.text_ops.text_normalizer import TextNormalizer
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.app_types.validation_findings import ValidationFindings, ValidationInvalidReason
from tts_audiobook_tool.validator import Validator
from tts_audiobook_tool.app_types.validation_result import POSSIBLE_TRUNCATION_UI_MESSAGE, TranscriptResult, TrimmedResult, ValidationResult
from tts_audiobook_tool.transcriber import Transcriber
from tts_audiobook_tool.util import *

class SegmentTranscriptUtil:

    VERSION = 3
    TYPE = "segment_stt_info"
    EXCEPTION_MUSIC_DETECTED = ValidationInvalidReason.MUSIC_DETECTED.value
    EXCEPTION_EXCESSIVE_DURATION = ValidationInvalidReason.EXCESSIVE_DURATION.value
    EXCEPTION_WORD_ERROR_SENTINEL = ValidationFindings.LEGACY_INVALID_SCORE

    @staticmethod
    def from_validation_result(
            project: Project,
            phrase_group: PhraseGroup,
            index: int,
            validation_result: TranscriptResult
    ) -> SegmentTranscriptData:

        prompt = phrase_group.as_flattened_phrase().text
        source = phrase_group.as_flattened_phrase().text
        transcript = Transcriber.get_flat_text_from_words(validation_result.transcript_words)
        normalized_source, normalized_transcript = TextNormalizer.normalize_source_and_transcript(
            source, transcript, project.language_code
        )
        
        findings = validation_result.findings

        return SegmentTranscriptData(
            version=SegmentTranscriptUtil.VERSION,
            type=SegmentTranscriptUtil.TYPE,
            language_code=project.language_code,
            index_1b=index + 1,
            source=source,
            prompt=prompt,
            transcript=transcript,
            normalized_source=normalized_source,
            normalized_transcript=normalized_transcript,
            timed_phrases=SegmentTranscriptUtil.make_timed_phrases(phrase_group, validation_result),
            transcript_words=Transcriber.words_to_json(validation_result.transcript_words),
            findings=findings,
        )

    @staticmethod
    def make_generation_word_error_count(validation_result: ValidationResult) -> int:
        return validation_result.findings.legacy_filename_score

    @staticmethod
    def get_findings(info: SegmentTranscriptData) -> ValidationFindings:
        return info.findings

    @staticmethod
    def make_timed_phrases(
            phrase_group: PhraseGroup,
            validation_result: TranscriptResult
    ) -> list[TimedPhrase]:

        timed_phrases = ForceAlignUtil.make_timed_phrases(
            phrases=phrase_group.phrases,
            words=validation_result.transcript_words,
            sound_duration=validation_result.sound.duration
        )
        if isinstance(validation_result, TrimmedResult):
            timed_phrases = SegmentTranscriptUtil.adjust_timed_phrases_trimmed(timed_phrases, validation_result)
        return timed_phrases

    @staticmethod
    def adjust_timed_phrases_trimmed(
            timed_phrases: list[TimedPhrase],
            trimmed_result: TrimmedResult
    ) -> list[TimedPhrase]:

        start_offset = trimmed_result.start_time or 0
        end_time = trimmed_result.end_time or trimmed_result.sound.duration
        full_duration = end_time - start_offset

        results = []
        for item in timed_phrases:
            time_start = max(item.time_start - start_offset, 0)
            time_end = min(item.time_end, full_duration)
            new_item = TimedPhrase(text=item.text, time_start=time_start, time_end=time_end)
            results.append(new_item)

        return results

    @staticmethod
    def to_dict(info: SegmentTranscriptData) -> dict[str, Any]:
        return {
            "version": info.version,
            "type": info.type,
            "language_code": info.language_code,
            "index_1b": info.index_1b,
            "source": info.source,
            "prompt": info.prompt,
            "transcript": info.transcript,
            "normalized_source": info.normalized_source,
            "normalized_transcript": info.normalized_transcript,
            "findings": info.findings.to_dict(),
            "timed_phrases": TimedPhrase.timed_phrases_to_dicts(info.timed_phrases),
            "transcript_words": info.transcript_words,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> SegmentTranscriptData | str:
        if payload.get("type") != SegmentTranscriptUtil.TYPE:
            return f"Unsupported segment STT info type: {payload.get('type')}"

        timed_phrase_dicts = payload.get("timed_phrases")
        if not isinstance(timed_phrase_dicts, list):
            return "Missing or invalid timed_phrases"
        timed_phrases = TimedPhrase.dicts_to_timed_phrases(timed_phrase_dicts)
        if isinstance(timed_phrases, str):
            return timed_phrases

        transcript_words = payload.get("transcript_words", [])
        if not isinstance(transcript_words, list):
            return "Missing or invalid transcript_words"

        findings_payload = payload.get("findings", payload.get("validation"))
        findings = None
        if findings_payload is not None:
            if not isinstance(findings_payload, dict):
                return "Missing or invalid findings"
            findings = ValidationFindings.from_dict(findings_payload)
            if isinstance(findings, str):
                return findings
        else:
            invalid_reason = None
            exception = payload.get("exception")
            if exception is not None:
                try:
                    invalid_reason = ValidationInvalidReason(exception)
                except ValueError:
                    # Historical behavior treats an unknown exception as a hard fail.
                    invalid_reason = ValidationInvalidReason.MUSIC_DETECTED
            findings = ValidationFindings(
                transcript_errors=Validator.get_word_errors(
                    str(payload["normalized_source"]),
                    str(payload["normalized_transcript"]),
                    str(payload["language_code"]),
                ),
                possible_truncation=bool(payload.get("possible_truncation", False)),
                invalid_reason=invalid_reason,
            )

        try:
            return SegmentTranscriptData(
                version=int(payload["version"]),
                type=str(payload["type"]),
                language_code=str(payload["language_code"]),
                index_1b=int(payload["index_1b"]),
                source=str(payload["source"]),
                prompt=str(payload["prompt"]),
                transcript=str(payload["transcript"]),
                normalized_source=str(payload["normalized_source"]),
                normalized_transcript=str(payload["normalized_transcript"]),
                timed_phrases=timed_phrases,
                transcript_words=transcript_words,
                findings=findings,
            )
        except Exception as e:
            return make_error_string(e)

    @staticmethod
    def save(path: str | Path, info: SegmentTranscriptData) -> str:
        try:
            json_string = json.dumps(SegmentTranscriptUtil.to_dict(info), indent=4)
            Path(path).write_text(json_string, encoding="utf-8")
        except Exception as e:
            return make_error_string(e)
        return ""

    @staticmethod
    def load(path: str | Path) -> SegmentTranscriptData | str:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            return make_error_string(e)
        if not isinstance(payload, dict):
            return f"Unsupported segment STT info root type: {type(payload)}"
        return SegmentTranscriptUtil.from_dict(payload)

    @staticmethod
    def load_timed_phrases(path: str | Path) -> list[TimedPhrase] | str:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            return make_error_string(e)

        if isinstance(payload, list):
            timed_phrase_dicts = payload
        elif isinstance(payload, dict):
            timed_phrase_dicts = payload.get("timed_phrases")
        else:
            return f"Unsupported timing JSON root type: {type(payload)}"

        if not isinstance(timed_phrase_dicts, list):
            return "Missing or invalid timed_phrases"
        return TimedPhrase.dicts_to_timed_phrases(timed_phrase_dicts)

    @staticmethod
    def get_word_errors(info: SegmentTranscriptData) -> list[str]:
        return Validator.get_word_errors(
            info.normalized_source,
            info.normalized_transcript,
            info.language_code
        )

    @staticmethod
    def get_word_error_count(info: SegmentTranscriptData) -> int:
        return SegmentTranscriptUtil.get_findings(info).legacy_filename_score

    @staticmethod
    def get_threshold(info: SegmentTranscriptData, strictness: Strictness) -> int:
        num_words = app_text.get_word_count(info.normalized_source, vocalizable_only=True)
        return Validator.compute_threshold(num_words, strictness)

    @staticmethod
    def is_failed(info: SegmentTranscriptData, strictness: Strictness) -> bool:
        return SegmentTranscriptUtil.get_findings(info).is_failed(
            SegmentTranscriptUtil.get_threshold(info, strictness)
        )

    @staticmethod
    def print_info(sound_segment_index: int, project: Project) -> None:
        """
        Prints info for a transcribed sound segment using json sidecar file
        """
        
        best_item = project.sound_segments.get_best_item_for(sound_segment_index)
        index_string = str(sound_segment_index + 1)

        if best_item is None:
            printt(f"{COL_DIM}{'-' * 60}")
            printt(f"{COL_DEFAULT}Line: {index_string}")
            printt(f"{COL_ERROR}No generated sound segment found.")
            printt()
            return

        sound_path = Path(os.path.join(project.sound_segments_path, best_item.file_name))
        stt_info_path = get_segment_stt_info_path(sound_path)
        info = SegmentTranscriptUtil.load(stt_info_path)
        if isinstance(info, str):
            timed_phrases = SegmentTranscriptUtil.load_timed_phrases(stt_info_path)
            if not isinstance(timed_phrases, str):
                SegmentTranscriptUtil.print_legacy_info(sound_segment_index, sound_path, best_item, project)
                return

            printt(f"{COL_DIM}{'-' * 60}")
            printt(f"{COL_DEFAULT}Line: {index_string}")
            printt(f"{COL_ERROR}Could not load segment STT info: {info}")
            printt(f"Filename: {text_util.make_terminal_hyperlink(str(sound_path), best_item.file_name, is_file=True)}")
            printt()
            return

        filename = text_util.make_terminal_hyperlink(str(sound_path), best_item.file_name, is_file=True)
        num_word_errors = SegmentTranscriptUtil.get_word_error_count(info)
        threshold = SegmentTranscriptUtil.get_threshold(info, project.strictness)
        findings = SegmentTranscriptUtil.get_findings(info)

        filename_line = f"{COL_DEFAULT}Filename: {COL_DEFAULT}{filename}"
        stroke_width = min( len(text_util.strip_ansi_codes(filename_line)), terminal.get_terminal_width())
        stroke = f"{COL_DIM}{stroke_width * '-'}"
        printt(stroke)
        printt(f"{COL_DEFAULT}Line: {COL_ACCENT}{info.index_1b}, {findings.make_status_message(app_text.get_word_count(info.normalized_source, vocalizable_only=True), threshold)}")
        printt(filename_line)
        if findings.invalid_reason is not None:
            printt(f"{COL_DEFAULT}Invalid reason: {COL_ERROR}{findings.invalid_reason.value}")
        if findings.possible_truncation:
            printt(f"{COL_ERROR}{POSSIBLE_TRUNCATION_UI_MESSAGE}")
        printt()

        SegmentTranscriptUtil.print_stt_details(info, should_show_diff=num_word_errors > 0)

        printt()

    @staticmethod
    def print_stt_details(info: SegmentTranscriptData, should_show_diff: bool) -> None:
        printt(f"{COL_DEFAULT}Source text               : {COL_DIM_ITALICS}{info.source.strip()}")
        printt(f"{COL_DEFAULT}TTS prompt                : {COL_DIM_ITALICS}{info.prompt.strip()}")
        printt(f"{COL_DEFAULT}STT transcript            : {COL_DIM_ITALICS}{info.transcript}")
        printt()
        printt(f"{COL_DEFAULT}Source text normalized    : {COL_DIM_ITALICS}{info.normalized_source}")
        printt(f"{COL_DEFAULT}STT transcript normalized : {COL_DIM_ITALICS}{info.normalized_transcript}")

        if should_show_diff:
            printt()
            printt(f"{COL_ACCENT}Word error visualization: {COL_DIM}[-: missing], [+: extra], [=/=: expected/heard], [!: diagnostic], <word> = skipped uncommon word")
            printt(SegmentTranscriptUtil.make_word_error_visualization(info))

    @staticmethod
    def print_legacy_info(
            sound_segment_index: int,
            sound_path: str | Path,
            sound_segment,
            project: Project,
    ) -> None:
        """
        Prints minimal info for legacy timing-only JSON sidecars.

        Legacy sidecars are lists of timed phrase data, not SegmentSttInfo
        payloads. They do not contain prompt/transcript/normalization data, so
        the only reliable word-error count is the tag parsed from the sound
        segment filename.
        """
        phrase_group = project.phrase_groups[sound_segment_index]
        normalized_source = TextNormalizer.normalize_source(phrase_group.text, project.language_code)
        num_words = app_text.get_word_count(normalized_source, vocalizable_only=True)
        threshold = Validator.compute_threshold(num_words, project.strictness)
        filename = text_util.make_terminal_hyperlink(str(sound_path), sound_segment.file_name, is_file=True)
        num_errors = "?" if sound_segment.num_errors < 0 else str(sound_segment.num_errors)

        filename_line = f"{COL_DEFAULT}Filename: {COL_DEFAULT}{filename}"
        stroke = f"{COL_DIM}{len(text_util.strip_ansi_codes(filename_line)) * '-'}"
        printt(stroke)
        printt(f"{COL_DEFAULT}Line: {COL_DEFAULT}{sound_segment_index + 1}, {COL_ACCENT}word errors detected: {COL_DEFAULT}{num_errors}, {COL_ACCENT}word error threshold: {COL_DEFAULT}{threshold}")
        printt(filename_line)
        printt(f"{COL_DIM_ITALICS}Legacy timing data; detailed STT info unavailable")

    @staticmethod
    def make_word_error_visualization(info: SegmentTranscriptData) -> str:
        path = Validator.get_word_error_alignment(
            info.normalized_source,
            info.normalized_transcript,
            info.language_code,
        )
        parts = []
        for step in path:
            if step.action == "match_direct":
                parts.append(step.source_text)
            elif step.action in ["match_homophone", "uncommon_pass_1", "uncommon_pass_2"]:
                parts.append(f"{COL_DIM}<{COL_DEFAULT}{step.source_text}{COL_DIM}>{Ansi.RESET}")
            elif step.action == "mismatch_sub":
                parts.append(f"{COL_DIM}[=/=: {COL_ERROR}{step.source_text}/{step.transcript_text}{COL_DIM}]{Ansi.RESET}")
            elif step.action == "skip_source":
                parts.append(f"{COL_DIM}[-: {COL_ERROR}{step.source_text}{COL_DIM}]{Ansi.RESET}")
            elif step.action == "skip_transcript":
                parts.append(f"{COL_DIM}[+: {COL_ERROR}{step.transcript_text}{COL_DIM}]{Ansi.RESET}")
        if SegmentTranscriptUtil.get_findings(info).possible_truncation:
            parts.append(f"{COL_DIM}[!: {COL_ERROR}possible truncation{COL_DIM}]{Ansi.RESET}")
        return " ".join(parts)
