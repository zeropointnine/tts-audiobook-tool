from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

from tts_audiobook_tool.app_types import Sound, Strictness, Word
from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool.text_ops.whitelist import Whitelist
from tts_audiobook_tool.sound.silence_util import SilenceUtil
from tts_audiobook_tool.sound.sound_extra_util import SoundExtraUtil
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.text_ops.text_normalizer import TextNormalizer
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.util import *
from tts_audiobook_tool.app_types.validation_result import MusicFailResult, ExcessiveDurationResult, TrimmedResult, ValidationResult, WordErrorResult


@dataclass(frozen=True)
class WordErrorAlignmentStep:
    action: str
    source_text: str
    transcript_text: str


from tts_audiobook_tool.transcriber import Transcriber

class Validator:
    """
    """

    POSSIBLE_TRUNCATION_ERROR = "i:POSSIBLE_TRUNCATION"

    @staticmethod
    def format_source_with_uncommon_words(text: str, language_code: str="") -> str:
        """
        Returns `text` with uncommon source words surrounded in brackets.

        Intended for debug printouts so it is easy to see which source words are
        being treated as wildcard-style uncommon words during validation.
        """
        if not Whitelist.supports_language(language_code):
            return text

        words = text.split()
        highlighted_words = [
            (f"{COL_ERROR}[{word}]{COL_DEFAULT}") if not Whitelist().has(word) else word
            for word in words
        ]
        return " ".join(highlighted_words)

    @staticmethod
    def validate(
        sound: Sound, 
        source: str, 
        transcript_words: list[Word],
        language_code: str,
        strictness: Strictness
    ) -> ValidationResult:
        """
        Calculates word errors.
        Returns either WordErrorResult or TrimmedResult
        """
        possible_truncation = SoundExtraUtil.is_possible_truncation(sound)
        transcript = Transcriber.get_flat_text_from_words(transcript_words)
        _, word_errors, num_words, threshold = Validator.get_word_error_fail(
            source, transcript, language_code=language_code, strictness=strictness
        )
        if possible_truncation:
            word_errors.append(Validator.POSSIBLE_TRUNCATION_ERROR)

        # Always does music detect if instance exists
        if ModelManager.has_yamnet_detector():
            if ModelManager.get_yamnet_detector().has_music(sound, threshold=0.1):
                return MusicFailResult(
                    sound,
                    transcript_words,
                    possible_truncation=possible_truncation,
                )

        word_error_result = WordErrorResult(
            sound=sound,
            transcript_words=transcript_words,
            errors=word_errors,
            num_words=num_words,
            threshold=threshold,
            possible_truncation=possible_truncation,
        )

        if not word_error_result.is_fail:
            # Transcription test has passed; now test for excessively/suspiciously long audio
            if ExcessiveDurationResult.is_excessively_long(source, language_code, sound.duration):
                return ExcessiveDurationResult(
                    sound,
                    transcript_words,
                    sound.duration,
                    possible_truncation=possible_truncation,
                )

        # A possible truncation is an audio-boundary diagnostic rather than an
        # extra transcript word. Do not let semantic trimming convert it into a
        # passing TrimmedResult merely because the genuine words otherwise match.
        trimmed_result = None if possible_truncation else Validator.make_trimmed_result(
            word_error_result,
            source,
            transcript_words,
            language_code,
        )
        
        # TODO: Disabled. Too unreliable. No good workarounds.
        # if trimmed_result and Tts.get_type().value.semantic_trim_last:            
        #    trimmed_result = Validator.make_trimmed_result_end_only(trimmed_result)

        if trimmed_result:
            delta = word_error_result.sound.duration - trimmed_result.sound.duration
            if delta <= 0.1:
                trimmed_result = None

        return trimmed_result if trimmed_result else word_error_result

    @staticmethod
    def make_trimmed_result(
        word_error_result: WordErrorResult,
        source: str, 
        transcript_words: list[Word],
        language_code: str
    ) -> TrimmedResult | None:
        """
        """
        from tts_audiobook_tool.tts import Tts
        if word_error_result.num_errors == 0 and not Tts.get_type().value.semantic_trim_last:
            return None 

        normalized_source = TextNormalizer.normalize_source(source, language_code)
        source_word_count = app_text.get_word_count(normalized_source, vocalizable_only=True)
        overage = len(transcript_words) - source_word_count
        if overage < 0:
            return None
                
        transcript = Transcriber.get_flat_text_from_words(transcript_words)
        normalized_transcript = TextNormalizer.normalize_transcript(
            transcript, normalized_source, language_code
        )

        for i in range(overage + 1):
            
            sub_transcript_words = transcript_words[i:i+source_word_count]
            sub_transcript = Transcriber.get_flat_text_from_words(sub_transcript_words)
            normalized_sub_transcript = TextNormalizer.normalize_transcript(sub_transcript, normalized_source, language_code)

            errors = Validator.get_word_errors(normalized_source, normalized_sub_transcript, language_code)
            if errors:
                continue

            # "Semantic match" found
            first_minus_one = transcript_words[i-1] if i > 0 else None
            first = transcript_words[i]
            last = transcript_words[i + source_word_count - 1]
            last_plus_one = transcript_words[i + source_word_count] if len(transcript_words) > i + source_word_count else None

            if first_minus_one is None:
                start = 0
            else:
                start = (first_minus_one.end + first.start) / 2
                start = SoundExtraUtil.get_local_minima(word_error_result.sound, start)

            if i + source_word_count == len(transcript_words):
                end = word_error_result.sound.duration
            else:
                if last_plus_one is not None:
                    end = (last_plus_one.start + last.end) / 2
                else:
                    end = last.end
                end = min(end, word_error_result.sound.duration)
                end = SoundExtraUtil.get_local_minima(word_error_result.sound, end)

            if start == end: # TODO: revisit
                return None


            new_sound = SoundUtil.trim(word_error_result.sound, start, end)
            new_sound, trim_start, _ = SilenceUtil.trim_silence_ends(new_sound) 

            # Adjust Word timing data # TODO: Untested
            start_offset = start + trim_start
            for word in sub_transcript_words:
                word.start -= start_offset
                word.end -= start_offset            

            return TrimmedResult(
                new_sound,
                sub_transcript_words,
                start,
                end,
                original_duration=word_error_result.sound.duration,
                possible_truncation=word_error_result.possible_truncation,
            )

        return None

    @staticmethod
    def make_trimmed_result_end_only(trimmed_result: TrimmedResult) -> TrimmedResult | None:
        """
        Trims sound from the end of the sound, based on the end time of the last word in the transcript.

        Due to Whisper word time imprecision, should only be applied to TTS output that exhibits frequent
        "appended" hallucinations (ie, Chatterbox).
        """
        
        # Whisper word end time is usually 200+ ms too soon, and sometimes much more than that
        # Trying to be extra-conservative here
        OFFSET = 0.3

        sound = trimmed_result.sound
        
        end = trimmed_result.transcript_words[-1].end
        end += OFFSET
        if end + 0.1 >= sound.duration:
            return trimmed_result
        
        end = SoundExtraUtil.get_local_minima(sound, end)

        new_sound = SoundUtil.trim(sound, 0, end)
        new_sound = SilenceUtil.trim_silence_ends(new_sound, end_only=True)[0]

        # Even after adding 'offset' above, we may have landed in-between phonemes/syllables/words, so
        if not Validator._is_last_word_match(trimmed_result.sound, trimmed_result.transcript_words[-1].word):
            return None

        # Clamp word end times
        for word in trimmed_result.transcript_words:
            word.end = min(word.end, end)

        result = TrimmedResult(
            sound=new_sound,
            transcript_words=trimmed_result.transcript_words,
            start_time=trimmed_result.start_time, end_time=end, 
            original_duration=trimmed_result.original_duration
        )
        return result
    
    @staticmethod
    def _is_last_word_match(sound: Sound, last_word: str) -> bool:
        """
        Returns True if the transcribed last word of the passed-in (trimmed) Sound matches `last_word`,
        and has a high enough probability.
        TODO: Unverified
        """
        transcribed_words = Transcriber.transcribe_to_words(sound, "", Stt.get_variant(), Stt.get_config()) # yek 
        if isinstance(transcribed_words, str):
            return False
        if not transcribed_words:
            return False
        
        last_word = TextNormalizer.normalize_common(last_word)
        transcribed_last = transcribed_words[-1]
        transcribed_last_word = TextNormalizer.normalize_common(transcribed_last.word)
        if transcribed_last_word != last_word:
            return False
        if transcribed_last.probability < 0.66:
            return False
        return True

    @staticmethod
    def compute_threshold(num_words: int, strictness: Strictness) -> int:
        """
        Computes the word-error fail threshold for a given word count and strictness level.
        
        Returns the number of word errors allowed before a segment is considered failed.
        """
        match strictness:
            case Strictness.LOW:
                # 1-10 words = 2, 11-20 words = 3, etc
                return math.ceil(num_words / 10) + 1
            case Strictness.MODERATE:
                # 1-10 words = 1, 11-20 words = 2, etc
                return math.ceil(num_words / 10)
            case Strictness.HIGH:
                # 1-10 words = 0; 11-20 words = 1; etc
                return max(0, math.ceil(num_words / 10) - 1)
            case Strictness.INTOLERANT:
                # 0 word errors allowed regardless of length
                return 0

    @staticmethod
    def get_word_error_fail(
        source: str,
        transcript: str,
        strictness: Strictness,
        language_code: str=""
    ) -> tuple[bool, list[str], int, int]:
        """
        Returns:
            True if failed, word failures, source word count, calculated fail threshold
        """
        normalized_source, normalized_transcript = \
            TextNormalizer.normalize_source_and_transcript(source, transcript, language_code=language_code)
        
        word_errors = \
            Validator.get_word_errors(normalized_source, normalized_transcript, language_code)
        num_word_errors = len(word_errors)
        num_words = app_text.get_word_count(normalized_source, vocalizable_only=True)
        
        fail_threshold = Validator.compute_threshold(num_words, strictness)
            
        return (num_word_errors > fail_threshold), word_errors, num_words, fail_threshold

    @staticmethod
    def get_word_errors(
            normalized_source: str,
            normalized_transcript: str,
            language_code: str="",
            verbose=False
    ) -> list[str]:
        """
        Compares source and transcribed text, and returns list of word errors.

        For en, source word not found in dictionary are treated as "wildcards" (plus potentially one extra word)
        For en, homophones are treated as matches.
        
        Word error codes:
            "d:word" (Deletion - word in source missing from transcript)
            "i:word" (Insertion - word in transcript missing from source)
            "s:word/word" (Substitution - word in source and transcript do not match)

        Uses dynamic programming table (dp)
        """

        highlighted_source = Validator.format_source_with_uncommon_words(
            normalized_source, language_code
        )

        if verbose:
            print("")
            print(f"source: {highlighted_source}")
            print(f"transc: {normalized_transcript}")
            print("")

        path = Validator.get_word_error_alignment(
            normalized_source,
            normalized_transcript,
            language_code,
        )

        failure_codes = []

        for step in path:
            action = step.action
            s_text = step.source_text
            t_text = step.transcript_text
            word_info = f"'{s_text}' vs '{t_text}'"
            if action == "match_direct":
                if verbose:
                    print(f"{word_info} -> direct match")
            elif action == "match_homophone":
                if verbose:
                    print(f"{word_info} -> homophone match")
            elif action == "uncommon_pass_1":
                if verbose:
                    print(f"{word_info} -> no match but giving it a free pass because source word is uncommon")
            elif action == "uncommon_pass_2":
                if verbose:
                    print(f"{word_info} -> no match (2 words) but giving it a free pass because source word is uncommon")
            elif action == "skip_source":
                code = f"d:{s_text}"
                failure_codes.append(code)
                if verbose:
                    print(f"{word_info} -> transcript out of words - fail code: {code}")
            elif action == "skip_transcript":
                code = f"i:{t_text}"
                failure_codes.append(code)
                if verbose:
                    print(f"{word_info} -> source out of words - fail code: {code}")
            elif action == "mismatch_sub":
                code = f"s:{s_text}/{t_text}"
                failure_codes.append(code)
                if verbose:
                    print(f"{word_info} -> mismatch - fail code: {code}")

        return failure_codes

    @staticmethod
    def get_word_error_alignment(
            normalized_source: str,
            normalized_transcript: str,
            language_code: str=""
    ) -> list[WordErrorAlignmentStep]:
        """
        Returns the word alignment path used for word-error detection.

        Actions match the historical internal DP action names used by
        get_word_errors(), allowing callers to render richer diagnostics while
        keeping get_word_errors() backward-compatible.
        """

        def is_match(a: str, b: str) -> bool:
            return (a == b) or sounds_the_same(a, b)

        def sounds_the_same(a: str, b: str) -> bool:
            return TextNormalizer.sounds_the_same_en(a, b) if language_code == "en" else False

        def is_uncommon_word(word: str) -> bool:
            return not Whitelist().has(word) if Whitelist.supports_language(language_code) else False

        source_words = normalized_source.split()
        transcript_words = normalized_transcript.split()
        n = len(source_words)
        m = len(transcript_words)

        # dp[i][j] stores min failures to align source[:i] and transcript[:j]
        dp: list[list[float]] = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        # parent[i][j] stores (prev_i, prev_j, action_description)
        parent: list[list[Optional[tuple[int, int, str]]]] = [[None] * (m + 1) for _ in range(n + 1)]
        
        dp[0][0] = 0

        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0 and j == 0:
                    continue
                
                # 1. Match or Substitution (Source matches Transcript)
                if i > 0 and j > 0:
                    s_word = source_words[i-1]
                    t_word = transcript_words[j-1]
                    
                    matches = is_match(s_word, t_word)
                    is_uncommon = is_uncommon_word(s_word)
                    
                    if matches:
                        cost = dp[i-1][j-1] + 0
                        if cost < dp[i][j]:
                            dp[i][j] = cost
                            parent[i][j] = (i-1, j-1, "match_direct" if s_word == t_word else "match_homophone")
                    elif is_uncommon:
                        # Free pass 1-to-1
                        cost = dp[i-1][j-1] + 0
                        if cost < dp[i][j]:
                            dp[i][j] = cost
                            parent[i][j] = (i-1, j-1, "uncommon_pass_1")
                    else:
                        # Substitution (Mismatch) -> Cost 1
                        cost = dp[i-1][j-1] + 1
                        if cost < dp[i][j]:
                            dp[i][j] = cost
                            parent[i][j] = (i-1, j-1, "mismatch_sub")

                # 2. Uncommon Match 1-to-2
                if i > 0 and j > 1:
                    s_word = source_words[i-1]
                    if is_uncommon_word(s_word):
                        cost = dp[i-1][j-2] + 0
                        if cost < dp[i][j]:
                            dp[i][j] = cost
                            parent[i][j] = (i-1, j-2, "uncommon_pass_2")

                # 3. Skip Source (Deletion) -> Cost 1
                if i > 0:
                    cost = dp[i-1][j] + 1
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        parent[i][j] = (i-1, j, "skip_source")

                # 4. Skip Transcript (Insertion) -> Cost 1
                if j > 0:
                    cost = dp[i][j-1] + 1
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        parent[i][j] = (i, j-1, "skip_transcript")

        # Reconstruct Path
        path: list[WordErrorAlignmentStep] = []
        curr_i, curr_j = n, m

        while curr_i > 0 or curr_j > 0:
            node = parent[curr_i][curr_j]
            if node is None:
                # Should not happen if reachable
                break
            prev_i, prev_j, action = node

            s_sub = source_words[prev_i:curr_i]
            t_sub = transcript_words[prev_j:curr_j]
            s_text = " ".join(s_sub) if s_sub else "-"
            t_text = " ".join(t_sub) if t_sub else "-"

            path.append(WordErrorAlignmentStep(action, s_text, t_text))
            curr_i, curr_j = prev_i, prev_j

        path.reverse()
        return path

    @staticmethod
    def is_unsupported_language_code(code: str) -> bool:
        for item in VALIDATION_UNSUPPORTED_LANGUAGES:
            if code.startswith(item):
                return True
        return False
