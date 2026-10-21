"""Non-runtime model capabilities shared by class-based and configured providers."""
from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelSpec

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


class ModelSupport(Protocol):
    INFO: TtsModelSpec

    def prepare_text_for_inference(self, project: Project, text: str, *, apply_word_substitutions: bool = True) -> str: ...
    def get_blocking_issues(self, project: Project, instance: object = None) -> list[ReadinessIssue]: ...
    def get_output_sample_rate(self, project: Project, instance: object = None) -> int: ...
    def get_menu_text(self, project: Project, instance: object = None) -> str: ...
    def get_voice_tag(self, project: Project) -> str: ...
    def get_voice_tag_for_value(self, value: str) -> str: ...
    def get_max_words_range_reco(self, project: Project, instance: object = None) -> tuple[int, int, str]: ...
    def should_trim_trailing_token_noise(self, project: Project, instance: object = None) -> bool: ...
    def can_hallucinate_music(self, project: Project, instance: object = None) -> bool: ...
    def uses_reference_transcript(self, project: Project) -> bool: ...
    # Runtime authorities for these capabilities; INFO.can_* are only the catalog defaults.
    def can_instructions(self, project: Project, instance: object = None) -> bool: ...
    def can_voice_clone(self, project: Project, instance: object = None) -> bool: ...
    # Whether a voice without a sample still gets a fixed speaker (eg a VibeVoice LoRA),
    # so the "voices may vary" run note does not apply.
    def has_fixed_default_voice(self, project: Project) -> bool: ...
    # Checks user-entered Voice.instructions; returns an error message or "".
    def validate_instructions(self, value: str) -> str: ...
    # Whether an instruction-less voice gets an arbitrary voice (non-blocking hint only).
    def expects_voice_instructions(self, project: Project) -> bool: ...
    # Whether a voice's instructions still apply when it has a voice sample (False: sample-less voices only, eg AuK).
    def uses_instructions_with_sample(self, project: Project) -> bool: ...
    # Whether voices use per-voice preset ids (Voice.presets), which take precedence over samples.
    def can_voice_presets(self, project: Project, instance: object = None) -> bool: ...
    # Whether every used voice needs a preset (eg Qwen3-TTS CustomVoice).
    def requires_voice_preset(self, project: Project) -> bool: ...
    def get_voice_tag_for_preset(self, preset: str) -> str: ...


def max_words_exceeds_recommended(max_words: int, limit: int) -> bool:
    """
    Whether a segmentation `max_words_per_segment` value exceeds a model's
    recommended maximum `limit` (the upper bound of get_max_words_range_reco).

    A non-positive `limit` means the model makes no recommendation, so it is
    never exceeded. Shared by every max-words-per-segment warning site.
    """
    return limit > 0 and max_words > limit


def make_max_words_exceed_warning(project: Project, reco: tuple[int, int, str], fallback_name: str) -> str:
    """
    The pre-generation warning for when the book's segmentation max words per
    segment exceeds the model's recommended maximum (`reco`, as returned by
    get_max_words_range_reco), else "". The model is named by the reco's
    display name, or else `fallback_name`. Shared by every model's
    `get_max_words_exceed_warning`.
    """
    from tts_audiobook_tool.constants import Ansi

    limit = reco[1]
    max_words = project.book.segmentation_settings.max_words_per_segment
    if not max_words_exceeds_recommended(max_words, limit):
        return ""
    name = reco[2] or fallback_name
    return (
        f"{Ansi.ITALICS}Source text's max word length ({max_words})"
        f" exceeds {name} recommended model limit ({limit})\n"
        f"{Ansi.ITALICS}Output accuracy on longer prompts may be degraded"
    )


def make_language_value_note(value: str | None, language_code: str | None, detail: str = "") -> str:
    """
    The informational pre-flight note naming the language value a model receives:
    the value actually sent (after any mapping from the project language code),
    with the project code it was derived from, plus an optional qualifying `detail`
    (eg "applies to IndexTTS2.5 only"). Shared by every model that passes a
    language to inference, eg::

        Language value passed to model: English (project language code = "en")

    A `value` of None or "" means no language reaches the model ("none").
    """
    code = (language_code or "").strip()
    source = f'project language code = "{code}"' if code else "project language code not set"
    if detail:
        source += f"; {detail}"
    return f"Language value passed to model: {value or 'none'} ({source})"
