"""Definition-bound metadata, project access and HTTP runtime for configured models."""
from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
from tts_audiobook_tool.app_types import ReadinessIssue, Sound, StreamChunkCallback, StreamEndCallback, VoiceDisplayInfo
from tts_audiobook_tool.constants import MAX_WORDS_PER_SEGMENT_RECO_RANGE, SEED_MAX
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.seed_util import get_random_seed_max
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.tts_models.model_support import max_words_exceeds_recommended
from tts_audiobook_tool.tts_models.sgl_omni_definition import NumericParameter, SglOmniModelDefinition
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


class ConfiguredSettings:
    """Read variant-specific effective values without modifying shared overrides."""

    @staticmethod
    def stored(project: Project, parameter: NumericParameter, model_id: str | None = None) -> int | float:
        return project.get_model_setting(model_id or parameter.model_id, parameter.name)

    @staticmethod
    def is_unset(project: Project, parameter: NumericParameter, model_id: str | None = None) -> bool:
        stored = ConfiguredSettings.stored(project, parameter, model_id)
        return parameter.default_sentinel is not None and stored == parameter.default_sentinel

    @staticmethod
    def get(project: Project, parameter: NumericParameter, model_id: str | None = None) -> int | float:
        stored = ConfiguredSettings.stored(project, parameter, model_id)
        effective = parameter.default if parameter.default_sentinel is not None and stored == parameter.default_sentinel else stored
        return parameter.validate(effective)

    @staticmethod
    def set(project: Project, parameter: NumericParameter, value: int | float | None, model_id: str | None = None) -> str:
        model_id = model_id or parameter.model_id
        if value is None:
            project.set_model_setting(model_id, parameter.name, None, reset=True)
        else:
            project.set_model_setting(model_id, parameter.name, parameter.validate(value))
        return project.save()


def _cached_server_readiness_issue(model_id: str) -> ReadinessIssue | None:
    snapshot = RemoteTtsDiscovery.get_snapshot()  # Never poll per sentence.
    if snapshot.issue is not None:
        return ReadinessIssue("SGL-Omni server", snapshot.issue.message)
    from tts_audiobook_tool.tts import Tts
    if not Tts._selected_server_model_id or (TtsModelType.require_by_id(model_id), Tts._selected_server_model_id) not in snapshot.candidates:
        return ReadinessIssue("SGL-Omni server", Tts._remote_issue or "Selected server model is incompatible")
    return None


class ConfiguredModelSupport:
    """Class-free metadata provider usable in the interactive process."""

    def __init__(self, definition: SglOmniModelDefinition):
        self.definition = definition
        self.INFO = definition.spec
        self.model_type = TtsModelType.require_by_id(self.INFO.id)

    def massage_for_inference(self, text: str) -> str:
        for before, after in self.INFO.substitutions:
            text = text.replace(before, after)
        return text

    def prepare_text_for_inference(self, project: Project, text: str, *, apply_word_substitutions: bool = True) -> str:
        from tts_audiobook_tool.text_ops.prompt_normalizer import PromptNormalizer
        if apply_word_substitutions:
            text = PromptNormalizer.apply_prompt_word_substitutions(text, project.word_substitutions, project.language_code)
        text = PromptNormalizer.normalize_prompt(text=text, language_code=project.language_code, un_all_caps=self.INFO.un_all_caps)
        return self.massage_for_inference(text)

    def get_output_sample_rate(self, project: Project, instance: object = None) -> int:
        return self.INFO.default_output_sample_rate

    def get_max_words_range_reco(self, project: Project, instance: object = None) -> tuple[int, int, str]:
        return MAX_WORDS_PER_SEGMENT_RECO_RANGE

    def get_max_words_exceed_warning(self, project: Project) -> str:
        # Higgs uses the shared recommended range; no warning when within it.
        limit = self.get_max_words_range_reco(project)[1]
        count = project.book.segmentation_settings.max_words_per_segment
        if not max_words_exceeds_recommended(count, limit):
            return ""
        from tts_audiobook_tool.constants import Ansi
        name = self.INFO.ui.get("proper_name") or self.INFO.id
        return (f"{Ansi.ITALICS}Source text's max word length ({count}) exceeds {name} recommended model limit ({limit})\n"
                f"{Ansi.ITALICS}Output accuracy on longer prompts may be degraded")

    def get_menu_text(self, project: Project, instance: object = None) -> str:
        return self.INFO.ui.get("proper_name") or ""

    def get_primary_voice_value(self, project: Project) -> str:
        return ProjectVoiceUtil.get_primary_voice_value(project, self.model_type)

    def get_voice_tag(self, project: Project) -> str:
        voice = self.get_primary_voice_value(project)
        return self.get_voice_tag_for_value(voice) if voice else "none"

    def get_voice_tag_for_value(self, value: str) -> str:
        return ProjectVoiceUtil.make_voice_file_name_tag(value, self.INFO.file_tag)

    def get_voice_display_info(self, project: Project, instance: object = None) -> VoiceDisplayInfo:
        from tts_audiobook_tool.constants import COL_ERROR
        voice = self.get_primary_voice_value(project)
        if voice:
            value = ProjectVoiceUtil.make_voice_sample_display_label(project, voice, self.INFO)
            count = len(ProjectVoiceUtil.get_voice_values(project, self.model_type))
            if count > 1:
                value += f", +{count - 1} more"
        else:
            value = COL_ERROR + ("required" if self.INFO.requires_voice else "none")
        return VoiceDisplayInfo("Voice clone", "current voice clone", value)

    def uses_reference_transcript(self, project: Project) -> bool:
        """Whether generation requires the voice sample transcript. Mirrors the
        storage binding, so "optional" policies still count as required here
        (the pre-flight fills them in)."""
        from tts_audiobook_tool.project_support.model_settings import REGISTRY
        return REGISTRY.transcript_binding(self.INFO.id) is not None

    def get_blocking_issues(self, project: Project, instance: object = None) -> list[ReadinessIssue]:
        issues = []
        if self.definition.language_policy == "moss":
            server_issue = _cached_server_readiness_issue(self.INFO.id)
            return [server_issue] if server_issue else []
        # Voice-clone state (required sample, file existence, transcripts) is
        # validated lazily by the interactive pre-flight (validate_voices) and
        # at generation time, not by readiness.
        for parameter in self.definition.parameters.values():
            try:
                ConfiguredSettings.get(project, parameter, self.INFO.id)
            except ValueError as exc:
                issues.append(ReadinessIssue(parameter.name, str(exc)))
        server_issue = _cached_server_readiness_issue(self.INFO.id)
        if server_issue:
            issues.append(server_issue)
        return issues

    def get_warning_issues(self, project: Project) -> list[str]:
        warnings = []
        if self.definition.language_policy == "moss":
            from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel
            language = MossBaseModel.get_language_name(project.language_code)
            return [f"Using MOSS-TTS language value: {language or 'None'}"]
        if not self.INFO.requires_voice and not self.get_primary_voice_value(project):
            warnings.append("Note: Generated voices may vary because no voice reference has been configured.")
        warning = self.get_max_words_exceed_warning(project)
        if warning:
            warnings.append(warning)
        top_k = self.definition.parameters.get("top_k")
        if top_k and top_k.max_request_value is not None:
            stored = ConfiguredSettings.stored(project, top_k, self.INFO.id)
            if stored != top_k.default_sentinel and stored > top_k.max_request_value:
                name = "Fish S2 Pro" if self.INFO.id == "fish_s2_sglomni" else self.INFO.ui.get("proper_name", self.INFO.id)
                warnings.append(f"Top_k ({stored}) out of range for server version of {name} inference, will clamp to {top_k.max_request_value}")
        return warnings

    def should_trim_trailing_token_noise(self, project: Project, instance: object = None) -> bool:
        return self.definition.music_and_trim

    def can_hallucinate_music(self, project: Project, instance: object = None) -> bool:
        return self.definition.music_and_trim


class SglOmniBackendAdapter:
    """Runtime generation object; the worker owns and clears this instance."""

    def __init__(self, definition: SglOmniModelDefinition, support: ConfiguredModelSupport):
        self.definition = definition
        self.support = support
        self.INFO = definition.spec

    def prepare_text_for_inference(self, project: Project, text: str, *, apply_word_substitutions: bool = True) -> str:
        return self.support.prepare_text_for_inference(project, text, apply_word_substitutions=apply_word_substitutions)

    def get_device_type(self) -> None:
        return None

    def clear_stream_state(self) -> None:
        pass

    def clear_voice_clone_cache(self) -> None:
        pass

    def clear_continuation(self) -> None:
        pass

    def release_inference_memory(self) -> None:
        pass

    def kill(self) -> None:
        pass

    def get_warning_issues(self, project: Project) -> list[str]:
        return self.support.get_warning_issues(project)

    def generate_using_project(
        self, project: Project, prompts: list[str], force_random_seed: bool = False,
        on_stream_chunk: StreamChunkCallback | None = None, on_stream_end: StreamEndCallback | None = None,
        voice_selection_index: int = 0, print_params: bool = False,
        print_generation_request: bool = False,
        max_random_seed: int = -1,
    ) -> list[Sound] | str:
        # Nonstreaming implementations ignore callback arguments, as the
        # previous server adapters did. Qwen exposes a PCM callback route even
        # though its catalog marks streaming unsupported for normal selection.
        streaming = (on_stream_chunk is not None or on_stream_end is not None) and (
            self.INFO.can_stream or bool(self.definition.request_stream_defaults))
        if streaming and len(prompts) != 1:
            return "Streaming generation supports exactly one prompt"
        policy = self.definition.prompt_policy
        try:
            mapped: dict[str, int | float] = {}
            for parameter in self.definition.parameters.values():
                value = ConfiguredSettings.get(project, parameter, self.INFO.id)
                if parameter.omit_when_unset and ConfiguredSettings.is_unset(project, parameter, self.INFO.id):
                    continue
                if policy == "auk_seconds" and parameter.name == "speed":
                    continue
                mapped[parameter.request_key] = (min(value, parameter.max_request_value)
                                                 if parameter.max_request_value is not None else value)
            voice, transcript = ProjectVoiceUtil.current_voice_reference_pair(
                project, self.support.model_type, voice_selection_index)
            if self.INFO.requires_voice and not voice:
                return "A voice clone sample is required"
            references = None
            path = ""
            if voice:
                if self.definition.transcript_policy == "required_when_voice_present" and not (
                        transcript.strip() if policy == "auk_seconds" else transcript):
                    return ("Voice clone transcript required for every AuK voice sample" if policy == "auk_seconds"
                            else "Voice clone transcript required when a voice clone sample is supplied")
                path = ProjectVoiceUtil.resolve_voice_file_path(project, voice)
                if not os.path.isfile(path):
                    return f"Voice clone sample file not found: {voice}"
                reference = {"audio_path": SoundUtil.make_audio_data_uri(path)}
                if self.definition.transcript_policy != "omitted" and (
                        transcript or policy == "auk_seconds"):
                    reference["text"] = transcript
                references = [reference]
            seed = None
            if self.definition.seed_policy == "resolved":
                seed = -1 if force_random_seed else project.get_model_setting(self.INFO.id, "seed")
                if seed == -1:
                    seed = random.randrange(0, get_random_seed_max(SEED_MAX - 1, max_random_seed) + 1)
            reference_seconds = None
            speed: int | float = 1.0
            if policy == "auk_seconds":
                speed = ConfiguredSettings.get(project, self.definition.parameters["speed"], self.INFO.id)
                if speed != self.definition.parameters["speed"].default:
                    reference_sound = SoundFileUtil.load(path)
                    if isinstance(reference_sound, str):
                        return reference_sound
                    reference_seconds = reference_sound.duration
        except (ValueError, OSError) as exc:
            return str(exc)
        payloads = []
        for prompt in prompts:
            payload = {**self.definition.request_defaults, **mapped, "input": prompt, "stream": streaming}
            if streaming:
                payload.update(self.definition.request_stream_defaults)
            if references:
                payload["references"] = references
            if seed is not None:
                payload["seed"] = seed
            if policy == "zonos_tokens":
                payload["max_new_tokens"] = min(200 + 40 * len(prompt.split()), 4096)
            elif policy == "auk_seconds" and reference_seconds is not None:
                payload["stage_params"] = {"auk_engine": {"gen_seconds": (
                    reference_seconds * len(prompt.encode("utf-8")) /
                    max(1, len(transcript.encode("utf-8"))) / speed)}}
            if self.definition.language_policy == "moss":
                from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel
                language = MossBaseModel.get_language_name(project.language_code)
                if language:
                    payload["language"] = language
            payloads.append(payload)
        if streaming:
            result = SglOmniUtil.generate_streaming(SglOmniUtil.get_base_url(), payloads[0],
                on_stream_chunk=on_stream_chunk, on_stream_end=on_stream_end,
                should_print=print_generation_request, fallback_sample_rate=self.INFO.default_output_sample_rate)
            return result if isinstance(result, str) else [result]
        return SglOmniUtil.generate_concurrent(SglOmniUtil.get_base_url(), payloads,
            print_request=print_generation_request, fallback_sample_rate=self.INFO.default_output_sample_rate)
