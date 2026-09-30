"""Project settings, metadata and inference adapter for audio.cpp families."""
from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.app_types import ReadinessIssue, Sound, StreamChunkCallback, StreamEndCallback, VoiceDisplayInfo
from tts_audiobook_tool.constants import MAX_WORDS_PER_SEGMENT_RECO_RANGE
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.seed_util import get_random_seed_max
from tts_audiobook_tool.tts_models.audio_cpp_definition import (
    AudioCppModelDefinition,
    AudioCppParameter,
    AudioCppTextParameter,
)
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


class AudioCppSettings:
    """Interpret bounded catalog defaults without changing shared project storage."""

    @staticmethod
    def get(project: Project, parameter: AudioCppParameter | AudioCppTextParameter) -> int | float | str:
        stored = project.get_model_setting(parameter.model_id, parameter.name)
        if isinstance(parameter, AudioCppTextParameter):
            return parameter.default if stored is None else parameter.validate(stored)
        if stored is None or parameter.default_sentinel is not None and stored == parameter.default_sentinel:
            stored = parameter.default
        return parameter.validate(stored)

    @staticmethod
    def set(project: Project, parameter: AudioCppParameter | AudioCppTextParameter,
            value: int | float | str | None) -> str:
        if isinstance(parameter, AudioCppTextParameter):
            if value is None or value == parameter.default:
                project.set_model_setting(parameter.model_id, parameter.name, None, reset=True)
            else:
                project.set_model_setting(parameter.model_id, parameter.name, parameter.validate(value))
        elif value is None:
            project.set_model_setting(parameter.model_id, parameter.name, None, reset=True)
        else:
            project.set_model_setting(parameter.model_id, parameter.name, parameter.validate(value))
        return project.save()


class AudioCppModelSupport:
    """Lightweight metadata provider usable without loading a runtime model."""

    def __init__(self, definition: AudioCppModelDefinition):
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
        return self.definition.max_words_range_reco or MAX_WORDS_PER_SEGMENT_RECO_RANGE

    def get_max_words_exceed_warning(self, project: Project) -> str:
        from tts_audiobook_tool.constants import Ansi
        reco = self.get_max_words_range_reco(project)
        limit = reco[1]
        count = project.book.segmentation_settings.max_words_per_segment
        if count <= limit:
            return ""
        name = reco[2] or self.INFO.ui.get("proper_name") or self.INFO.id
        return (f"{Ansi.ITALICS}Source text's max word length ({count}) exceeds {name} recommended model limit ({limit})\n"
                f"{Ansi.ITALICS}Output accuracy may be degraded")

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
            label = ProjectVoiceUtil.make_voice_sample_display_label(project, voice, self.INFO)
            count = len(ProjectVoiceUtil.get_voice_values(project, self.model_type))
            if count > 1:
                label += f", +{count - 1} more"
        else:
            label = COL_ERROR + ("required" if self.definition.voice_required else "none")
        return VoiceDisplayInfo("Voice clone", "current voice clone", label)

    def get_blocking_issues(self, project: Project, instance: object = None) -> list[ReadinessIssue]:
        issues: list[ReadinessIssue] = []
        voice = self.get_primary_voice_value(project)
        if not voice:
            if self.definition.voice_required:
                issues.append(ReadinessIssue("voice sample", "A voice clone sample is required"))
        else:
            path = ProjectVoiceUtil.resolve_voice_file_path(project, voice)
            if not os.path.isfile(path):
                issues.append(ReadinessIssue("voice sample", f"Voice clone sample file not found: {voice}"))
            # WAV conversion/size checks happen during generation, not redraw.
        if self.definition.reference_transcript and voice:
            transcript = ProjectVoiceUtil.primary_voice_transcript(project, self.model_type)
            if not transcript:
                issues.append(ReadinessIssue("voice clone transcript", "Voice clone transcript required when a voice clone sample is supplied"))
        for parameter in self.definition.parameters.values():
            try:
                AudioCppSettings.get(project, parameter)
            except ValueError as exc:
                issues.append(ReadinessIssue(parameter.name, str(exc)))
        try:
            seed = project.get_model_setting(self.INFO.id, "seed")
            if type(seed) is not int or seed < -1 or seed > 2**32 - 1:
                raise ValueError("seed must be -1 or uint32")
        except ValueError as exc:
            issues.append(ReadinessIssue("seed", str(exc)))
        from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
        snapshot = RemoteTtsDiscovery.get_snapshot()  # cached; never probe per prompt/readiness check
        if snapshot.issue:
            issues.append(ReadinessIssue("audio.cpp server", snapshot.issue.message))
        else:
            from tts_audiobook_tool.tts import Tts
            if (self.model_type, Tts._selected_server_model_id) not in snapshot.candidates:
                issues.append(ReadinessIssue("audio.cpp server", Tts._remote_issue or "Selected audio.cpp model is unavailable"))
        return issues

    def get_warning_issues(self, project: Project) -> list[str]:
        warnings = []
        if not self.definition.voice_required and not self.get_primary_voice_value(project):
            warnings.append("Note: Generated voices may vary because no voice reference has been configured.")
        warning = self.get_max_words_exceed_warning(project)
        if warning:
            warnings.append(warning)
        return warnings

    def should_trim_trailing_token_noise(self, project: Project, instance: object = None) -> bool:
        return False

    def can_hallucinate_music(self, project: Project, instance: object = None) -> bool:
        return False


class AudioCppBackendAdapter:
    """Non-streaming worker object bound to an exact discovered server model ID."""

    def __init__(self, definition: AudioCppModelDefinition, support: AudioCppModelSupport, server_model_id: str):
        if not server_model_id or not server_model_id.strip():
            raise ValueError("An exact configured audio.cpp server model ID is required")
        self.definition = definition
        self.support = support
        self.INFO = definition.spec
        self.server_model_id = server_model_id

    def print_init_if_unloaded(self) -> None:
        """Report anticipated server initialization once at an operation start."""
        AudioCppUtil.print_model_init_if_unloaded(
            AudioCppUtil.get_base_url(), self.server_model_id,
            self.INFO.ui.get("proper_name") or self.INFO.id,
        )

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
        if on_stream_chunk is not None or on_stream_end is not None:
            return f"{self.INFO.ui.get('proper_name') or self.INFO.id} does not support streaming"
        if not prompts:
            return []
        try:
            values = {name: AudioCppSettings.get(project, parameter)
                      for name, parameter in self.definition.parameters.items()}
            seed = -1 if force_random_seed else project.get_model_setting(self.INFO.id, "seed")
            if type(seed) is not int or seed < -1 or seed > 2**32 - 1:
                return "seed must be -1 or uint32"
            if seed == -1:
                seed = random.randrange(get_random_seed_max(2**32 - 1, max_random_seed) + 1)
            voice, transcript = ProjectVoiceUtil.current_voice_reference_pair(
                project, self.support.model_type, voice_selection_index)
            if not voice and self.definition.voice_required:
                return "A voice clone sample is required"
            voice_ref: dict[str, str] | None = None
            if voice:
                if self.definition.reference_transcript and not transcript:
                    # App policy: provide a transcript with reference audio
                    # whenever the family supports reference_text.
                    return "Voice clone transcript required when a voice clone sample is supplied"
                path = ProjectVoiceUtil.resolve_voice_file_path(project, voice)
                voice_ref = {"type": "base64", "data": AudioCppUtil.make_voice_ref(path)}
        except (ValueError, OSError) as exc:
            return str(exc)
        payloads = [self._make_payload(prompt, values, voice_ref, seed, project.language_code,
                                       transcript if self.definition.reference_transcript else "")
                    for prompt in prompts]
        # One request at a time: the audio.cpp server serializes requests per model,
        # so fanning out would only queue them behind its model lock.
        sounds: list[Sound] = []
        for payload in payloads:
            result = AudioCppUtil.generate(AudioCppUtil.get_base_url(), payload,
                                           print_request=print_generation_request)
            if isinstance(result, str):
                return result
            sounds.append(result)
        return sounds

    def _make_payload(self, prompt: str, values: dict[str, int | float | str], voice_ref: dict[str, str] | None,
                      seed: int, language: str, transcript: str) -> dict:
        """Place each declared control where this family's server reads it.

        audio.cpp forwards a fixed set of top-level request fields
        (`temperature`, `top_k`, `top_p`, `seed`, `max_tokens`, ...) into the
        model request; family-specific controls that are not in that set must
        travel in the `options` object instead. `voice_ref` is absent for a
        request that uses an instruction-only (voice design) or plain auto
        voice route.
        """
        payload: dict = {
            "model": self.server_model_id, "input": prompt,
            "response_format": "wav", "seed": seed, "language": language,
        }
        if voice_ref is not None:
            payload["voice_ref"] = voice_ref
        # Family-pinned constants first, then the user's declared controls; a
        # parameter targeting the same key wins over a pinned one.
        options: dict[str, int | float | str] = dict(self.definition.request_options)
        for name, parameter in self.definition.parameters.items():
            value = values[name]
            if isinstance(parameter, AudioCppTextParameter) and value == "":
                # A cleared text control is "unset", not an empty instruction:
                # audio.cpp reads the request option's presence, so an empty
                # string would reach the model as an explicit instruct.
                continue
            if parameter.target == "options":
                options[parameter.request_key] = value
            else:
                payload[parameter.request_key] = value
        if options:
            payload["options"] = options
        if transcript:
            payload["reference_text"] = transcript
        return payload

