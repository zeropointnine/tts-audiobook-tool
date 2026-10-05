from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from tts_audiobook_tool import text_util
from tts_audiobook_tool.constants_config import PROJECT_BATCH_SIZE_DEFAULT, PROJECT_BATCH_SIZE_MAX
from tts_audiobook_tool.constants import PROJECT_VOICE_SUBDIR
from tts_audiobook_tool.app_support import app_text, path_norm
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelSpec, TtsModelType
from tts_audiobook_tool.util import *

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types import Sound
    from tts_audiobook_tool.project import Project


def _settings_registry():
    # ModelSettings registers Chatterbox's enum on import. Import lazily here
    # so the Chatterbox base class can import this module during its startup.
    from tts_audiobook_tool.project_support.model_settings import REGISTRY
    return REGISTRY


class ProjectVoiceUtil:
    """
    Voice/model-specific helpers for `Project`.
    """

    @staticmethod
    def get_voice_dir_path(project: Project) -> str:
        """ Returns the project's standard voice sample file subdir path """
        return os.path.join(project.dir_path, PROJECT_VOICE_SUBDIR)

    @staticmethod
    def resolve_voice_file_path(project: Project, file_name: str) -> str:
        """
        Resolves a saved voice sample file name to its path, preferring the
        project's voice subdir and falling back to the legacy project-dir
        root location for older projects.

        Joined component by component rather than with a bare `os.path.join`:
        a stored value that still carries a leading separator — written by an
        older build, or held by a project object that never passed through the
        load-time normalization funnel — would otherwise discard the project
        directory outright. The literal stored value is tried next, which is
        what lets a POSIX file whose name really does contain a backslash still
        resolve; the bare file name is the last resort, which is what lets a
        path recorded in another machine's grammar still find a sample that has
        been copied in.
        """
        voice_dir = ProjectVoiceUtil.get_voice_dir_path(project)

        name_parts = path_norm.split_relative(file_name)
        if not name_parts:
            return os.path.join(project.dir_path, file_name)

        canonical_name = "/".join(name_parts)
        # (name, is_literal): a literal name must be joined with `os.path.join`
        # so its separators are not translated into project-local components.
        candidates: list[tuple[str, bool]] = [(canonical_name, False)]
        if (
            file_name != canonical_name
            and not path_norm.is_absolute_any_grammar(file_name)
            and not path_norm.has_drive_or_unc(file_name)
        ):
            candidates.append((file_name, True))
        if len(name_parts) > 1:
            candidates.append((name_parts[-1], False))

        for candidate_name, is_literal in candidates:
            for base_dir in (voice_dir, project.dir_path):
                joiner = os.path.join if is_literal else path_norm.join_project_relative
                candidate_path = joiner(base_dir, candidate_name)
                if os.path.exists(candidate_path):
                    return candidate_path

        return path_norm.join_project_relative(project.dir_path, canonical_name)

    @staticmethod
    def get_voice_values(project: Project, tts_model_type: TtsModelType) -> list[str]:
        """Return the shared clone list for a model that supports voice samples."""
        if _settings_registry().voice_binding(tts_model_type.id) is None:
            return []
        return [entry["file_name"] for entry in project.voice_references]

    @staticmethod
    def get_voice_transcript_values(project: Project, tts_model_type: TtsModelType) -> list[str]:
        """Return paired transcripts, even if this model does not require them."""
        if _settings_registry().voice_binding(tts_model_type.id) is None:
            return []
        return [entry.get("transcript", "") for entry in project.voice_references]

    @staticmethod
    def get_primary_voice_value(project: Project, tts_model_type: TtsModelType) -> str:
        values = ProjectVoiceUtil.get_voice_values(project, tts_model_type)
        return values[0] if values else ""

    @staticmethod
    def primary_voice_transcript(project: Project, tts_model_type: TtsModelType) -> str:
        values = ProjectVoiceUtil.get_voice_transcript_values(project, tts_model_type)
        return values[0] if values else ""

    @staticmethod
    def current_voice_value(project: Project, tts_model_type: TtsModelType, voice_selection_index: int) -> str:
        values = ProjectVoiceUtil.get_voice_values(project, tts_model_type)
        if not values:
            return ""
        return values[voice_selection_index % len(values)]

    @staticmethod
    def make_voice_sample_display_label(project: Project, file_name: str, tts_model_spec: TtsModelSpec) -> str:
        """ 
        Standard UI treatment for a Project's voice clone file name
        Strips model type qualifier and file suffix, ellipsizes, and makes it a terminal hyperlink
        """
        suffix = f"_{tts_model_spec.file_tag}.flac"
        value = file_name.removesuffix(suffix)
        value = ellipsize(value, 40)
        path = ProjectVoiceUtil.resolve_voice_file_path(project, file_name)
        if os.path.exists(path):
            value = text_util.make_terminal_hyperlink(path, value, is_file=True)
        return value

    @staticmethod
    def make_voice_file_name_tag(voice_file_name: str, file_tag: str) -> str:
        if not voice_file_name:
            return "none"

        voice_file_name = Path(voice_file_name).stem
        postfix_decorator = "_" + file_tag
        if voice_file_name.endswith(postfix_decorator):
            voice_file_name = voice_file_name[:-len(postfix_decorator)]

        return app_text.sanitize_for_filename(voice_file_name[:30])

    @staticmethod
    def current_voice_reference_pair(
            project: Project,
            tts_model_type: TtsModelType,
            voice_selection_index: int,
    ) -> tuple[str, str]:
        voices = ProjectVoiceUtil.get_voice_values(project, tts_model_type)
        if not voices:
            return "", ""

        index = voice_selection_index % len(voices)
        transcripts = ProjectVoiceUtil.get_voice_transcript_values(project, tts_model_type)
        transcript = transcripts[index] if index < len(transcripts) else ""
        return voices[index], transcript

    @staticmethod
    def voice_reference_pairs(project: Project, tts_model_type: TtsModelType) -> list[tuple[str, str]]:
        voices = ProjectVoiceUtil.get_voice_values(project, tts_model_type)
        transcripts = ProjectVoiceUtil.get_voice_transcript_values(project, tts_model_type)
        return [(voice, transcripts[i] if i < len(transcripts) else "") for i, voice in enumerate(voices)]

    @staticmethod
    def get_used_voice_file_names(project: Project, exclude_owner: tuple[str, str] | None, *, exclude_secondary: bool=False) -> set[str]:
        """Collect names outside the storage being edited.

        All catalog voice bindings now identify one shared list; any primary
        owner excludes that whole list. Secondary edits exclude only emo_voice.
        """
        used: set[str] = set()
        if exclude_owner is None:
            used.update(entry["file_name"] for entry in project.voice_references)
        emo_voice = project.get_model_setting("indextts2_local", "emo_voice")
        if not exclude_secondary and emo_voice:
            used.add(emo_voice)
        return used

    @staticmethod
    def set_voice_and_save(
            project: Project,
            sound: Sound,
            voice_file_stem: str,
            transcript: str,
            tts_type: TtsModelType,
            is_secondary: bool=False,
            append: bool=False,
    ) -> str:
        # Voice sample files are stored undecorated in the project's voice
        # subdir. Disambiguate the stem if a different storage owner already
        # references the same file name.
        secondary = tts_type.id == "indextts2_local" and is_secondary
        binding = None if secondary else _settings_registry().voice_binding(tts_type.id)
        exclude_owner = (binding.group or binding.model_id, binding.name) if binding else None
        used = ProjectVoiceUtil.get_used_voice_file_names(project, exclude_owner, exclude_secondary=secondary)
        if append:
            # When appending, a same-stem sample should not clobber an
            # existing entry; give it a distinct name.
            used.update(ProjectVoiceUtil.get_voice_values(project, tts_type))
        stem = voice_file_stem
        dest_file_name = f"{stem}.flac"
        count = 2
        while dest_file_name in used:
            dest_file_name = f"{stem}_{count}.flac"
            count += 1

        voice_dir = Path(ProjectVoiceUtil.get_voice_dir_path(project))
        dest_path = voice_dir / dest_file_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        err = SoundFileUtil.save_flac(sound, str(dest_path))
        if err:
            return err

        if secondary:
            project.set_model_setting("indextts2_local", "emo_voice", dest_file_name)
        else:
            if binding is None:
                raise ValueError(f"Unsupported tts type {tts_type}")
            entry = {"file_name": dest_file_name, "transcript": transcript}
            project.voice_references = project.voice_references + [entry] if append else [entry]

        if tts_type.id == "pocket_local":
            project.set_model_setting('pocket_local', 'predefined_voice', "")

        return project.save()

    @staticmethod
    def remove_voice_at_index_and_save(project: Project, tts_type: TtsModelType, index: int) -> str:
        if _settings_registry().voice_binding(tts_type.id) is None:
            raise ValueError(f"Unsupported tts_type: {tts_type}")

        voices = ProjectVoiceUtil.get_voice_values(project, tts_type)
        if index < 0 or index >= len(voices):
            raise IndexError(f"Voice sample index out of range: {index}")

        removed = voices[index]
        project.voice_references = project.voice_references[:index] + project.voice_references[index + 1:]

        project.save()
        return removed

    @staticmethod
    def move_voice_at_index_and_save(project: Project, index: int, new_index: int) -> str:
        """Reorder paired references; per-line numeric selections stay unchanged."""
        previous = project.voice_references
        if not (0 <= index < len(previous) and 0 <= new_index < len(previous)):
            raise IndexError("Voice sample index out of range")
        if index == new_index:
            return ""
        entries = list(previous)
        entries.insert(new_index, entries.pop(index))
        project.voice_references = entries
        error = project.save()
        if error:
            project.voice_references = previous
        return error

    @staticmethod
    def clear_voice_and_save(project: Project, tts_type: TtsModelType, is_secondary: bool=False) -> None:
        if tts_type.id == "indextts2_local" and is_secondary:
            project.set_model_setting("indextts2_local", "emo_voice", "")
        else:
            if _settings_registry().voice_binding(tts_type.id) is None:
                raise ValueError(f"Unsupported tts_type: {tts_type}")
            project.voice_references = []

        project.save()

    @staticmethod
    def set_voice_transcript_at_index_and_save(project: Project, index: int, transcript: str) -> str:
        """Edit exactly one shared entry without capability-dependent storage."""
        if index < 0 or index >= len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")
        entries = list(project.voice_references)
        entries[index] = {**entries[index], "transcript": transcript}
        project.voice_references = entries
        return project.save()

    @staticmethod
    def get_voice_label(project: Project) -> str:
        if project.get_tts_model_type().id == "pocket_local":
            if project.get_model_setting('pocket_local', 'predefined_voice'):
                return project.get_model_setting('pocket_local', 'predefined_voice')
            value = ProjectVoiceUtil.get_primary_voice_value(project, TtsModelType.require_by_id("pocket_local"))
            if not value:
                return "none"
            return ellipsize_path_for_menu(value.removesuffix("_pocket.flac"))

        value = ProjectVoiceUtil.get_primary_voice_value(project, project.get_tts_model_type())
        if not value:
            return "none"
        value = value.removesuffix(f"_{project.get_tts_model_type().value.file_tag}.flac")
        return ellipsize_path_for_menu(value)

    @staticmethod
    def has_voice(project: Project) -> bool:
        if project.get_tts_model_type().id == "pocket_local":
            return bool(project.get_model_setting('pocket_local', 'predefined_voice') or ProjectVoiceUtil.get_primary_voice_value(project, TtsModelType.require_by_id("pocket_local")))
        value = ProjectVoiceUtil.get_primary_voice_value(project, project.get_tts_model_type())
        return bool(value)

    @staticmethod
    def emo_vector_to_string(project: Project) -> str:
        if not project.get_model_setting('indextts2_local', 'emo_vector') or sum(project.get_model_setting('indextts2_local', 'emo_vector')) == 0:
            return "none"
        strings = []
        for item in project.get_model_setting('indextts2_local', 'emo_vector'):
            string = f"{item:.1f}".replace(".0", "")
            strings.append(string)
        return ",".join(strings)

    @staticmethod
    def get_batch_size(project: Project) -> int:
        from tts_audiobook_tool.tts import Tts
        model_type = project.get_tts_model_type()
        binding = _settings_registry().orchestration_binding(model_type.id)
        if binding is None:
            return 1
        if not Tts.can_batch(model_type):
            # The catalog declares the model as non-batchable (its server
            # serializes requests). Any stored value stays untouched so the
            # project remains portable and the declaration stays reversible.
            return 1
        value = project.get_model_setting(model_type.id, binding.name)
        if value == -1:
            value = PROJECT_BATCH_SIZE_DEFAULT
        elif value > PROJECT_BATCH_SIZE_MAX:
            value = PROJECT_BATCH_SIZE_MAX
        return value

    @staticmethod
    def set_batch_size(project: Project, value: int) -> None:
        binding = _settings_registry().orchestration_binding(project.get_tts_model_type().id)
        if binding is None:
            raise ValueError(f"No support for batch_size for the current model")
        if value > PROJECT_BATCH_SIZE_MAX:
            value = PROJECT_BATCH_SIZE_MAX
        project.set_model_setting(project.get_tts_model_type().id, binding.name, value)
        project.save()

    @staticmethod
    def is_language_cjk(project: Project) -> bool:
        if project.language_code in ["zh", "ja", "ko"]:
            return True
        if project.language_code.startswith(("zh-", "ja-", "ko-")):
            return True
        return False
