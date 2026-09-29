from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

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


class VoiceFileVerificationResult(NamedTuple):
    """
    Outcome of `ProjectVoiceUtil.verify_voice_files_exist`.

    Saved references are split by why they were reported. A reference whose
    file is merely missing is kept, in memory and on disk alike: the file may
    be one that has not been copied over yet, as when a project's settings
    arrive from another computer, and generation is blocked with a clear
    message until it shows up. Only a reference to a file that exists but
    cannot be decoded is dropped, and that drop is persisted.

    Truthy when anything was reported, which keeps the older boolean usage
    working.
    """
    not_found: dict[str, list[str]]
    corrupt: dict[str, list[str]]
    warnings: list[tuple[str, str, str]]

    @property
    def did_report_any(self) -> bool:
        return bool(self.not_found or self.corrupt)

    def __bool__(self) -> bool:
        return self.did_report_any


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
        """ Returns the project's voice sample file names for the given model type """
        if _settings_registry().voice_binding(tts_model_type.id) is None:
            return []
        value = project.get_model_setting(tts_model_type.id, "file_name")
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        raise Exception(f"Bad value for {tts_model_type.id}.file_name: {value}")

    @staticmethod
    def get_voice_transcript_values(project: Project, tts_model_type: TtsModelType) -> list[str]:
        """ Returns the project's voice transcript values for the given model type """
        if _settings_registry().transcript_binding(tts_model_type.id) is None:
            return []
        value = project.get_model_setting(tts_model_type.id, "transcript")
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        raise Exception(f"Bad value for {tts_model_type.id}.transcript: {value}")

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
        """Collect names outside the voice storage being edited.

        Shared variants have the same owner, so exclude them together. Editing
        the secondary emotion voice instead excludes only that scalar file,
        not IndexTTS 2's primary voice list.
        """
        used: set[str] = set()
        for model_type in TtsModelType.all():
            binding = _settings_registry().voice_binding(model_type.id)
            if binding is None or (binding.group or binding.model_id, binding.name) == exclude_owner:
                continue
            used.update(ProjectVoiceUtil.get_voice_values(project, model_type))
        emo_voice = project.get_model_setting(TtsModelType.INDEXTTS2.id, "emo_voice")
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
        secondary = tts_type == TtsModelType.INDEXTTS2 and is_secondary
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

        has_transcript_storage = _settings_registry().transcript_binding(tts_type.id) is not None

        # Read the existing transcripts before the voice list grows, so appending
        # one sample pairs its transcript positionally with its file name.
        appended_transcripts: list[str] | None = None
        if has_transcript_storage and append:
            appended_transcripts = ProjectVoiceUtil.get_voice_transcript_values(project, tts_type)

        if tts_type == TtsModelType.INDEXTTS2 and is_secondary:
            project.set_model_setting(TtsModelType.INDEXTTS2.id, "emo_voice", dest_file_name)
        else:
            if _settings_registry().voice_binding(tts_type.id) is None:
                raise Exception(f"Unsupported tts type {tts_type}")
            if append:
                values = ProjectVoiceUtil.get_voice_values(project, tts_type)
                project.set_model_setting(tts_type.id, "file_name", values + [dest_file_name])
            else:
                project.set_model_setting(tts_type.id, "file_name", [dest_file_name])

        if has_transcript_storage:
            if append:
                assert appended_transcripts is not None
                project.set_model_setting(tts_type.id, "transcript", appended_transcripts + [transcript])
            else:
                project.set_model_setting(tts_type.id, "transcript", [transcript] if transcript else [])

        if tts_type == TtsModelType.POCKET:
            project.set_model_setting('pocket', 'predefined_voice', "")

        return project.save()

    @staticmethod
    def remove_voice_at_index_and_save(project: Project, tts_type: TtsModelType, index: int) -> str:
        if _settings_registry().voice_binding(tts_type.id) is None:
            raise ValueError(f"Unsupported tts_type: {tts_type}")

        voices = ProjectVoiceUtil.get_voice_values(project, tts_type)
        if index < 0 or index >= len(voices):
            raise IndexError(f"Voice sample index out of range: {index}")

        removed = voices.pop(index)
        project.set_model_setting(tts_type.id, "file_name", voices)

        if _settings_registry().transcript_binding(tts_type.id) is not None:
            transcripts = ProjectVoiceUtil.get_voice_transcript_values(project, tts_type)
            if index < len(transcripts):
                transcripts.pop(index)
            project.set_model_setting(tts_type.id, "transcript", transcripts)

        if tts_type == TtsModelType.POCKET and not voices:
            project.set_model_setting('pocket', 'predefined_voice', "")

        project.save()
        return removed

    @staticmethod
    def clear_voice_and_save(project: Project, tts_type: TtsModelType, is_secondary: bool=False) -> None:
        if tts_type == TtsModelType.INDEXTTS2 and is_secondary:
            project.set_model_setting(TtsModelType.INDEXTTS2.id, "emo_voice", "")
        else:
            if _settings_registry().voice_binding(tts_type.id) is None:
                raise ValueError(f"Unsupported tts_type: {tts_type}")
            project.set_model_setting(tts_type.id, "file_name", [])

        if _settings_registry().transcript_binding(tts_type.id) is not None:
            project.set_model_setting(tts_type.id, "transcript", [])

        if tts_type == TtsModelType.POCKET:
            project.set_model_setting('pocket', 'predefined_voice', "")

        project.save()

    @staticmethod
    def get_voice_label(project: Project) -> str:
        from tts_audiobook_tool.tts import Tts
        if Tts.get_type() == TtsModelType.POCKET:
            if project.get_model_setting('pocket', 'predefined_voice'):
                return project.get_model_setting('pocket', 'predefined_voice')
            value = ProjectVoiceUtil.get_primary_voice_value(project, TtsModelType.POCKET)
            if not value:
                return "none"
            return ellipsize_path_for_menu(value.removesuffix("_pocket.flac"))

        value = ProjectVoiceUtil.get_primary_voice_value(project, Tts.get_type())
        if not value:
            return "none"
        value = value.removesuffix(f"_{Tts.get_type().value.file_tag}.flac")
        return ellipsize_path_for_menu(value)

    @staticmethod
    def has_voice(project: Project) -> bool:
        from tts_audiobook_tool.tts import Tts
        if Tts.get_type() == TtsModelType.POCKET:
            return bool(project.get_model_setting('pocket', 'predefined_voice') or ProjectVoiceUtil.get_primary_voice_value(project, TtsModelType.POCKET))
        value = ProjectVoiceUtil.get_primary_voice_value(project, Tts.get_type())
        return bool(value)

    @staticmethod
    def emo_vector_to_string(project: Project) -> str:
        if not project.get_model_setting('indextts2', 'emo_vector') or sum(project.get_model_setting('indextts2', 'emo_vector')) == 0:
            return "none"
        strings = []
        for item in project.get_model_setting('indextts2', 'emo_vector'):
            string = f"{item:.1f}".replace(".0", "")
            strings.append(string)
        return ",".join(strings)

    @staticmethod
    def verify_voice_files_exist(project: Project) -> VoiceFileVerificationResult:
        from tts_audiobook_tool.tts import Tts
        model_type = Tts.get_type()
        info = model_type.value

        has_voice_storage = _settings_registry().voice_binding(model_type.id) is not None
        has_emo_voice = model_type == TtsModelType.INDEXTTS2
        if not has_voice_storage and not has_emo_voice:
            return VoiceFileVerificationResult({}, {}, [])

        warnings = []
        not_found_by_attr: dict[str, list[str]] = {}
        corrupt_by_attr: dict[str, list[str]] = {}
        # Labels are display-only grouping keys for the report below.
        file_names_by_attr: list[tuple[str, list[str]]] = []
        if has_voice_storage:
            file_names_by_attr.append(("voice samples", ProjectVoiceUtil.get_voice_values(project, model_type)))
        if has_emo_voice:
            value = project.get_model_setting(model_type.id, "emo_voice")
            file_names_by_attr.append(("emotion voice sample", [value] if value else []))

        for attrib, file_names in file_names_by_attr:
            kept_file_names = []
            kept_indices = []
            for index, file_name in enumerate(file_names):
                file_path = ProjectVoiceUtil.resolve_voice_file_path(project, file_name)
                if not os.path.exists(file_path):
                    # Deliberately kept as saved. The file may simply not have
                    # been copied over yet, as when a project's settings arrive
                    # from another computer; clearing it here would lose the
                    # reference on the next save. Generation is blocked with a
                    # clear message by `get_missing_voice_file_issue` until the
                    # file shows up.
                    warnings.append((attrib, file_name, "file not found"))
                    not_found_by_attr.setdefault(attrib, []).append(file_name)
                    kept_file_names.append(file_name)
                    kept_indices.append(index)
                    continue

                err = SoundFileUtil.is_valid_sound_file(file_path)
                if err:
                    warnings.append((attrib, file_name, err))
                    corrupt_by_attr.setdefault(attrib, []).append(file_name)
                    continue

                kept_file_names.append(file_name)
                kept_indices.append(index)

            if len(kept_file_names) != len(file_names):
                if attrib == "voice samples":
                    transcripts = ProjectVoiceUtil.get_voice_transcript_values(project, model_type)
                    project.set_model_setting(model_type.id, "file_name", kept_file_names)
                    if _settings_registry().transcript_binding(model_type.id) is not None:
                        project.set_model_setting(model_type.id, "transcript", [transcripts[i] for i in kept_indices if i < len(transcripts)])
                else:
                    # Named secondary voice files are scalar strings, not voice lists.
                    project.set_model_setting(model_type.id, "emo_voice", kept_file_names[0] if kept_file_names else "")

        if warnings:
            printt(f"{COL_ERROR}Warning/info: {COL_DEFAULT}Problem with saved voice clone file(s) for current model {COL_ACCENT}{info.ui['proper_name']}{COL_DEFAULT}")
            for attrib, file_name, reason in warnings:
                printt(f"- {COL_ACCENT}{attrib}{COL_DEFAULT}: {file_name}")
                printt(f"  {COL_DIM}{reason}{COL_DEFAULT}")
            printt("Saved reference(s) whose file could not be read are dropped.")
            printt("Reference(s) whose file is simply missing are kept.")
            printt()

        return VoiceFileVerificationResult(not_found_by_attr, corrupt_by_attr, warnings)

    @staticmethod
    def get_batch_size(project: Project) -> int:
        from tts_audiobook_tool.tts import Tts
        binding = _settings_registry().orchestration_binding(Tts.get_type().id)
        if binding is None:
            return 1
        value = project.get_model_setting(Tts.get_type().id, binding.name)
        if value == -1:
            value = PROJECT_BATCH_SIZE_DEFAULT
        elif value > PROJECT_BATCH_SIZE_MAX:
            value = PROJECT_BATCH_SIZE_MAX
        return value

    @staticmethod
    def set_batch_size(project: Project, value: int) -> None:
        from tts_audiobook_tool.tts import Tts
        binding = _settings_registry().orchestration_binding(Tts.get_type().id)
        if binding is None:
            raise ValueError(f"No support for batch_size for the current model")
        if value > PROJECT_BATCH_SIZE_MAX:
            value = PROJECT_BATCH_SIZE_MAX
        project.set_model_setting(Tts.get_type().id, binding.name, value)
        project.save()

    @staticmethod
    def is_language_cjk(project: Project) -> bool:
        if project.language_code in ["zh", "ja", "ko"]:
            return True
        if project.language_code.startswith(("zh-", "ja-", "ko-")):
            return True
        return False
