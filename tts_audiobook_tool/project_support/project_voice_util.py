from __future__ import annotations

import os
from pathlib import Path
import tempfile
import uuid
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

    # Minimum duration of a crop's net span; shorter spans give models too
    # little reference audio to imitate reliably.
    MIN_VOICE_CROP_DURATION_S = 2.0

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
        from tts_audiobook_tool.project_support.voice_reference_migration import (
            VOICE_CROP_SUBDIR, normalize_crop_file_name,
        )
        crop_parts = file_name.replace("\\", "/").split("/")
        if (len(crop_parts) == 2 and crop_parts[0] == VOICE_CROP_SUBDIR
                and normalize_crop_file_name(crop_parts[1]) is not None):
            # Generation passes a derived voice-relative path, not the stored
            # basename. Managed crops never use legacy/root/basename guesses.
            return ProjectVoiceUtil.resolve_cropped_voice_file_path(project, {"crop_file_name": crop_parts[1]})

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
    def get_crop_range(entry: dict) -> tuple[float, float] | None:
        """Return the entry's validated crop range in seconds, or None."""
        from tts_audiobook_tool.project_support.voice_reference_migration import parse_crop_range
        return parse_crop_range(entry)

    @staticmethod
    def get_cropped_voice_relative_path(entry: dict) -> str:
        """Derive the voice-relative crop path from its stored basename."""
        from tts_audiobook_tool.project_support.voice_reference_migration import (
            VOICE_CROP_SUBDIR, normalize_crop_file_name,
        )
        crop_file_name = normalize_crop_file_name(entry.get("crop_file_name"))
        if crop_file_name is None:
            raise ValueError("Voice reference has no valid explicit trim file")
        return f"{VOICE_CROP_SUBDIR}/{crop_file_name}"

    @staticmethod
    def resolve_cropped_voice_file_path(project: Project, entry: dict) -> str:
        """Resolve the crop basename under voice/crops, without legacy guesses."""
        return path_norm.join_project_relative(
            ProjectVoiceUtil.get_voice_dir_path(project),
            ProjectVoiceUtil.get_cropped_voice_relative_path(entry),
        )

    @staticmethod
    def effective_voice_file_path(project: Project, entry: dict) -> str:
        """Resolve the active crop when present, otherwise the original sample."""
        if ProjectVoiceUtil.get_crop_range(entry) is not None:
            cropped_path = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry)
            if os.path.isfile(cropped_path):
                return cropped_path
        return ProjectVoiceUtil.resolve_voice_file_path(project, entry["file_name"])

    @staticmethod
    def effective_voice_reference(
            project: Project,
            tts_model_type: TtsModelType,
            voice_selection_index: int,
    ) -> tuple[str, str]:
        """The (file_name, transcript) pair generation should actually use.

        Applies the entry's crop in lockstep when one is set and its cropped
        file exists on disk; otherwise returns the original pair. File and
        transcript must never switch independently: the clone-cache key and
        rolling-continuation voice identity are both the (path, transcript)
        pair.
        """
        voices = ProjectVoiceUtil.get_voice_values(project, tts_model_type)
        if not voices:
            return "", ""
        index = voice_selection_index % len(voices)
        entry = project.voice_references[index]
        if ProjectVoiceUtil.get_crop_range(entry) is not None:
            cropped_path = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry)
            if os.path.isfile(cropped_path):
                return ProjectVoiceUtil.get_cropped_voice_relative_path(entry), entry.get("crop_transcript", "")
            # Cropped file missing (manually deleted, partial copy): fall back
            # to the original rather than blocking generation. Pre-flight
            # validation surfaces this as a problem to fix.
        transcripts = ProjectVoiceUtil.get_voice_transcript_values(project, tts_model_type)
        return voices[index], transcripts[index] if index < len(transcripts) else ""

    @staticmethod
    def current_voice_value(project: Project, tts_model_type: TtsModelType, voice_selection_index: int) -> str:
        return ProjectVoiceUtil.effective_voice_reference(
            project, tts_model_type, voice_selection_index
        )[0]

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
        return ProjectVoiceUtil.effective_voice_reference(
            project, tts_model_type, voice_selection_index
        )

    @staticmethod
    def voice_reference_pairs(project: Project, tts_model_type: TtsModelType) -> list[tuple[str, str]]:
        if not ProjectVoiceUtil.get_voice_values(project, tts_model_type):
            return []
        return [
            ProjectVoiceUtil.effective_voice_reference(project, tts_model_type, index)
            for index in range(len(project.voice_references))
        ]

    @staticmethod
    def get_used_voice_file_names(project: Project, exclude_owner: tuple[str, str] | None, *, exclude_secondary: bool=False) -> set[str]:
        """Collect names outside the storage being edited.

        All catalog voice bindings now identify one shared list; any primary
        owner excludes that whole list. Secondary edits exclude only emo_voice.
        """
        used: set[str] = {
            ProjectVoiceUtil.get_cropped_voice_relative_path(entry) for entry in project.voice_references
            if ProjectVoiceUtil.get_crop_range(entry) is not None
        }
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
        # Imports never enter the application's managed crop namespace.
        from tts_audiobook_tool.project_support.voice_reference_migration import VOICE_CROP_SUBDIR
        stem_parts = path_norm.split_relative(voice_file_stem)
        if len(stem_parts) > 1 and stem_parts[0].casefold() == VOICE_CROP_SUBDIR:
            return "Voice samples cannot be imported into the managed trim directory"
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
        crop_dir = voice_dir / VOICE_CROP_SUBDIR
        if dest_path.resolve().is_relative_to(crop_dir.resolve()):
            return "Voice samples cannot be imported into the managed trim directory"
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
    def set_voice_crop_transcript_at_index_and_save(project: Project, index: int, transcript: str) -> str:
        """Edit one entry's crop transcript (the cropped span's transcript)."""
        if index < 0 or index >= len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")
        entries = list(project.voice_references)
        entries[index] = {**entries[index], "crop_transcript": transcript}
        project.voice_references = entries
        return project.save()

    @staticmethod
    def _crop_path_is_shared(project: Project, index: int, crop_path: str) -> bool:
        """Never overwrite/delete a crop that also identifies another input."""
        def normalized(path: str) -> str:
            return os.path.normcase(os.path.realpath(os.path.abspath(path)))

        target = normalized(crop_path)
        # Include this entry's original too: corrupt crop metadata must not
        # grant permission to overwrite its own source file.
        original_names = {entry["file_name"] for entry in project.voice_references}
        secondary = project.get_model_setting("indextts2_local", "emo_voice")
        if secondary:
            original_names.add(secondary)
        for name in original_names:
            if normalized(ProjectVoiceUtil.resolve_voice_file_path(project, name)) == target:
                return True
        for other_index, entry in enumerate(project.voice_references):
            if other_index != index and ProjectVoiceUtil.get_crop_range(entry) is not None:
                if normalized(ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry)) == target:
                    return True
        return False

    @staticmethod
    def _allocate_crop_file(project: Project, index: int) -> tuple[str, str]:
        """Reserve a fresh unique path; unrelated existing files are never reused."""
        from tts_audiobook_tool.project_support.voice_reference_migration import VOICE_CROP_SUBDIR
        crop_dir = Path(ProjectVoiceUtil.get_voice_dir_path(project)) / VOICE_CROP_SUBDIR
        crop_dir.mkdir(parents=True, exist_ok=True)
        while True:
            crop_file_name = f"{uuid.uuid4().hex}.flac"
            crop_path = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, {"crop_file_name": crop_file_name})
            if ProjectVoiceUtil._crop_path_is_shared(project, index, crop_path):
                continue
            try:
                fd = os.open(crop_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            os.close(fd)
            return crop_file_name, crop_path

    @staticmethod
    def apply_voice_crop_and_save(
            project: Project,
            index: int,
            start_s: float,
            end_s: float,
            crop_transcript: str,
    ) -> str:
        """Materialize and activate a crop for one voice sample entry.

        Allocates an explicitly owned unique file under voice/crops, retaining
        its name on subsequent edits unless another reference shares the path.
        The original sample is left untouched. Returns an error string or ""
        on success.
        """
        if index < 0 or index >= len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")
        if not (start_s >= 0 and end_s > start_s):
            return f"Invalid trim range {start_s}-{end_s}"
        if end_s - start_s < ProjectVoiceUtil.MIN_VOICE_CROP_DURATION_S:
            return (
                f"Trim span must be at least {ProjectVoiceUtil.MIN_VOICE_CROP_DURATION_S:g}s "
                f"(got {end_s - start_s:g}s)"
            )

        from tts_audiobook_tool.sound.sound_util import SoundUtil
        from tts_audiobook_tool.project_support.voice_reference_migration import (
            format_crop_seconds,
        )

        file_name = project.voice_references[index]["file_name"]
        source_path = ProjectVoiceUtil.resolve_voice_file_path(project, file_name)
        sound_result = SoundFileUtil.load(source_path)
        if isinstance(sound_result, str):
            return f"Couldn't load voice sample {file_name}: {sound_result}"
        if end_s > sound_result.duration:
            return f"Trim end {end_s}s is beyond the {sound_result.duration:.2f}s duration of {file_name}"
        try:
            trimmed = SoundUtil.trim(sound_result, start_s, end_s)
        except ValueError as e:
            return f"Invalid trim range for {file_name}: {e}"
        if trimmed.data.size == 0:
            return f"Trim of {file_name} is empty"

        # Post-process exactly like an imported voice sample's silence trim,
        # so the saved crop has no dead air at its edges. The crop UI's
        # preview applies the same step; preview always equals what is saved.
        from tts_audiobook_tool.sound.silence_util import SilenceUtil
        trimmed, _, _ = SilenceUtil.trim_silence_ends(trimmed)
        if trimmed.data.size == 0:
            return f"Trim of {file_name} is entirely silence"
        if trimmed.duration < ProjectVoiceUtil.MIN_VOICE_CROP_DURATION_S:
            return (
                f"Trim of {file_name} falls to {trimmed.duration:.1f}s after silence "
                f"trimming (minimum {ProjectVoiceUtil.MIN_VOICE_CROP_DURATION_S:g}s); "
                "adjust the range to include more audio"
            )

        entry = project.voice_references[index]
        crop_file_name = entry.get("crop_file_name") if ProjectVoiceUtil.get_crop_range(entry) is not None else None
        cropped_path = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry) if crop_file_name else ""
        if cropped_path and ProjectVoiceUtil._crop_path_is_shared(project, index, cropped_path):
            # A malformed/shared entry gets its own new crop rather than
            # overwriting an original, emotion clip, or another entry's crop.
            crop_file_name = None
        allocated = False
        committed = False
        temp_path = ""
        try:
            if crop_file_name is None:
                crop_file_name, cropped_path = ProjectVoiceUtil._allocate_crop_file(project, index)
                allocated = True
            Path(cropped_path).parent.mkdir(parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(suffix=".flac", dir=Path(cropped_path).parent)
            os.close(fd)
            save_error = SoundFileUtil.save_flac(trimmed, temp_path)
            if save_error:
                return save_error
            # Replace only this explicit owned destination. Writing a temporary
            # file also avoids truncating an existing crop or following a file
            # symlink through to an unrelated input.
            os.replace(temp_path, cropped_path)
            temp_path = ""
            committed = True
        except OSError as e:
            return f"Couldn't save trimmed file: {e}"
        finally:
            cleanup_paths = [temp_path]
            if allocated and not committed:
                cleanup_paths.append(cropped_path)
            for cleanup_path in cleanup_paths:
                if cleanup_path:
                    try:
                        os.remove(cleanup_path)
                    except OSError:
                        # Failed cleanup leaves only an unreferenced managed
                        # temporary file, never an original or active crop.
                        pass

        entries = list(project.voice_references)
        entries[index] = {
            **entries[index],
            "crop_file_name": crop_file_name,
            "crop_start": format_crop_seconds(start_s),
            "crop_end": format_crop_seconds(end_s),
            "crop_transcript": crop_transcript,
        }
        project.voice_references = entries
        return project.save()

    @staticmethod
    def discard_voice_crop_and_save(project: Project, index: int) -> str:
        """Clear one entry's crop fields and delete its cropped file.

        Returns an error string or "" on success. A missing cropped file is
        tolerated (the entry fields are still cleared).
        """
        if index < 0 or index >= len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")

        from tts_audiobook_tool.project_support.voice_reference_migration import CROP_FIELDS

        entry = project.voice_references[index]
        cropped_path = ""
        if ProjectVoiceUtil.get_crop_range(entry) is not None:
            candidate = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry)
            if not ProjectVoiceUtil._crop_path_is_shared(project, index, candidate):
                cropped_path = candidate

        previous_entries = project.voice_references
        entries = list(previous_entries)
        entries[index] = {k: v for k, v in entry.items() if k not in CROP_FIELDS}
        project.voice_references = entries
        save_error = project.save()
        if save_error:
            project.voice_references = previous_entries
            return save_error
        # Persist the reset first. Never delete a still-active crop when saving
        # fails, and leave files used by any other reference untouched.
        if cropped_path and os.path.lexists(cropped_path):
            try:
                os.remove(cropped_path)
            except OSError as e:
                return f"Trim reset, but couldn't delete trimmed file {cropped_path}: {e}"
        return ""

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
