from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
import os
from pathlib import Path
import tempfile
import uuid
from typing import TYPE_CHECKING, Literal

from tts_audiobook_tool import text_util
from tts_audiobook_tool.constants_config import PROJECT_BATCH_SIZE_DEFAULT, PROJECT_BATCH_SIZE_MAX
from tts_audiobook_tool.constants import PROJECT_VOICE_SUBDIR
from tts_audiobook_tool.app_support import app_text, path_norm
from tts_audiobook_tool.app_types.voice import VOICE_CROP_SUBDIR, Voice, VoiceCrop, ensure_voice_slot, normalize_crop_file_name
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelSpec, TtsModelType
from tts_audiobook_tool.util import *

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types import RoomReverb, Sound
    from tts_audiobook_tool.project import Project


def _settings_registry():
    # ModelSettings registers Chatterbox's enum on import. Import lazily here
    # so the Chatterbox base class can import this module during its startup.
    from tts_audiobook_tool.project_support.model_settings import REGISTRY
    return REGISTRY


# Max length of instruction previews (voice list, submenu items, status block)
INSTRUCTIONS_PREVIEW_LENGTH = 40

# User-facing voice name (`Voice.name`) validation
VOICE_NAME_MAX_LENGTH = 64
VOICE_NAME_ONE_WORD_ERROR = "Name must be one word (no spaces)"
VOICE_NAME_TOO_LONG_ERROR = f"Name must be at most {VOICE_NAME_MAX_LENGTH} characters"
VOICE_NAME_UNIQUE_ERROR = "Name must be unique"

VoiceSourceKind = Literal["preset", "sample", "instructions", "builtin", "missing"]


@dataclass(frozen=True)
class VoiceSource:
    """What one voice uses to generate under the project's current TTS model
    and settings; see `ProjectVoiceUtil.resolve_voice_source`.

    `kind`:
        "preset"        an active preset voice id (eg a Pocket predefined voice)
        "sample"        a voice clone sample
        "instructions"  free-text voice instructions (eg AuK's voice description)
        "builtin"       no value; the model falls back to its built-in voice
        "missing"       no value, but the model requires one (sample or preset)

    `label` is short display text for the value ("" when there is none): the
    preset id, the hyperlinked sample name, or the quoted, ellipsized instructions.
    """

    kind: VoiceSourceKind
    label: str = ""

    @property
    def has_value(self) -> bool:
        """Whether the voice has a value the model uses (is "valid")."""
        return self.kind in ("preset", "sample", "instructions")


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
        crop_parts = file_name.replace("\\", "/").split("/")
        if (len(crop_parts) == 2 and crop_parts[0] == VOICE_CROP_SUBDIR
                and normalize_crop_file_name(crop_parts[1]) is not None):
            # Generation passes a derived voice-relative path, not the stored
            # basename. Managed crops never use legacy/root/basename guesses.
            return ProjectVoiceUtil.resolve_crop_file_path(project, crop_parts[1])

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
        """Return the shared clone list for a model that supports voice samples.

        One entry per voice slot, so a voice without a sample is "" here.
        """
        if _settings_registry().voice_binding(tts_model_type.id) is None:
            return []
        return [voice.file_name for voice in project.voice_references]

    @staticmethod
    def get_effective_voice_index(requested_index: int, num_voices: int) -> int:
        """Use voice 0 for an invalid per-line assignment, as inference does."""
        return requested_index if 0 <= requested_index < num_voices else 0

    @staticmethod
    def get_job_voice_indices(project: Project, phrase_indices: Iterable[int] | None = None) -> list[int]:
        """Voice slots to pre-flight, without changing the list or advancing rotation.

        Uses the job's effective line assignments. None denotes a literal
        prompt/chat with no line assignment (slot 0); an explicitly empty job
        uses no voices.
        """
        num_voices = len(ProjectVoiceUtil.get_voice_values(project, project.get_tts_model_type()))
        indices = None if phrase_indices is None else list(phrase_indices)
        if not num_voices or indices == []:
            return []
        if indices is None:
            return [0]
        phrase_groups = project.phrase_groups
        return sorted({
            ProjectVoiceUtil.get_effective_voice_index(phrase_groups[index].voice_index, num_voices)
            for index in indices
        })

    @staticmethod
    def get_voice_transcript_values(project: Project, tts_model_type: TtsModelType) -> list[str]:
        """Return paired transcripts, even if this model does not require them."""
        if _settings_registry().voice_binding(tts_model_type.id) is None:
            return []
        return [voice.transcript for voice in project.voice_references]

    @staticmethod
    def has_any_voice_sample(project: Project) -> bool:
        """Whether any voice has a sample. When none does, generation behaves as with no voices."""
        return any(voice.has_sample for voice in project.voice_references)

    @staticmethod
    def get_voice_sample_count(project: Project, tts_model_type: TtsModelType) -> int:
        """Number of voices that have a sample, for a model that supports voice samples."""
        return sum(1 for value in ProjectVoiceUtil.get_voice_values(project, tts_model_type) if value)

    @staticmethod
    def get_primary_voice_value(project: Project, tts_model_type: TtsModelType) -> str:
        values = ProjectVoiceUtil.get_voice_values(project, tts_model_type)
        return values[0] if values else ""

    @staticmethod
    def primary_voice_transcript(project: Project, tts_model_type: TtsModelType) -> str:
        values = ProjectVoiceUtil.get_voice_transcript_values(project, tts_model_type)
        return values[0] if values else ""

    @staticmethod
    def get_crop_range(voice: Voice) -> tuple[float, float] | None:
        """Return the voice's crop range in seconds, or None."""
        return voice.crop_range

    @staticmethod
    def get_cropped_voice_relative_path(voice: Voice) -> str:
        """Derive the voice-relative crop path from its stored basename."""
        if voice.crop is None:
            raise ValueError("Voice reference has no valid explicit trim file")
        return voice.crop.relative_path

    @staticmethod
    def resolve_crop_file_path(project: Project, crop_file_name: str) -> str:
        """Resolve a crop basename under voice/crops, without legacy guesses."""
        if normalize_crop_file_name(crop_file_name) is None:
            raise ValueError("Voice reference has no valid explicit trim file")
        return path_norm.join_project_relative(
            ProjectVoiceUtil.get_voice_dir_path(project),
            f"{VOICE_CROP_SUBDIR}/{crop_file_name}",
        )

    @staticmethod
    def resolve_cropped_voice_file_path(project: Project, voice: Voice) -> str:
        """Resolve the voice's crop file under voice/crops, without legacy guesses."""
        return path_norm.join_project_relative(
            ProjectVoiceUtil.get_voice_dir_path(project),
            ProjectVoiceUtil.get_cropped_voice_relative_path(voice),
        )

    @staticmethod
    def effective_voice_file_path(project: Project, voice: Voice) -> str:
        """Resolve the active crop when present, otherwise the original sample."""
        if voice.crop is not None:
            cropped_path = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, voice)
            if os.path.isfile(cropped_path):
                return cropped_path
        return ProjectVoiceUtil.resolve_voice_file_path(project, voice.file_name)

    @staticmethod
    def current_voice(
            project: Project,
            tts_model_type: TtsModelType,
            voice_selection_index: int,
    ) -> Voice | None:
        """The voice entry selected by `voice_selection_index` (wrapping), or
        None when the model has no voice samples or none are configured.
        Same selection as `effective_voice_reference()`."""
        if not ProjectVoiceUtil.get_voice_values(project, tts_model_type):
            return None
        voices = project.voice_references
        return voices[voice_selection_index % len(voices)]

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
        voice = ProjectVoiceUtil.current_voice(project, tts_model_type, voice_selection_index)
        if voice is None:
            return "", ""
        if voice.crop is not None:
            cropped_path = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, voice)
            if os.path.isfile(cropped_path):
                return voice.crop.relative_path, voice.crop.transcript
            # Cropped file missing (manually deleted, partial copy): fall back
            # to the original rather than blocking generation. Pre-flight
            # validation surfaces this as a problem to fix.
        return voice.file_name, voice.transcript

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
        tagged_suffix = f"_{tts_model_spec.file_tag}.flac"
        if file_name.endswith(tagged_suffix):
            value = file_name.removesuffix(tagged_suffix)
        else:
            value = os.path.splitext(file_name)[0] or file_name
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
            voice.crop.relative_path for voice in project.voice_references if voice.crop is not None
        }
        if exclude_owner is None:
            used.update(voice.file_name for voice in project.voice_references if voice.has_sample)
        emo_voice = project.get_model_setting("indextts2_local", "emo_voice")
        if not exclude_secondary and emo_voice:
            used.add(emo_voice)
        return used

    @staticmethod
    def _same_audio(path_a: str, path_b: str) -> bool:
        """Whether two sound files decode to identical 16-bit PCM at the same rate."""
        import numpy as np
        import soundfile
        try:
            data_a, sr_a = soundfile.read(path_a, dtype="int16", always_2d=True)
            data_b, sr_b = soundfile.read(path_b, dtype="int16", always_2d=True)
        except Exception:
            return False
        return sr_a == sr_b and np.array_equal(data_a, data_b)

    @staticmethod
    def _existing_case_variant(parent: Path, file_name: str) -> str:
        """`file_name` re-spelled to match an existing file in `parent` that
        differs from it only in case; unchanged when there is none (or an
        exact match exists)."""
        prefix, _, base_name = file_name.replace("\\", "/").rpartition("/")
        try:
            entries = sorted(os.listdir(parent))
        except OSError:
            return file_name
        if base_name in entries:
            return file_name
        key = base_name.casefold()
        for entry in entries:
            if entry.casefold() == key and os.path.isfile(parent / entry):
                return f"{prefix}/{entry}" if prefix else entry
        return file_name

    @staticmethod
    def _save_voice_sample_file(
            project: Project, sound: Sound, voice_file_stem: str, used: set[str],
    ) -> tuple[str, str]:
        """Write an imported voice sample into the voice subdir.

        `used` holds names owned by other references (compared case
        insensitively). When `<stem>.flac` clashes with one of them whose
        audio is identical to what is about to be saved, that existing file
        is shared rather than duplicated. Otherwise a clash is disambiguated
        with a numeric suffix. An unreferenced file on disk whose name
        differs only in case is overwritten under its existing spelling.
        Returns (file name, error string or "").
        """
        used_by_key = {name.casefold(): name for name in used}
        stem = voice_file_stem
        dest_file_name = f"{stem}.flac"
        voice_dir = Path(ProjectVoiceUtil.get_voice_dir_path(project))
        crop_dir = voice_dir / VOICE_CROP_SUBDIR
        if (voice_dir / dest_file_name).resolve().is_relative_to(crop_dir.resolve()):
            return "", "Voice samples cannot be imported into the managed trim directory"
        dest_parent = (voice_dir / dest_file_name).parent
        dest_parent.mkdir(parents=True, exist_ok=True)

        # Encode first, so the comparison is against exactly what would be saved.
        fd, temp_path = tempfile.mkstemp(suffix=".flac", dir=dest_parent)
        os.close(fd)
        try:
            err = SoundFileUtil.save_flac(sound, temp_path)
            if err:
                return "", err

            clashing_name = used_by_key.get(dest_file_name.casefold())
            if clashing_name is not None:
                clashing_path = ProjectVoiceUtil.resolve_voice_file_path(project, clashing_name)
                if os.path.isfile(clashing_path) and ProjectVoiceUtil._same_audio(temp_path, clashing_path):
                    return clashing_name, ""
                count = 2
                while dest_file_name.casefold() in used_by_key:
                    dest_file_name = f"{stem}_{count}.flac"
                    count += 1

            # An unreferenced file whose name differs only in case is
            # overwritten under its on-disk spelling (as an exact-name one
            # always was), so the stored name always matches the disk and
            # case-only variants never accumulate side by side.
            dest_file_name = ProjectVoiceUtil._existing_case_variant(dest_parent, dest_file_name)
            dest_path = voice_dir / dest_file_name
            if dest_path.resolve().is_relative_to(crop_dir.resolve()):
                return "", "Voice samples cannot be imported into the managed trim directory"
            try:
                os.replace(temp_path, dest_path)
            except OSError as e:
                return "", f"Error saving sound file {dest_path}: {e}"
            temp_path = ""
            return dest_file_name, ""
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

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
        stem_parts = path_norm.split_relative(voice_file_stem)
        if len(stem_parts) > 1 and stem_parts[0].casefold() == VOICE_CROP_SUBDIR:
            return "Voice samples cannot be imported into the managed trim directory"
        # Voice sample files are stored undecorated in the project's voice
        # subdir. Disambiguate the stem if a different storage owner already
        # references the same file name.
        secondary = tts_type.id == "indextts2_local" and is_secondary
        binding = None if secondary else _settings_registry().voice_binding(tts_type.id)
        exclude_owner = (binding.model_id, binding.name) if binding else None
        used = ProjectVoiceUtil.get_used_voice_file_names(project, exclude_owner, exclude_secondary=secondary)
        # A sole voice still at its defaults is a placeholder (a project always
        # has one voice); an appended sample takes its place.
        replaces_placeholder = (
            append and len(project.voice_references) == 1 and project.voice_references[0].is_default
        )
        if append and not replaces_placeholder:
            # When appending, a same-stem sample should not clobber an
            # existing entry; give it a distinct name (or share it, when
            # its audio is identical).
            used.update(value for value in ProjectVoiceUtil.get_voice_values(project, tts_type) if value)

        dest_file_name, err = ProjectVoiceUtil._save_voice_sample_file(project, sound, voice_file_stem, used)
        if err:
            return err

        if secondary:
            project.set_model_setting("indextts2_local", "emo_voice", dest_file_name)
        else:
            if binding is None:
                raise ValueError(f"Unsupported tts type {tts_type}")
            # A new voice: room reverb defaults to off.
            voice = Voice(file_name=dest_file_name, transcript=transcript)
            if append and not replaces_placeholder:
                project.voice_references = project.voice_references + [voice]
            else:
                project.voice_references = [voice]

        return project.save()

    @staticmethod
    def set_voice_sample_at_index_and_save(
            project: Project,
            index: int,
            sound: Sound,
            voice_file_stem: str,
            tts_type: TtsModelType,
    ) -> str:
        """Set (or replace) the sample of the voice at `index`.

        Writes the sample file and, in one save, points the voice at it,
        resets its trim (deleting the managed trim file) and clears its
        transcript; the caller fills the transcript afterward. Other voice
        properties are kept. The voice's own current file name is not
        treated as taken, so re-setting the same stem overwrites in place.
        Returns an error string or "" on success.
        """
        if index < 0 or index >= len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")
        if _settings_registry().voice_binding(tts_type.id) is None:
            raise ValueError(f"Unsupported tts type {tts_type}")
        stem_parts = path_norm.split_relative(voice_file_stem)
        if len(stem_parts) > 1 and stem_parts[0].casefold() == VOICE_CROP_SUBDIR:
            return "Voice samples cannot be imported into the managed trim directory"

        voice = project.voice_references[index]
        used = ProjectVoiceUtil.get_used_voice_file_names(project, exclude_owner=None)
        own_name = voice.file_name.casefold()
        emo_voice = project.get_model_setting("indextts2_local", "emo_voice") or ""
        shared_elsewhere = bool(own_name) and (emo_voice.casefold() == own_name or any(
            other.file_name.casefold() == own_name
            for other_index, other in enumerate(project.voice_references) if other_index != index
        ))
        if not shared_elsewhere:
            used = {name for name in used if name.casefold() != own_name}

        # Decided before the change: never delete a trim file that is also
        # another input.
        old_crop_path = ""
        if voice.crop is not None:
            candidate = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, voice)
            if not ProjectVoiceUtil._crop_path_is_shared(project, index, candidate):
                old_crop_path = candidate

        dest_file_name, err = ProjectVoiceUtil._save_voice_sample_file(project, sound, voice_file_stem, used)
        if err:
            return err

        previous_entries = project.voice_references
        entries = list(previous_entries)
        entries[index] = replace(voice, file_name=dest_file_name, transcript="", crop=None)
        project.voice_references = entries
        save_error = project.save()
        if save_error:
            project.voice_references = previous_entries
            return save_error
        if old_crop_path and os.path.lexists(old_crop_path):
            try:
                os.remove(old_crop_path)
            except OSError:
                # Leaves only an unreferenced managed trim file.
                pass
        return ""

    @staticmethod
    def clear_voice_sample_at_index_and_save(project: Project, index: int) -> str:
        """Clear only a voice's sample, transcript and trim, keeping its slot/settings.

        Keep the original sample on disk. Delete an unshared managed trim only
        after saving; on save failure restore the original voice list.
        """
        if not 0 <= index < len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")

        voice = project.voice_references[index]
        cropped_path = ""
        if voice.crop is not None:
            candidate = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, voice)
            if not ProjectVoiceUtil._crop_path_is_shared(project, index, candidate):
                cropped_path = candidate

        previous_entries = project.voice_references
        entries = list(previous_entries)
        entries[index] = replace(voice, file_name="", transcript="", crop=None)
        project.voice_references = entries
        save_error = project.save()
        if save_error:
            project.voice_references = previous_entries
            return save_error
        if cropped_path and os.path.lexists(cropped_path):
            try:
                os.remove(cropped_path)
            except OSError:
                # Leaves only an unreferenced managed trim file, as on Set.
                pass
        return ""

    @staticmethod
    def add_voice_and_save(project: Project) -> tuple[int, str]:
        """Append a default (sample-less) voice. Returns (its index, error string or "")."""
        previous_entries = project.voice_references
        project.voice_references = previous_entries + [Voice()]
        error = project.save()
        if error:
            project.voice_references = previous_entries
            return -1, error
        return len(project.voice_references) - 1, ""

    @staticmethod
    def is_voice_list_cleared(project: Project) -> bool:
        """Whether the voice list is just the one default voice."""
        voices = project.voice_references
        return len(voices) == 1 and voices[0].is_default

    @staticmethod
    def clear_voices_and_save(project: Project) -> str:
        """Reset the voice list to a single default voice. Returns the save
        error, in which case the previous list is restored, or ""."""
        previous = project.voice_references
        project.voice_references = ensure_voice_slot([])
        error = project.save()
        if error:
            project.voice_references = previous
        return error

    @staticmethod
    def remove_voice_at_index_and_save(project: Project, tts_type: TtsModelType, index: int) -> str:
        if _settings_registry().voice_binding(tts_type.id) is None:
            raise ValueError(f"Unsupported tts_type: {tts_type}")

        voices = ProjectVoiceUtil.get_voice_values(project, tts_type)
        if index < 0 or index >= len(voices):
            raise IndexError(f"Voice sample index out of range: {index}")

        removed = voices[index]
        # Removing the last voice leaves a fresh default voice.
        project.voice_references = ensure_voice_slot(
            project.voice_references[:index] + project.voice_references[index + 1:]
        )

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
            project.voice_references = ensure_voice_slot([])

        project.save()

    @staticmethod
    def set_voice_transcript_at_index_and_save(project: Project, index: int, transcript: str) -> str:
        """Edit exactly one shared entry without capability-dependent storage."""
        if index < 0 or index >= len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")
        entries = list(project.voice_references)
        entries[index] = replace(entries[index], transcript=transcript)
        project.voice_references = entries
        return project.save()

    @staticmethod
    def set_voice_instructions_at_index_and_save(
            project: Project, index: int, model_id: str, value: str,
    ) -> str:
        """Set one voice's model-specific instructions; empty removes only that key."""
        if not 0 <= index < len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")
        voice = project.voice_references[index]
        instructions = dict(voice.instructions)
        value = value.strip()
        if value:
            instructions[model_id] = value
        else:
            instructions.pop(model_id, None)
        if instructions == voice.instructions:
            return ""
        previous = project.voice_references
        entries = list(previous)
        entries[index] = replace(voice, instructions=instructions)
        project.voice_references = entries
        error = project.save()
        if error:
            project.voice_references = previous
        return error

    @staticmethod
    def set_voice_preset_at_index_and_save(
            project: Project, index: int, model_id: str, value: str,
    ) -> str:
        """Set one voice's model-specific preset voice id; empty removes only that key."""
        if not 0 <= index < len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")
        voice = project.voice_references[index]
        presets = dict(voice.presets)
        value = value.strip()
        if value:
            presets[model_id] = value
        else:
            presets.pop(model_id, None)
        if presets == voice.presets:
            return ""
        previous = project.voice_references
        entries = list(previous)
        entries[index] = replace(voice, presets=presets)
        project.voice_references = entries
        error = project.save()
        if error:
            project.voice_references = previous
        return error

    @staticmethod
    def active_voice_preset(project: Project, voice: Voice | None) -> str:
        """The voice's preset for the project's current model when the model
        uses presets under the current settings, else "". An active preset
        takes precedence over the voice's sample."""
        if voice is None:
            return ""
        model_id = project.get_tts_model_type().id
        preset = voice.preset_for(model_id)
        if not preset:
            return ""
        from tts_audiobook_tool.tts import Tts
        return preset if Tts.get_model_support(project).can_voice_presets(project) else ""

    @staticmethod
    def make_instructions_preview(instructions: str) -> str:
        """Eg '"A calm, low male voice…"': whitespace-collapsed, ellipsized, quoted."""
        return f"\"{ellipsize(' '.join(instructions.split()), INSTRUCTIONS_PREVIEW_LENGTH)}\""

    @staticmethod
    def resolve_voice_source(project: Project, voice: Voice) -> VoiceSource:
        """What the voice uses to generate under the project's current model
        and settings. Cheap and side-effect free (no file reads), so it is safe
        on every menu redraw; file existence and transcripts are checked by the
        voice pre-flight (`VoiceMenuShared.validate_voices`).

        Precedence: an active preset, then the sample (when the model can clone
        under the current settings), then the instructions (when the model uses
        them and does not require a sample). A model that requires a preset
        (eg Qwen3-TTS CustomVoice) accepts nothing else.

        The catalog's `requires_voice` is model-wide; a sample is required only
        while the model can clone under the current settings (eg not Qwen3-TTS
        VoiceDesign, whose voices are defined by their instructions).
        """
        from tts_audiobook_tool.tts import Tts

        preset = ProjectVoiceUtil.active_voice_preset(project, voice)
        if preset:
            return VoiceSource("preset", preset)
        model_type = project.get_tts_model_type()
        support = Tts.get_model_support(project)
        requires_preset = support.can_voice_presets(project) and support.requires_voice_preset(project)
        can_clone = support.can_voice_clone(project)
        requires_sample = model_type.value.requires_voice and can_clone
        if not requires_preset:
            if (voice.has_sample and _settings_registry().voice_binding(model_type.id) is not None
                    and can_clone):
                label = ProjectVoiceUtil.make_voice_sample_display_label(project, voice.file_name, model_type.value)
                return VoiceSource("sample", label)
            instructions = voice.instructions.get(model_type.id, "")
            if instructions.strip() and not requires_sample and support.can_instructions(project):
                return VoiceSource("instructions", ProjectVoiceUtil.make_instructions_preview(instructions))
        if requires_preset or requires_sample:
            return VoiceSource("missing")
        return VoiceSource("builtin")

    @staticmethod
    def make_no_voice_value_text(project: Project) -> str:
        """What a voice with no value lacks under the current model and settings
        (non-blocking: the model falls back to its built-in voice): "no sample set",
        or "no instructions set" when the model can't clone but uses instructions."""
        from tts_audiobook_tool.tts import Tts

        support = Tts.get_model_support(project)
        if support.can_voice_clone(project) or not support.can_instructions(project):
            return "no sample set"
        return "no instructions set"

    @staticmethod
    def has_any_active_voice_preset(project: Project) -> bool:
        return any(ProjectVoiceUtil.active_voice_preset(project, voice) for voice in project.voice_references)

    @staticmethod
    def set_voice_room_reverb_at_index_and_save(project: Project, index: int, room_reverb: RoomReverb) -> str:
        """Set one entry's room reverb preset."""
        if index < 0 or index >= len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")
        entries = list(project.voice_references)
        entries[index] = replace(entries[index], room_reverb=room_reverb)
        project.voice_references = entries
        return project.save()

    @staticmethod
    def is_voice_name_taken(project: Project, index: int, name: str) -> bool:
        """Whether another voice already has this name (case-insensitive, ignoring
        surrounding whitespace). An empty name is never taken."""
        key = name.strip().casefold()
        return bool(key) and any(
            i != index and voice.name.strip().casefold() == key
            for i, voice in enumerate(project.voice_references)
        )

    @staticmethod
    def get_voice_name_error(project: Project, index: int, name: str) -> str:
        """Why `name` can't be voice `index`'s name, or "". A name is one word
        (no whitespace), at most `VOICE_NAME_MAX_LENGTH` characters, and unique
        among the project's voices. Empty is valid (unset)."""
        name = name.strip()
        if any(char.isspace() for char in name):
            return VOICE_NAME_ONE_WORD_ERROR
        if len(name) > VOICE_NAME_MAX_LENGTH:
            return VOICE_NAME_TOO_LONG_ERROR
        if ProjectVoiceUtil.is_voice_name_taken(project, index, name):
            return VOICE_NAME_UNIQUE_ERROR
        return ""

    @staticmethod
    def set_voice_name_at_index_and_save(project: Project, index: int, name: str) -> str:
        """Set one entry's user-facing name; empty clears it. Returns the validation
        error (see `get_voice_name_error`) without saving, or the save error, in which
        case the previous list is restored."""
        if not 0 <= index < len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")
        voice = project.voice_references[index]
        name = name.strip()
        if name == voice.name:
            return ""
        error = ProjectVoiceUtil.get_voice_name_error(project, index, name)
        if error:
            return error
        previous = project.voice_references
        entries = list(previous)
        entries[index] = replace(voice, name=name)
        project.voice_references = entries
        error = project.save()
        if error:
            project.voice_references = previous
        return error

    @staticmethod
    def set_voice_pan_at_index_and_save(project: Project, index: int, pan: float) -> str:
        """Set one entry's stereo position (-1 full left to 1 full right)."""
        if index < 0 or index >= len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")
        entries = list(project.voice_references)
        entries[index] = replace(entries[index], pan=pan)
        project.voice_references = entries
        return project.save()

    @staticmethod
    def set_voice_crop_transcript_at_index_and_save(project: Project, index: int, transcript: str) -> str:
        """Edit one entry's crop transcript (the cropped span's transcript)."""
        if index < 0 or index >= len(project.voice_references):
            raise IndexError(f"Voice sample index out of range: {index}")
        voice = project.voice_references[index]
        if voice.crop is None:
            # Nothing to annotate: a crop transcript belongs to an active crop.
            return ""
        entries = list(project.voice_references)
        entries[index] = replace(voice, crop=replace(voice.crop, transcript=transcript))
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
        original_names = {voice.file_name for voice in project.voice_references if voice.has_sample}
        secondary = project.get_model_setting("indextts2_local", "emo_voice")
        if secondary:
            original_names.add(secondary)
        for name in original_names:
            if normalized(ProjectVoiceUtil.resolve_voice_file_path(project, name)) == target:
                return True
        for other_index, voice in enumerate(project.voice_references):
            if other_index != index and voice.crop is not None:
                if normalized(ProjectVoiceUtil.resolve_cropped_voice_file_path(project, voice)) == target:
                    return True
        return False

    @staticmethod
    def _allocate_crop_file(project: Project, index: int) -> tuple[str, str]:
        """Reserve a fresh unique path; unrelated existing files are never reused."""
        crop_dir = Path(ProjectVoiceUtil.get_voice_dir_path(project)) / VOICE_CROP_SUBDIR
        crop_dir.mkdir(parents=True, exist_ok=True)
        while True:
            crop_file_name = f"{uuid.uuid4().hex}.flac"
            crop_path = ProjectVoiceUtil.resolve_crop_file_path(project, crop_file_name)
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

        file_name = project.voice_references[index].file_name
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

        voice = project.voice_references[index]
        crop_file_name = voice.crop.file_name if voice.crop is not None else None
        cropped_path = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, voice) if crop_file_name else ""
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
        entries[index] = replace(entries[index], crop=VoiceCrop(
            file_name=crop_file_name, start=start_s, end=end_s, transcript=crop_transcript,
        ))
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

        voice = project.voice_references[index]
        cropped_path = ""
        if voice.crop is not None:
            candidate = ProjectVoiceUtil.resolve_cropped_voice_file_path(project, voice)
            if not ProjectVoiceUtil._crop_path_is_shared(project, index, candidate):
                cropped_path = candidate

        previous_entries = project.voice_references
        entries = list(previous_entries)
        entries[index] = replace(voice, crop=None)
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
