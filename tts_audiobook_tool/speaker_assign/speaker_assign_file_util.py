from __future__ import annotations

import json
import os
import shutil

from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.roster import RosterFile
from tts_audiobook_tool.speaker_assign.voice_map import VoiceMapFile
from tts_audiobook_tool.util import make_error_string


# Speaker assignment artifacts live in this subdirectory of the project directory.
SPEAKER_ASSIGN_DIR_NAME = "speaker_assign"
ROSTER_FILE_NAME = "roster.json"
ATTRIBUTIONS_FILE_NAME = "attributions.json"
VOICES_FILE_NAME = "voices.json"


class SpeakerAssignFileUtil:
    """
    Locations and persistence of a project's speaker assignment artifacts
    (`[project]/speaker_assign/`).
    """

    @staticmethod
    def dir_path(project_dir: str) -> str:
        return os.path.join(project_dir, SPEAKER_ASSIGN_DIR_NAME)

    @staticmethod
    def has_any_data(project_dir: str) -> bool:
        """True if the project has any saved speaker assignment file."""
        return (
            SpeakerAssignFileUtil.has_roster(project_dir)
            or os.path.isfile(SpeakerAssignFileUtil.attributions_path(project_dir))
            or os.path.isfile(SpeakerAssignFileUtil.voices_path(project_dir))
        )

    @staticmethod
    def roster_path(project_dir: str) -> str:
        return os.path.join(SpeakerAssignFileUtil.dir_path(project_dir), ROSTER_FILE_NAME)

    @staticmethod
    def has_roster(project_dir: str) -> bool:
        return os.path.isfile(SpeakerAssignFileUtil.roster_path(project_dir))

    @staticmethod
    def load_roster_file(project_dir: str) -> RosterFile | str:
        """The saved roster, or an error string (missing, unreadable or invalid file)."""
        path = SpeakerAssignFileUtil.roster_path(project_dir)
        if not os.path.isfile(path):
            return f"No speaker roster: {path}"
        try:
            with open(path, "r", encoding="utf-8") as file:
                value = json.load(file)
            return RosterFile.from_dict(value)
        except Exception as e:
            return f"Error loading speaker roster {path}: {make_error_string(e)}"

    @staticmethod
    def save_roster_file(project_dir: str, roster_file: RosterFile) -> str:
        """Atomically write the roster file (creating the directory). Returns an
        error string on failure, otherwise ""."""
        error = SpeakerAssignFileUtil._make_dir(project_dir)
        if error:
            return error
        return JsonSaveUtil.save(
            JsonArtifactType.SPEAKER_ROSTER,
            SpeakerAssignFileUtil.roster_path(project_dir),
            roster_file.to_dict,
        )

    @staticmethod
    def attributions_path(project_dir: str) -> str:
        return os.path.join(SpeakerAssignFileUtil.dir_path(project_dir), ATTRIBUTIONS_FILE_NAME)

    @staticmethod
    def load_attributions_file(project_dir: str) -> AttributionsFile | str:
        """The saved attributions (an empty AttributionsFile if there are none
        yet), or an error string (unreadable or invalid file)."""
        path = SpeakerAssignFileUtil.attributions_path(project_dir)
        if not os.path.isfile(path):
            return AttributionsFile()
        try:
            with open(path, "r", encoding="utf-8") as file:
                value = json.load(file)
            return AttributionsFile.from_dict(value)
        except Exception as e:
            return f"Error loading speaker attributions {path}: {make_error_string(e)}"

    @staticmethod
    def save_attributions_file(project_dir: str, attributions: AttributionsFile) -> str:
        """Atomically write the attributions file (creating the directory).
        Returns an error string on failure, otherwise ""."""
        error = SpeakerAssignFileUtil._make_dir(project_dir)
        if error:
            return error
        return JsonSaveUtil.save(
            JsonArtifactType.SPEAKER_ATTRIBUTIONS,
            SpeakerAssignFileUtil.attributions_path(project_dir),
            attributions.to_dict,
        )

    @staticmethod
    def delete_attributions_file(project_dir: str) -> str:
        """Remove the saved attributions, if any (eg when the roster they were
        made against is replaced). Returns an error string on failure, otherwise ""."""
        try:
            os.remove(SpeakerAssignFileUtil.attributions_path(project_dir))
        except FileNotFoundError:
            pass
        except OSError as e:
            return f"Error deleting speaker attributions: {make_error_string(e)}"
        return ""

    @staticmethod
    def voices_path(project_dir: str) -> str:
        return os.path.join(SpeakerAssignFileUtil.dir_path(project_dir), VOICES_FILE_NAME)

    @staticmethod
    def load_voices_file(project_dir: str) -> VoiceMapFile | str:
        """The saved character voices (an empty VoiceMapFile if there are none
        yet), or an error string (unreadable or invalid file)."""
        path = SpeakerAssignFileUtil.voices_path(project_dir)
        if not os.path.isfile(path):
            return VoiceMapFile()
        try:
            with open(path, "r", encoding="utf-8") as file:
                value = json.load(file)
            return VoiceMapFile.from_dict(value)
        except Exception as e:
            return f"Error loading character voices {path}: {make_error_string(e)}"

    @staticmethod
    def save_voices_file(project_dir: str, voices: VoiceMapFile) -> str:
        """Atomically write the character voices file (creating the directory).
        Returns an error string on failure, otherwise ""."""
        error = SpeakerAssignFileUtil._make_dir(project_dir)
        if error:
            return error
        return JsonSaveUtil.save(
            JsonArtifactType.SPEAKER_VOICES,
            SpeakerAssignFileUtil.voices_path(project_dir),
            voices.to_dict,
        )

    @staticmethod
    def copy_files(source_project_dir: str, dest_project_dir: str) -> list[str]:
        """
        Copy the source project's speaker assignment files (those that exist)
        into the destination project, eg when making a new project from the
        current project's settings. The files reference characters and
        utterance ranges rather than project file paths, so they stay valid
        alongside an identical copy of the project text.

        Returns error strings for files that could not be copied.
        """
        errors: list[str] = []
        for file_name in (ROSTER_FILE_NAME, ATTRIBUTIONS_FILE_NAME, VOICES_FILE_NAME):
            source_path = os.path.join(SpeakerAssignFileUtil.dir_path(source_project_dir), file_name)
            if not os.path.isfile(source_path):
                continue
            error = SpeakerAssignFileUtil._make_dir(dest_project_dir)
            if error:
                return [error]
            dest_path = os.path.join(SpeakerAssignFileUtil.dir_path(dest_project_dir), file_name)
            try:
                shutil.copy(source_path, dest_path)
            except OSError as e:
                errors.append(f"Error copying {source_path}: {make_error_string(e)}")
        return errors

    @staticmethod
    def _make_dir(project_dir: str) -> str:
        try:
            os.makedirs(SpeakerAssignFileUtil.dir_path(project_dir), exist_ok=True)
        except OSError as e:
            return f"Error creating speaker assignment directory: {make_error_string(e)}"
        return ""
