from __future__ import annotations

import json
import os

from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.roster import RosterFile
from tts_audiobook_tool.util import make_error_string


# Speaker assignment artifacts live in this subdirectory of the project directory.
SPEAKER_ASSIGN_DIR_NAME = "speaker_assign"
ROSTER_FILE_NAME = "roster.json"
ATTRIBUTIONS_FILE_NAME = "attributions.json"


class SpeakerAssignFileUtil:
    """
    Locations and persistence of a project's speaker assignment artifacts
    (`[project]/speaker_assign/`).
    """

    @staticmethod
    def dir_path(project_dir: str) -> str:
        return os.path.join(project_dir, SPEAKER_ASSIGN_DIR_NAME)

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
    def _make_dir(project_dir: str) -> str:
        try:
            os.makedirs(SpeakerAssignFileUtil.dir_path(project_dir), exist_ok=True)
        except OSError as e:
            return f"Error creating speaker assignment directory: {make_error_string(e)}"
        return ""
