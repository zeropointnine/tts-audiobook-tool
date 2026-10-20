"""Exact server selections survive persistence and settings transfers unchanged."""
import json

import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil


@pytest.mark.parametrize("server_id", ["", "entry-server/custom Model:V2", "future-unadvertised-id"])
def test_server_model_id_round_trips_save_export_snapshot_and_copy(tmp_path, server_id):
    # Keep exact opaque IDs, even when no installed catalog/server recognizes them.
    project = Project(dir_path=str(tmp_path), tts_model_type="future_catalog_id",
                      tts_server_model_id=server_id)
    assert project.save() == ""
    payload = json.loads((tmp_path / "project.json").read_text(encoding="utf-8"))
    assert payload["tts_server_model_id"] == server_id
    loaded = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)
    assert isinstance(loaded, Project)
    assert loaded.tts_server_model_id == server_id
    assert loaded.tts_model_type == "future_catalog_id"

    snapshot = ProjectSerializationUtil.to_snapshot_dict(project)
    assert snapshot["tts_server_model_id"] == server_id
    imported = ProjectTransferUtil.validate_abr_snapshot(snapshot)
    assert imported.tts_server_model_id == server_id
    restored = ProjectTransferUtil.make_project_from_snapshot("", snapshot)
    assert restored.tts_server_model_id == server_id
    destination = Project(tts_server_model_id="previous-selection")
    ProjectTransferUtil.apply_project_settings(destination, project)
    assert destination.tts_server_model_id == server_id
    assert "tts_server_model_id" not in ProjectTransferUtil.get_missing_project_settings_transfer_fields(Project)


def test_legacy_project_without_server_model_id_defaults_to_empty():
    assert Project.model_validate({"tts_model_type": "future_catalog_id"}).tts_server_model_id == ""


@pytest.mark.parametrize("invalid", [None, 7, [], {}])
def test_server_model_id_requires_a_string(invalid):
    with pytest.raises(ValueError):
        Project.model_validate({"tts_server_model_id": invalid})
