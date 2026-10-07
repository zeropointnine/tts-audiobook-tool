"""Explicitly owned voice crops: schema, resolution, and safe lifecycle."""
import json
from pathlib import Path
import uuid

import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.project_support.voice_reference_migration import (
    normalize_voice_references,
)
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

GLM = TtsModelType.require_by_id("glm_local")


def make_project(voice_dir: Path, *entries: dict) -> Project:
    voice_dir.mkdir(parents=True, exist_ok=True)
    project = Project.model_validate({"voice_references": list(entries)})
    project.dir_path = str(voice_dir.parent)
    return project


def write_sample(voice_dir: Path, name: str, seconds: float = 1.0, sr: int = 24000) -> None:
    from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
    (voice_dir / name).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = rng.uniform(0.01, 0.02, int(seconds * sr)).astype(np.float32)
    assert SoundFileUtil.save_flac(Sound(data, sr), str(voice_dir / name)) == ""


class TestNormalizeVoiceReferencesCropFields:

    def test_crop_fields_preserved_and_canonicalized(self):
        result = normalize_voice_references([
            {"file_name": "a.flac", "transcript": "t", "crop_file_name": "a.flac", "crop_start": 1.5, "crop_end": 9, "crop_transcript": "cut"},
        ])
        assert result == [{
            "file_name": "a.flac", "transcript": "t",
            "crop_file_name": "a.flac", "crop_start": "1.5", "crop_end": "9.0", "crop_transcript": "cut",
        }]

    def test_numeric_strings_accepted(self):
        result = normalize_voice_references([
            {"file_name": "a.flac", "crop_file_name": "a.flac", "crop_start": "0.0", "crop_end": "2.5", "crop_transcript": ""},
        ])
        assert result[0]["crop_start"] == "0.0"
        assert result[0]["crop_end"] == "2.5"

    @pytest.mark.parametrize("crop", [
        {"crop_file_name": "a.flac", "crop_start": 5.0, "crop_end": 5.0},
        {"crop_file_name": "a.flac", "crop_start": 5.0, "crop_end": 1.0},
        {"crop_file_name": "a.flac", "crop_start": -1.0, "crop_end": 2.0},
        {"crop_file_name": "a.flac", "crop_start": "abc", "crop_end": 2.0},
        {"crop_file_name": "a.flac", "crop_start": 1.0},
        {"crop_transcript": "orphan"},
        {"crop_transcript": 7, "crop_file_name": "a.flac", "crop_start": 0.0, "crop_end": 1.0},
        {"crop_file_name": "a.flac", "crop_start": float("inf"), "crop_end": 2.0},
    ])
    def test_invalid_crop_dropped_entry_kept(self, crop):
        result = normalize_voice_references([{"file_name": "a.flac", "transcript": "t", **crop}])
        assert result == [{"file_name": "a.flac", "transcript": "t"}]

    def test_project_load_and_resave_round_trip(self, tmp_path):
        project = Project.model_validate({
            "dir_path": str(tmp_path),
            "voice_references": [
                {"file_name": "a.flac", "transcript": "t", "crop_file_name": "a.flac", "crop_start": "1.0", "crop_end": "2.0", "crop_transcript": "cut"},
            ],
        })
        assert project.voice_references[0]["crop_transcript"] == "cut"
        assert project.save() == ""
        text = (tmp_path / "project.json").read_text()
        assert '"crop_start"' in text and '"crop_transcript"' in text


class TestSetModelSettingCropPreservation:

    def test_transcript_replacement_preserves_crop(self):
        project = Project.model_validate({
            "voice_references": [
                {"file_name": "a.flac", "transcript": "old", "crop_file_name": "a.flac", "crop_start": "1.0", "crop_end": "2.0", "crop_transcript": "cut"},
            ],
        })
        project.set_model_setting("glm_local", "transcript", ["new"])
        entry = project.voice_references[0]
        assert entry["transcript"] == "new"
        assert entry["crop_start"] == "1.0"
        assert entry["crop_end"] == "2.0"
        assert entry["crop_transcript"] == "cut"
        assert entry["crop_file_name"] == "a.flac"

    def test_file_name_replacement_drops_crop(self):
        project = Project.model_validate({
            "voice_references": [
                {"file_name": "a.flac", "transcript": "t", "crop_file_name": "a.flac", "crop_start": "1.0", "crop_end": "2.0", "crop_transcript": "cut"},
            ],
        })
        project.set_model_setting("glm_local", "file_name", ["b.flac"])
        assert project.voice_references == [{"file_name": "b.flac", "transcript": "t"}]


class TestEffectiveVoiceReference:

    @pytest.mark.parametrize("original_name", ["a.flac", "nested/original.wav", "crops/original.flac"])
    def test_crop_basename_derives_path_independently_of_original_location(self, tmp_path, original_name):
        project = make_project(tmp_path / "voice", {"file_name": original_name, "transcript": "full"})
        entry = {"file_name": original_name, "crop_file_name": "a.flac"}
        assert ProjectVoiceUtil.get_cropped_voice_relative_path(entry) == "crops/a.flac"
        assert ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry) == str(tmp_path / "voice/crops/a.flac")
        assert entry["crop_file_name"] == "a.flac"
        with pytest.raises(ValueError, match="explicit trim file"):
            ProjectVoiceUtil.get_cropped_voice_relative_path({"file_name": original_name})
        with pytest.raises(ValueError, match="explicit trim file"):
            ProjectVoiceUtil.resolve_cropped_voice_file_path(project, {"file_name": original_name})

    def test_no_crop_returns_original_pair(self, tmp_path):
        project = make_project(tmp_path / "voice", {"file_name": "a.flac", "transcript": "full"})
        assert ProjectVoiceUtil.effective_voice_reference(project, GLM, 0) == ("a.flac", "full")

    def test_crop_with_file_switches_in_lockstep(self, tmp_path):
        write_sample(tmp_path / "voice", "a.flac")
        write_sample(tmp_path / "voice", "crops/a.flac")
        project = make_project(
            tmp_path / "voice",
            {"file_name": "a.flac", "transcript": "full", "crop_file_name": "a.flac", "crop_start": "0.0", "crop_end": "0.5", "crop_transcript": "cut"},
        )
        pair = ProjectVoiceUtil.effective_voice_reference(project, GLM, 0)
        assert pair == ("crops/a.flac", "cut")
        # All generation-facing accessors agree (lockstep invariant).
        assert ProjectVoiceUtil.current_voice_value(project, GLM, 0) == "crops/a.flac"
        assert ProjectVoiceUtil.current_voice_reference_pair(project, GLM, 0) == pair
        assert ProjectVoiceUtil.voice_reference_pairs(project, GLM) == [pair]
        # Display accessors keep pointing at the stored entry.
        assert ProjectVoiceUtil.get_voice_values(project, GLM) == ["a.flac"]

    def test_crop_with_missing_file_falls_back_to_original(self, tmp_path):
        project = make_project(
            tmp_path / "voice",
            {"file_name": "a.flac", "transcript": "full", "crop_file_name": "a.flac", "crop_start": "0.0", "crop_end": "0.5", "crop_transcript": "cut"},
        )
        assert ProjectVoiceUtil.effective_voice_reference(project, GLM, 0) == ("a.flac", "full")

    def test_index_wraps_like_the_old_accessors(self, tmp_path):
        project = make_project(
            tmp_path / "voice",
            {"file_name": "a.flac", "transcript": "A"},
            {"file_name": "b.flac", "transcript": "B"},
        )
        assert ProjectVoiceUtil.current_voice_reference_pair(project, GLM, 3) == ("b.flac", "B")


class TestApplyAndDiscard:

    def test_apply_then_discard_round_trip(self, tmp_path):
        voice_dir = tmp_path / "voice"
        write_sample(voice_dir, "a.flac", seconds=6.0)
        project = make_project(voice_dir, {"file_name": "a.flac", "transcript": "full"})

        assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 1.0, 4.5, "cut text") == ""
        entry = project.voice_references[0]
        cropped = Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry))
        assert cropped.is_file()
        assert cropped.parent == voice_dir / "crops"
        assert len(cropped.stem) == 32
        assert entry["crop_file_name"] == cropped.name
        assert "/" not in entry["crop_file_name"] and "\\" not in entry["crop_file_name"]
        saved_entry = json.loads((tmp_path / "project.json").read_text())["voice_references"][0]
        assert saved_entry["crop_file_name"] == cropped.name
        assert "/" not in saved_entry["crop_file_name"] and "\\" not in saved_entry["crop_file_name"]
        assert entry["crop_start"] == "1.0"
        assert entry["crop_end"] == "4.5"
        assert entry["crop_transcript"] == "cut text"
        assert entry["transcript"] == "full"
        pair = ProjectVoiceUtil.effective_voice_reference(project, GLM, 0)
        assert pair == (f"crops/{entry['crop_file_name']}", "cut text")

        # The cropped file is one second of the original two.
        from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
        sound = SoundFileUtil.load(str(cropped))
        assert not isinstance(sound, str)
        assert sound.duration == pytest.approx(3.5, abs=0.05)

        assert ProjectVoiceUtil.discard_voice_crop_and_save(project, 0) == ""
        assert not cropped.exists()
        assert project.voice_references == [{"file_name": "a.flac", "transcript": "full"}]
        assert ProjectVoiceUtil.effective_voice_reference(project, GLM, 0) == ("a.flac", "full")

    def test_apply_accepts_two_second_crop(self, tmp_path):
        voice_dir = tmp_path / "voice"
        write_sample(voice_dir, "a.flac", seconds=6.0)
        project = make_project(voice_dir, {"file_name": "a.flac", "transcript": "full"})
        assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 1.0, 3.0, "cut") == ""
        assert project.voice_references[0]["crop_start"] == "1.0"
        assert project.voice_references[0]["crop_end"] == "3.0"
        assert Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(project, project.voice_references[0])).is_file()

    def test_discard_tolerates_missing_file(self, tmp_path):
        project = make_project(
            tmp_path / "voice",
            {"file_name": "a.flac", "transcript": "t", "crop_file_name": "a.flac", "crop_start": "0.0", "crop_end": "1.0", "crop_transcript": "c"},
        )
        assert ProjectVoiceUtil.discard_voice_crop_and_save(project, 0) == ""
        assert project.voice_references == [{"file_name": "a.flac", "transcript": "t"}]

    def test_apply_rejects_bad_range(self, tmp_path):
        voice_dir = tmp_path / "voice"
        write_sample(voice_dir, "a.flac", seconds=6.0)
        project = make_project(voice_dir, {"file_name": "a.flac", "transcript": "t"})
        assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 2.0, 1.0, "x").startswith("Invalid trim range")
        assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 0.0, 99.0, "x").startswith("Trim end")
        assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 0.0, 1.95, "x").startswith("Trim span must be at least 2s")
        assert project.voice_references == [{"file_name": "a.flac", "transcript": "t"}]


@pytest.mark.parametrize("crop_file_name", [
    None, 123, "", ".", "..", "../a.flac", "..\\a.flac", "crops/../a.flac",
    "/a.flac", "C:\\book\\voice\\crops\\a.flac", "C:a.flac", "crops/nested/a.flac",
    "a.wav", ".flac", "a.flac:stream",
])
def test_crop_file_name_must_be_explicit_flac_basename(tmp_path, crop_file_name):
    entry = {"file_name": "a.flac", "transcript": "full", "crop_start": "0", "crop_end": "3"}
    if crop_file_name is not None:
        entry["crop_file_name"] = crop_file_name
    assert normalize_voice_references([entry]) == [{"file_name": "a.flac", "transcript": "full"}]
    assert ProjectVoiceUtil.get_crop_range(entry) is None
    with pytest.raises(ValueError):
        ProjectVoiceUtil.get_cropped_voice_relative_path(entry)
    with pytest.raises(ValueError):
        ProjectVoiceUtil.resolve_cropped_voice_file_path(make_project(tmp_path / "voice"), entry)


@pytest.mark.parametrize("crop_file_name", ["crops/a.flac", "crops\\a.flac", "nested/a.flac", "nested\\a.flac"])
def test_crop_file_name_rejects_separators_instead_of_normalizing(tmp_path, crop_file_name):
    entry = {"file_name": "a.flac", "crop_file_name": crop_file_name, "crop_start": "0", "crop_end": "3"}
    assert normalize_voice_references([entry]) == [{"file_name": "a.flac", "transcript": ""}]
    assert ProjectVoiceUtil.get_crop_range(entry) is None
    with pytest.raises(ValueError):
        ProjectVoiceUtil.get_cropped_voice_relative_path(entry)
    with pytest.raises(ValueError):
        ProjectVoiceUtil.resolve_cropped_voice_file_path(make_project(tmp_path / "voice"), entry)


@pytest.mark.parametrize("crop_file_name", ["a.flac", "a_crop.flac", "owned.flac", f"{uuid.UUID(int=1).hex}.flac"])
def test_crop_basename_may_differ_from_original(crop_file_name):
    entry = {"file_name": "nested/original.wav", "transcript": "full", "crop_file_name": crop_file_name,
             "crop_start": "0", "crop_end": "3", "crop_transcript": "cut"}
    normalized = normalize_voice_references([entry])[0]
    assert normalized["crop_file_name"] == crop_file_name
    assert normalized["file_name"] == "nested/original.wav"
    assert ProjectVoiceUtil.get_crop_range(normalized) == (0.0, 3.0)
    assert ProjectVoiceUtil.get_cropped_voice_relative_path(normalized) == f"crops/{crop_file_name}"


def test_old_implicit_crop_is_neither_used_nor_deleted(tmp_path):
    voice_dir = tmp_path / "voice"
    write_sample(voice_dir, "a.flac", 6)
    write_sample(voice_dir, "a_crop.flac", 6)
    old_bytes = (voice_dir / "a_crop.flac").read_bytes()
    project = make_project(voice_dir, {
        "file_name": "a.flac", "transcript": "full",
        "crop_start": "0", "crop_end": "3", "crop_transcript": "old trim",
    })
    assert project.voice_references == [{"file_name": "a.flac", "transcript": "full"}]
    assert ProjectVoiceUtil.effective_voice_reference(project, GLM, 0) == ("a.flac", "full")
    assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 0, 3, "new trim") == ""
    assert ProjectVoiceUtil.discard_voice_crop_and_save(project, 0) == ""
    assert (voice_dir / "a_crop.flac").read_bytes() == old_bytes


def test_missing_explicit_crop_does_not_resolve_unrelated_basename(tmp_path):
    voice_dir = tmp_path / "voice"
    write_sample(voice_dir, "a.flac", 6)
    write_sample(voice_dir, "crop.flac", 6)
    write_sample(tmp_path / "crops", "crop.flac", 6)
    project = make_project(voice_dir, {
        "file_name": "a.flac", "transcript": "full", "crop_file_name": "crop.flac",
        "crop_start": "0", "crop_end": "3", "crop_transcript": "cut",
    })
    assert ProjectVoiceUtil.resolve_voice_file_path(project, "crops/crop.flac") == str(voice_dir / "crops/crop.flac")
    assert ProjectVoiceUtil.effective_voice_reference(project, GLM, 0) == ("a.flac", "full")
    assert ProjectVoiceUtil.effective_voice_file_path(project, project.voice_references[0]) == str(voice_dir / "a.flac")


@pytest.mark.parametrize("import_first", [True, False])
@pytest.mark.parametrize("secondary", [True, False])
def test_crop_does_not_collide_with_imported_crop_named_sample(tmp_path, import_first, secondary):
    from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
    voice_dir = tmp_path / "voice"
    write_sample(voice_dir, "a.flac", 6)
    original_bytes = (voice_dir / "a.flac").read_bytes()
    project = make_project(voice_dir, {"file_name": "a.flac", "transcript": "full"})
    sound = SoundFileUtil.load(str(voice_dir / "a.flac"))
    assert isinstance(sound, Sound)

    def import_sample():
        model = TtsModelType.require_by_id("indextts2_local") if secondary else GLM
        assert ProjectVoiceUtil.set_voice_and_save(
            project, sound, "a_crop", "other", model, is_secondary=secondary, append=True,
        ) == ""

    if import_first:
        import_sample()
    assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 0, 3, "cut") == ""
    crop = Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(project, project.voice_references[0]))
    crop_bytes = crop.read_bytes()
    if not import_first:
        import_sample()
    imported_bytes = (voice_dir / "a_crop.flac").read_bytes()
    assert crop.read_bytes() == crop_bytes
    assert ProjectVoiceUtil.discard_voice_crop_and_save(project, 0) == ""
    assert (voice_dir / "a.flac").read_bytes() == original_bytes
    assert (voice_dir / "a_crop.flac").read_bytes() == imported_bytes
    assert not crop.exists()


@pytest.mark.parametrize("import_first", [True, False])
@pytest.mark.parametrize("secondary", [True, False])
def test_import_matching_crop_basename_is_distinct_and_safe(tmp_path, monkeypatch, import_first, secondary):
    voice_dir = tmp_path / "voice"
    write_sample(voice_dir, "a.flac", 6)
    project = make_project(voice_dir, {"file_name": "a.flac", "transcript": "full"})
    crop_id = uuid.UUID(int=1)
    crop_name = f"{crop_id.hex}.flac"
    # A bare original with this name must not reserve crops/<same name>.
    ids = iter([crop_id, uuid.UUID(int=2)])
    monkeypatch.setattr("tts_audiobook_tool.project_support.project_voice_util.uuid.uuid4", lambda: next(ids))
    imported_sound = Sound(np.full(24000 * 4, 0.03, dtype=np.float32), 24000)

    def import_sample():
        model = TtsModelType.require_by_id("indextts2_local") if secondary else GLM
        assert ProjectVoiceUtil.set_voice_and_save(
            project, imported_sound, crop_id.hex, "other", model, is_secondary=secondary, append=True,
        ) == ""

    if import_first:
        import_sample()
    assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 0, 3, "cut") == ""
    entry = project.voice_references[0]
    assert entry["crop_file_name"] == crop_name
    crop = voice_dir / "crops" / crop_name
    assert Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry)) == crop
    crop_bytes = crop.read_bytes()
    if not import_first:
        import_sample()
    original = voice_dir / crop_name
    original_bytes = original.read_bytes()
    assert original_bytes != crop_bytes
    assert crop.read_bytes() == crop_bytes
    assert ProjectVoiceUtil.resolve_voice_file_path(project, crop_name) == str(original)
    assert ProjectVoiceUtil.current_voice_reference_pair(project, GLM, 0) == (f"crops/{crop_name}", "cut")
    if secondary:
        assert project.get_model_setting("indextts2_local", "emo_voice") == crop_name
    else:
        assert project.voice_references[1]["file_name"] == crop_name
        assert ProjectVoiceUtil.current_voice_reference_pair(project, GLM, 1) == (crop_name, "other")
    assert ProjectVoiceUtil.discard_voice_crop_and_save(project, 0) == ""
    assert not crop.exists()
    assert original.read_bytes() == original_bytes


def test_duplicate_original_entries_have_independent_owned_crops(tmp_path):
    voice_dir = tmp_path / "voice"
    write_sample(voice_dir, "a.flac", 6)
    project = make_project(voice_dir, {"file_name": "a.flac", "transcript": "first"},
                           {"file_name": "a.flac", "transcript": "second"})
    assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 0, 3, "first cut") == ""
    assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 1, 2, 5, "second cut") == ""
    first, second = [Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry))
                     for entry in project.voice_references]
    assert first != second
    second_bytes = second.read_bytes()
    assert ProjectVoiceUtil.discard_voice_crop_and_save(project, 0) == ""
    assert not first.exists()
    assert second.read_bytes() == second_bytes
    assert ProjectVoiceUtil.effective_voice_reference(project, GLM, 1) == (
        f"crops/{project.voice_references[1]['crop_file_name']}", "second cut",
    )


@pytest.mark.parametrize("owner", ["original", "secondary", "other_crop", "own_original", "alias"])
@pytest.mark.parametrize("operation", ["apply", "discard"])
def test_shared_explicit_crop_never_overwrites_or_deletes_another_input(tmp_path, owner, operation):
    voice_dir = tmp_path / "voice"
    write_sample(voice_dir, "a.flac", 6)
    write_sample(voice_dir, "crops/shared.flac", 6)
    crop_entry = {"file_name": "a.flac", "transcript": "full", "crop_file_name": "shared.flac",
                  "crop_start": "0", "crop_end": "3", "crop_transcript": "cut"}
    others = []
    if owner == "original":
        others.append({"file_name": "crops/shared.flac", "transcript": "other"})
    elif owner == "own_original":
        crop_entry["file_name"] = "crops/shared.flac"
    elif owner == "other_crop":
        others.append({**crop_entry, "file_name": "b.flac", "transcript": "other"})
    elif owner == "alias":
        try:
            (voice_dir / "alias.flac").symlink_to(voice_dir / "crops/shared.flac")
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks unavailable on this filesystem")
        others.append({"file_name": "alias.flac", "transcript": "other"})
    project = make_project(voice_dir, crop_entry, *others)
    if owner == "secondary":
        project.set_model_setting("indextts2_local", "emo_voice", "crops/shared.flac")
    protected = voice_dir / "crops/shared.flac"
    protected_bytes = protected.read_bytes()
    if operation == "apply":
        assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 1, 4, "new cut") == ""
        assert project.voice_references[0]["crop_file_name"] != "shared.flac"
    else:
        assert ProjectVoiceUtil.discard_voice_crop_and_save(project, 0) == ""
        assert "crop_file_name" not in project.voice_references[0]
    assert protected.read_bytes() == protected_bytes


def test_allocator_skips_existing_and_referenced_missing_paths(tmp_path, monkeypatch):
    voice_dir = tmp_path / "voice"
    write_sample(voice_dir, "a.flac", 6)
    ids = [uuid.UUID(int=i) for i in (1, 2, 3)]
    occupied_name = f"crops/{ids[0].hex}.flac"
    missing_name = f"crops/{ids[1].hex}.flac"
    write_sample(voice_dir, occupied_name, 6)
    occupied_bytes = (voice_dir / occupied_name).read_bytes()
    project = make_project(voice_dir, {"file_name": "a.flac", "transcript": "full"},
                           {"file_name": missing_name, "transcript": "missing"})
    id_iter = iter(ids)
    monkeypatch.setattr("tts_audiobook_tool.project_support.project_voice_util.uuid.uuid4", lambda: next(id_iter))
    assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 0, 3, "cut") == ""
    assert project.voice_references[0]["crop_file_name"] == f"{ids[2].hex}.flac"
    assert (voice_dir / occupied_name).read_bytes() == occupied_bytes
    assert not (voice_dir / missing_name).exists()


@pytest.mark.parametrize("existing_crop", [True, False])
def test_failed_audio_write_preserves_existing_crop_and_cleans_allocations(tmp_path, monkeypatch, existing_crop):
    from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
    voice_dir = tmp_path / "voice"
    write_sample(voice_dir, "a.flac", 6)
    project = make_project(voice_dir, {"file_name": "a.flac", "transcript": "full"})
    if existing_crop:
        assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 0, 3, "cut") == ""
    previous_entries = [dict(entry) for entry in project.voice_references]
    crop_dir = voice_dir / "crops"
    previous_files = {path.name: path.read_bytes() for path in crop_dir.iterdir()} if crop_dir.exists() else {}
    monkeypatch.setattr(SoundFileUtil, "save_flac", lambda *_: "Disk full")
    assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 1, 4, "new cut") == "Disk full"
    assert project.voice_references == previous_entries
    assert {path.name: path.read_bytes() for path in crop_dir.iterdir()} == previous_files


def test_discard_save_failure_preserves_active_crop(tmp_path, monkeypatch):
    voice_dir = tmp_path / "voice"
    write_sample(voice_dir, "a.flac", 6)
    project = make_project(voice_dir, {"file_name": "a.flac", "transcript": "full"})
    assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 0, 3, "cut") == ""
    previous_entries = [dict(entry) for entry in project.voice_references]
    crop = Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(project, previous_entries[0]))
    crop_bytes = crop.read_bytes()
    monkeypatch.setattr(Project, "save", lambda _: "Disk full")
    assert ProjectVoiceUtil.discard_voice_crop_and_save(project, 0) == "Disk full"
    assert project.voice_references == previous_entries
    assert crop.read_bytes() == crop_bytes


@pytest.mark.parametrize("stem", ["crops/owned", "CROPS/owned", "./crops/owned", "crops\\owned", "alias/owned"])
def test_imports_cannot_enter_managed_crop_namespace(tmp_path, stem):
    voice_dir = tmp_path / "voice"
    write_sample(voice_dir, "a.flac", 6)
    write_sample(voice_dir, "crops/owned.flac", 3)
    crop_path = voice_dir / "crops/owned.flac"
    crop_bytes = crop_path.read_bytes()
    if stem.startswith("alias/"):
        try:
            (voice_dir / "alias").symlink_to(voice_dir / "crops", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks unavailable on this filesystem")
    project = make_project(voice_dir, {"file_name": "a.flac", "transcript": "full"})
    sound = Sound(np.full(24000 * 3, 0.02, dtype=np.float32), 24000)
    result = ProjectVoiceUtil.set_voice_and_save(project, sound, stem, "other", GLM, append=True)
    assert "managed trim directory" in result
    assert project.voice_references == [{"file_name": "a.flac", "transcript": "full"}]
    assert crop_path.read_bytes() == crop_bytes
