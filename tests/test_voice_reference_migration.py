"""Project v4 migration and blocking choice at persistence boundaries."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tts_audiobook_tool import ask
from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM, PROJECT_SPEC_VERSION
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.voice_reference_migration import (
    VoiceReferenceMigrationRequired, prepare_project_voice_references,
)


def ref(name='a.flac', transcript='spoken text'):
    return {'file_name': name, 'transcript': transcript}


def private_sources():
    return {'version': 3, 'model_settings': {'models': {
        'glm_local': {'voice_references': [ref('first.flac'), ref('second.flac', 'second')]},
        'mira_local': {'voice_references': [ref('other.flac')]},
    }, 'shared': {}}}


@pytest.mark.parametrize('source', [
    {}, {'version': 3}, {'glm_voice_file_name': ''},
    {'model_settings': {'models': {'glm_local': {'voice_references': []}}}},
])
def test_empty_sources(source):
    assert prepare_project_voice_references(source)['voice_references'] == []


@pytest.mark.parametrize('source', [
    {'glm_voice_file_name': ['a.flac', 'b.flac', 'a.flac'], 'glm_voice_text': ['A', 'B']},
    {'version': 3, 'model_settings': {'models': {'glm_local': {'voice_references': [ref('a.flac', 'A'), ref('b.flac', 'B'), ref('a.flac', '')]}}}},
])
def test_one_source_preserves_order_pairing_duplicates_and_input(source):
    original = deepcopy(source)
    migrated = prepare_project_voice_references(source)
    assert source == original
    assert migrated['version'] == PROJECT_SPEC_VERSION == 4
    assert migrated['voice_references'] == [ref('a.flac', 'A'), ref('b.flac', 'B'), ref('a.flac', '')]
    assert 'glm_voice_file_name' not in migrated
    for obj in migrated.get('model_settings', {}).get('models', {}).values():
        assert 'voice_references' not in obj


@pytest.mark.parametrize('source', [
    {'fish_s2_voice_file_name': 'a.flac'},
    {'moss_voice_file_name': 'a.flac'},
    {'model_settings': {'shared': {'fish_s2': {'model_ids': ['fish_s2_local', 'fish_s2_sglomni'], 'voice_references': [ref('a.flac', '')]}}}},
    {'model_settings': {'shared': {'moss': {'model_ids': ['moss_local', 'moss_delay_sglomni', 'moss_local_sglomni'], 'voice_references': [ref('a.flac', '')]}}}},
])
def test_shared_historical_owner_counts_once(source, monkeypatch):
    monkeypatch.setattr(ask, 'ask_input', lambda *_args, **_kwargs: pytest.fail('should be silent'))
    project = Project.model_validate(source)
    assert project.voice_references == [ref('a.flac', '')]
    assert all('voice_references' not in obj for obj in project.model_settings.models.values())


def test_multiple_sources_noninteractive_error_does_not_mutate_input():
    source = private_sources()
    original = deepcopy(source)
    with pytest.raises(VoiceReferenceMigrationRequired, match='interactively') as exc:
        prepare_project_voice_references(source)
    assert len(exc.value.sources) == 2
    assert source == original


def test_independent_identical_lists_still_require_choice():
    with pytest.raises(VoiceReferenceMigrationRequired):
        prepare_project_voice_references({'glm_voice_file_name': 'a.flac', 'mira_voice_file_name': 'a.flac'})


def test_exact_prompt_all_filenames_and_no_default(monkeypatch):
    output = []
    monkeypatch.setattr('tts_audiobook_tool.util.printt', lambda message='': output.append(message))
    answers = iter(['', 'invalid', '2'])
    monkeypatch.setattr(ask, 'ask_input', lambda *_args: next(answers))
    migrated = prepare_project_voice_references(private_sources(), prompt=True)
    assert output[0] == (
        f'🔔 {COL_ACCENT}Action required:\n'
        f'{COL_DEFAULT}This project contains voice clone lists for multiple models, but the app \n'
        'now uses one voice clone list per project, shared by all models. \n'
        'Choose which model’s voice clone list to keep:'
    )
    assert output[1:] == [
        '',
        f'[1] GLM-TTS glm_local: {COL_DIM} first.flac, second.flac',
        f'[2] MiraTTS mira_local: {COL_DIM} other.flac',
        '[0] Abort',
        '',
        'Please choose a number from 1 to 2, or 0 to abort.',
        '',
        'Please choose a number from 1 to 2, or 0 to abort.',
        '',
    ]
    assert migrated['voice_references'] == [ref('other.flac')]


@pytest.mark.parametrize('explicit', [[], [ref('new.flac')]])
def test_explicit_top_level_list_wins_even_empty(explicit):
    source = private_sources()
    source['voice_references'] = explicit
    source['glm_voice_file_name'] = 'stale.flac'
    migrated = prepare_project_voice_references(source)
    assert migrated['voice_references'] == explicit
    assert 'glm_voice_file_name' not in migrated
    assert all('voice_references' not in obj for obj in migrated['model_settings']['models'].values())


def test_owner_object_masks_flat_list_and_retains_other_settings():
    project = Project.model_validate({'glm_voice_file_name': 'stale.flac', 'model_settings': {
        'models': {'glm_local': {'parameters': {'sr': 48000}}},
    }})
    assert project.voice_references == []
    assert project.get_model_setting('glm_local', 'sr') == 48000


def test_unavailable_model_and_unrelated_unknown_objects_survive():
    source = {'model_settings': {'models': {
        'missing_model': {'voice_references': [ref()], 'future': {'anything': 42}},
    }, 'shared': {'future_group': {'future': 12}}}}
    project = Project.model_validate(source)
    # Scoped voice data in unknown whole objects is disregarded as a migration
    # source and dropped, while the unrelated unknown sections survive intact.
    assert project.voice_references == []
    saved = ProjectSerializationUtil.to_project_json_dict(project)
    assert saved['model_settings']['models']['missing_model'] == {'future': {'anything': 42}}
    assert saved['model_settings']['shared']['future_group'] == {'future': 12}


@pytest.mark.parametrize('source', [
    {'model_settings': {'models': {'glm_local': {'voice_references': 'bad'}}}},
    {'model_settings': {'models': {'glm_local': {'voice_references': [{'file_name': 'a.flac', 'transcript': None}]}}}},
    {'voice_references': None},
])
def test_malformed_voice_data_fails_before_permissive_cleanup(source):
    with pytest.raises(ValueError):
        Project.model_validate(source)


@pytest.mark.parametrize('source, expected', [
    ({'glm_voice_file_name': 42}, []),
    ({'glm_voice_file_name': 'a.flac', 'glm_voice_transcript': [None]}, [ref('a.flac', '')]),
])
def test_legacy_flat_voice_junk_is_tolerated(source, expected):
    # Old flat fields were permissively flattened on load, so non-string
    # entries are ignored rather than failing the whole project.
    assert Project.model_validate(source).voice_references == expected


def test_unusable_file_name_drops_entry_instead_of_failing():
    # A name that cannot be a project-local path drops just that entry,
    # matching the old flat-field handling of unusable paths.
    project = Project.model_validate({'voice_references': [{'file_name': ''}, ref('a.flac')]})
    assert project.voice_references == [ref('a.flac')]


def test_canonical_paths_and_missing_transcripts():
    project = Project.model_validate({'voice_references': [{'file_name': r'C:\book\voice\a.flac'}]})
    assert project.voice_references == [ref('a.flac', '')]


def write_project(tmp_path, source):
    path = tmp_path / 'project.json'
    raw = (json.dumps(source, indent=3) + '\n').encode()
    path.write_bytes(raw)
    return path, raw


def test_load_saves_immediately_backs_up_original_and_never_reprompts(tmp_path, monkeypatch):
    path, original = write_project(tmp_path, private_sources())
    monkeypatch.setattr(ask, 'ask_input', lambda *_args: '1')
    result = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False, prompt_on_migration=True)
    assert isinstance(result, Project)
    assert result.voice_references == [ref('first.flac'), ref('second.flac', 'second')]
    assert (tmp_path / 'project.json.pre-v4.bak').read_bytes() == original
    saved = json.loads(path.read_bytes())
    assert saved['version'] == 4
    assert saved['voice_references'] == result.voice_references
    assert 'voice_references' not in saved['model_settings']['models']['glm_local']
    monkeypatch.setattr(ask, 'ask_input', lambda *_args: pytest.fail('should not prompt twice'))
    result2 = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)
    assert isinstance(result2, Project)
    assert result2.voice_references == result.voice_references
    assert not (tmp_path / 'project.json.pre-v4.bak.2').exists()


@pytest.mark.parametrize('interactive', [False, True])
def test_ambiguous_noninteractive_or_cancelled_load_does_not_write(tmp_path, monkeypatch, interactive):
    source = private_sources()
    source['text'] = []  # An existing inline-text migration must not save first.
    path, original = write_project(tmp_path, source)
    monkeypatch.setattr(ask, 'ask_input', lambda *_args: '0' if interactive else pytest.fail('must not prompt'))
    if interactive:
        result = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False, prompt_on_migration=True)
        assert isinstance(result, str)
        assert 'cancelled' in result
    else:
        from tts_audiobook_tool.project_support.voice_reference_migration import VoiceReferenceMigrationRequired
        with pytest.raises(VoiceReferenceMigrationRequired):
            ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False, prompt_on_migration=False)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_save_failure_aborts_load_with_original_backup(tmp_path, monkeypatch):
    path, original = write_project(tmp_path, {'version': 3, 'glm_voice_file_name': 'a.flac'})
    monkeypatch.setattr(Project, 'save', lambda self: 'save failed')
    result = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)
    assert result == 'save failed'
    assert path.read_bytes() == original
    assert (tmp_path / 'project.json.pre-v4.bak').read_bytes() == original


def test_existing_backup_not_overwritten(tmp_path):
    _path, original = write_project(tmp_path, {'version': 3})
    backup = tmp_path / 'project.json.pre-v4.bak'
    backup.write_bytes(b'older backup')
    result = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)
    assert isinstance(result, Project)
    assert backup.read_bytes() == b'older backup'
    assert (tmp_path / 'project.json.pre-v4.bak.2').read_bytes() == original


def test_retained_retired_group_does_not_back_up_on_every_open(tmp_path, monkeypatch):
    # A retired historical shared group (e.g. moss) whose reconciliation fails
    # is retained verbatim on disk, scoped voice_references included; a
    # re-open must not treat that as a pending migration and mint a new
    # numbered backup each time.
    source = {'version': 3, 'model_settings': {
        'shared': {'moss': {'model_ids': ['bogus_model'], 'voice_references': [ref('old.flac', '')]}},
    }}
    path, original = write_project(tmp_path, source)
    monkeypatch.setattr(ask, 'ask_input', lambda *_args: pytest.fail('should be silent'))
    result = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)
    assert isinstance(result, Project)
    assert result.voice_references == []
    assert (tmp_path / 'project.json.pre-v4.bak').read_bytes() == original
    saved = json.loads(path.read_bytes())
    assert saved['version'] == PROJECT_SPEC_VERSION
    # The unusable retired group is retained verbatim on disk.
    assert saved['model_settings']['shared']['moss']['voice_references'] == [ref('old.flac', '')]
    result2 = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)
    assert isinstance(result2, Project)
    assert result2.voice_references == []
    assert not (tmp_path / 'project.json.pre-v4.bak.2').exists()


def test_backup_failure_aborts_before_save(tmp_path, monkeypatch):
    import builtins
    path, original = write_project(tmp_path, {'version': 3})
    real_open = builtins.open
    def fail_backup(file, mode='r', *args, **kwargs):
        if mode == 'xb':
            raise PermissionError('backup forbidden')
        return real_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', fail_backup)
    result = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)
    assert isinstance(result, str) and 'back up' in result
    assert path.read_bytes() == original


def test_legacy_accessors_share_one_list_and_worker_snapshot_roundtrip():
    project = Project()
    project.set_model_setting('glm_local', 'file_name', ['a.flac'])
    project.set_model_setting('glm_local', 'transcript', ['A'])
    assert project.get_model_setting('mira_local', 'file_name') == ['a.flac']
    assert project.get_model_setting('fish_s2_sglomni', 'transcript') == ['A']
    snapshot = ProjectSerializationUtil.to_snapshot_dict(project)
    assert snapshot['version'] == 4
    assert Project.model_validate(snapshot).voice_references == [ref('a.flac', 'A')]
    assert not project.model_settings.models and not project.model_settings.shared


def test_historical_moss_fallback_only_counts_when_still_used():
    ids = ['moss_local', 'moss_delay_sglomni', 'moss_local_sglomni']
    source = {'model_settings': {
        'models': {id: {'voice_references': []} for id in ids},
        'shared': {'moss': {'model_ids': ids, 'voice_references': [ref('old.flac')]}},
    }}
    assert prepare_project_voice_references(source)['voice_references'] == []
    del source['model_settings']['models']['moss_local_sglomni']
    assert prepare_project_voice_references(source)['voice_references'] == [ref('old.flac')]
    source['model_settings']['models']['moss_local']['voice_references'] = [ref('private.flac')]
    with pytest.raises(VoiceReferenceMigrationRequired) as exc:
        prepare_project_voice_references(source)
    assert {s.model_ids for s in exc.value.sources} == {('moss_local',), ('moss_local_sglomni',)}


def test_historical_moss_flat_fallback_survives_partial_private_override():
    source = {'moss_voice_file_name': 'old.flac', 'model_settings': {
        'models': {'moss_local': {'voice_references': []}},
    }}
    assert prepare_project_voice_references(source)['voice_references'] == [ref('old.flac', '')]


def test_invalid_shared_voice_membership_fails_without_source_mutation():
    source = {'model_settings': {'shared': {'fish_s2': {
        'model_ids': ['fish_s2_local', 'unexpected'], 'voice_references': [ref()],
    }}}}
    original = deepcopy(source)
    with pytest.raises(ValueError, match='expected members'):
        prepare_project_voice_references(source)
    assert source == original


def test_future_format_load_refuses_to_rewrite(tmp_path):
    path, original = write_project(tmp_path, {'version': 5, 'voice_references': []})
    result = ProjectLoadUtil.load_using_dir_path(str(tmp_path), prompt_on_warnings=False)
    assert isinstance(result, str) and 'update the app' in result
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
