import os
from pathlib import Path
from typing import Any, cast

import pytest

from tts_audiobook_tool.constants import (
    PROJECT_TEXT_EPUB_FILE_NAME,
    PROJECT_TEXT_FILE_NAME,
    PROJECT_TEXT_RAW_FILE_NAME,
    PROJECT_VOICE_SUBDIR,
)
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def test_make_supporting_project_file_names_collects_project_local_voice_files(tmp_path: Path) -> None:
    # Absolute values are kept, reduced to their portable form: a saved path
    # written by another operating system still names a file that may have been
    # copied into the source project, and dropping it would lose the copy.
    project = Project.model_validate({
        'voice_references': [
            {'file_name': name, 'transcript': ''}
            for name in ['primary-a.flac', 'shared.flac', 'primary-b.flac', 'shared.flac', 'fish-s2.flac', 'higgs-v3.flac']
        ],
        'indextts2_emo_voice_file_name': 'emotion.flac',
        'fish_s2_server_voice_target': ['server-target.flac'],
        'higgs_v3_voice_target': ['https://example.com/voice.flac'],
    })
    # A project object that never passed through the load funnel can still
    # hold malformed entries; the collector must skip them rather than crash.
    project.voice_references.extend(cast(Any, [
        {'file_name': ''},
        {'file_name': 123},
        {'file_name': str(tmp_path / 'absolute.flac')},
        {'file_name': 'C:\\Users\\lee\\mybook\\voice\\narrator.flac'},
        {'file_name': 'primary-c.flac'},
    ]))
    # Scoped copies are obsolete and must not become a second source of files.
    project.model_settings.models['pocket_local'] = {'voice_references': [{'file_name': 'obsolete.flac'}]}

    result = ProjectTransferUtil.make_supporting_project_file_names(project)

    assert result == [
        PROJECT_TEXT_FILE_NAME,
        PROJECT_TEXT_RAW_FILE_NAME,
        PROJECT_TEXT_EPUB_FILE_NAME,
        'primary-a.flac',
        'shared.flac',
        'primary-b.flac',
        'fish-s2.flac',
        'higgs-v3.flac',
        'absolute.flac',
        'narrator.flac',
        'primary-c.flac',
        'emotion.flac',
    ]
    assert 'obsolete.flac' not in result
    assert 'server-target.flac' not in result
    assert 'https://example.com/voice.flac' not in result

    # The classified view drives copy destinations: text at the root, voice
    # files in the voice subdir.
    text_file_names, voice_file_names = ProjectTransferUtil.collect_supporting_project_file_names(project)
    assert text_file_names == result[:3]
    assert voice_file_names == result[3:]


def test_copy_supporting_project_files_copies_all_discovered_voice_files_and_reports_missing(
        tmp_path: Path,
) -> None:
    source_dir = tmp_path / 'source'
    dest_dir = tmp_path / 'destination'
    source_dir.mkdir()
    dest_dir.mkdir()

    source_project = Project(
        dir_path=str(source_dir),
        chatterbox_voice_file_name=['voice-a.flac', 'voice-b.flac', 'missing.flac'],
        indextts2_emo_voice_file_name='emotion.flac',
    )
    contents = {
        PROJECT_TEXT_FILE_NAME: b'project text',
        PROJECT_TEXT_RAW_FILE_NAME: b'raw text',
        PROJECT_TEXT_EPUB_FILE_NAME: b'epub text',
        'voice-a.flac': b'voice a',
        'voice-b.flac': b'voice b',
        'emotion.flac': b'emotion voice',
    }
    contents_voice_subdir = {'voice-a.flac', 'voice-b.flac', 'emotion.flac'}
    for file_name, content in contents.items():
        (source_dir / file_name).write_bytes(content)

    text_file_names, voice_file_names = ProjectTransferUtil.collect_supporting_project_file_names(source_project)
    missing_paths = ProjectTransferUtil.copy_supporting_project_files(
        Project(dir_path=str(dest_dir)),
        str(source_dir),
        text_file_names,
        voice_file_names,
    )

    assert missing_paths == [str(source_dir / 'missing.flac')]
    for file_name, content in contents.items():
        if file_name in contents_voice_subdir:
            assert (dest_dir / PROJECT_VOICE_SUBDIR / file_name).read_bytes() == content
        else:
            assert (dest_dir / file_name).read_bytes() == content
    assert not (dest_dir / 'missing.flac').exists()


def test_copy_supporting_project_files_upgrades_legacy_root_voice_files(tmp_path: Path) -> None:
    """
    Voice files found at a legacy project root are copied into the destination
    project's voice subdir. Bare saved references resolve against the voice
    subdir first, so the new project works on the current layout alone.
    """
    source_dir = tmp_path / 'source'
    dest_dir = tmp_path / 'destination'
    (source_dir / PROJECT_VOICE_SUBDIR).mkdir(parents=True)
    dest_dir.mkdir()

    source_project = Project(
        dir_path=str(source_dir),
        chatterbox_voice_file_name=['voice/narrator.flac', 'root.flac'],
    )
    (source_dir / PROJECT_VOICE_SUBDIR / 'narrator.flac').write_bytes(b'narrator')
    (source_dir / 'root.flac').write_bytes(b'root')

    missing_paths = ProjectTransferUtil.copy_supporting_project_files(
        Project(dir_path=str(dest_dir)),
        str(source_dir),
        *ProjectTransferUtil.collect_supporting_project_file_names(source_project),
    )

    assert missing_paths == [
        str(source_dir / PROJECT_TEXT_FILE_NAME),
        str(source_dir / PROJECT_TEXT_RAW_FILE_NAME),
        str(source_dir / PROJECT_TEXT_EPUB_FILE_NAME),
    ]
    assert (dest_dir / PROJECT_VOICE_SUBDIR / 'narrator.flac').read_bytes() == b'narrator'
    assert (dest_dir / PROJECT_VOICE_SUBDIR / 'root.flac').read_bytes() == b'root'
    assert not (dest_dir / 'root.flac').exists()

    # The bare saved reference resolves in the new project via the voice subdir.
    from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
    dest_project = Project(dir_path=str(dest_dir), chatterbox_voice_file_name=['root.flac'])
    resolved = ProjectVoiceUtil.resolve_voice_file_path(dest_project, 'root.flac')
    assert os.path.exists(resolved) and resolved == str(dest_dir / PROJECT_VOICE_SUBDIR / 'root.flac')


def test_copy_supporting_project_files_recovers_names_saved_by_another_os(tmp_path: Path) -> None:
    """
    A voice file name recorded in another operating system's grammar still names
    a file that may have been copied alongside the ABR audio, so it is reduced
    to its portable form and looked up rather than discarded.
    """
    source_dir = tmp_path / 'source'
    dest_dir = tmp_path / 'destination'
    (source_dir / PROJECT_VOICE_SUBDIR).mkdir(parents=True)
    dest_dir.mkdir()

    source_project = Project.model_validate({
        'chatterbox_voice_file_name': ['C:\\Users\\lee\\mybook\\voice\\narrator.flac'],
    })
    (source_dir / PROJECT_VOICE_SUBDIR / 'narrator.flac').write_bytes(b'narrator')

    missing_paths = ProjectTransferUtil.copy_supporting_project_files(
        Project(dir_path=str(dest_dir)),
        str(source_dir),
        *ProjectTransferUtil.collect_supporting_project_file_names(source_project),
    )

    assert missing_paths == [
        str(source_dir / PROJECT_TEXT_FILE_NAME),
        str(source_dir / PROJECT_TEXT_RAW_FILE_NAME),
        str(source_dir / PROJECT_TEXT_EPUB_FILE_NAME),
    ]
    assert (dest_dir / PROJECT_VOICE_SUBDIR / 'narrator.flac').read_bytes() == b'narrator'


def test_copy_supporting_project_files_does_not_escape_the_project_dir(tmp_path: Path) -> None:
    """
    A stored value that still carries a leading separator must not be joined in
    a way that discards the destination project directory.
    """
    source_dir = tmp_path / 'source'
    dest_dir = tmp_path / 'destination'
    source_dir.mkdir()
    dest_dir.mkdir()

    source_project = Project.model_validate({
        'chatterbox_voice_file_name': ['/etc/escaped.flac'],
    })
    (source_dir / 'escaped.flac').write_bytes(b'escaped')

    missing_paths = ProjectTransferUtil.copy_supporting_project_files(
        Project(dir_path=str(dest_dir)),
        str(source_dir),
        *ProjectTransferUtil.collect_supporting_project_file_names(source_project),
    )

    assert missing_paths == [
        str(source_dir / PROJECT_TEXT_FILE_NAME),
        str(source_dir / PROJECT_TEXT_RAW_FILE_NAME),
        str(source_dir / PROJECT_TEXT_EPUB_FILE_NAME),
    ]
    assert (dest_dir / PROJECT_VOICE_SUBDIR / 'escaped.flac').read_bytes() == b'escaped'


def _cropped_reference() -> dict[str, str]:
    return {
        'file_name': 'narrator.flac',
        'transcript': 'Original transcript.',
        'crop_file_name': 'a.flac',
        'crop_start': '1.25',
        'crop_end': '4.5',
        'crop_transcript': 'Trimmed transcript.',
    }


def test_transfer_collector_includes_missing_active_explicit_crops(tmp_path: Path) -> None:
    project = Project(dir_path=str(tmp_path), voice_references=[_cropped_reference()])
    project.voice_references.extend([
        # Old implicit crop metadata must not infer narrator_crop.flac.
        {'file_name': 'old.flac', 'crop_start': '0', 'crop_end': '3', 'crop_transcript': 'Old crop.'},
        {'file_name': 'invalid.flac', 'crop_file_name': 'invalid.flac',
         'crop_start': '4', 'crop_end': '1'},
    ])

    _, names = ProjectTransferUtil.collect_supporting_project_file_names(project)

    assert names == ['narrator.flac', 'old.flac', 'invalid.flac', 'crops/a.flac']
    assert not (tmp_path / PROJECT_VOICE_SUBDIR / 'crops/a.flac').exists()


@pytest.mark.parametrize('original_subdir', ['', PROJECT_VOICE_SUBDIR], ids=['legacy-root', 'voice'])
def test_transfer_preserves_explicit_crop_bytes_metadata_and_effective_pair(
    tmp_path: Path, original_subdir: str,
) -> None:
    source_dir = tmp_path / 'source'
    dest_dir = tmp_path / 'destination'
    dest_dir.mkdir()
    (source_dir / PROJECT_VOICE_SUBDIR / 'crops').mkdir(parents=True)
    original_path = source_dir / original_subdir / 'narrator.flac'
    original_path.write_bytes(b'complete original audio')
    crop_path = source_dir / PROJECT_VOICE_SUBDIR / 'crops/a.flac'
    crop_path.write_bytes(b'explicit trimmed audio')
    source = Project(dir_path=str(source_dir), voice_references=[_cropped_reference()])
    dest = Project(dir_path=str(dest_dir))
    ProjectTransferUtil.apply_project_settings(dest, source)
    _, voice_names = ProjectTransferUtil.collect_supporting_project_file_names(source)

    assert ProjectTransferUtil.copy_supporting_project_files(dest, str(source_dir), [], voice_names) == []
    assert dest.save() == ''
    reloaded = ProjectLoadUtil.load_using_dir_path(str(dest_dir))
    assert isinstance(reloaded, Project)
    assert reloaded.voice_references == source.voice_references == [_cropped_reference()]
    assert (dest_dir / PROJECT_VOICE_SUBDIR / 'narrator.flac').read_bytes() == original_path.read_bytes()
    assert (dest_dir / PROJECT_VOICE_SUBDIR / 'crops/a.flac').read_bytes() == crop_path.read_bytes()
    assert not (dest_dir / PROJECT_VOICE_SUBDIR / 'a.flac').exists()
    model = TtsModelType.require_by_id('fish_s2_local')
    assert ProjectVoiceUtil.effective_voice_reference(source, model, 0) == ('crops/a.flac', 'Trimmed transcript.')
    assert ProjectVoiceUtil.effective_voice_reference(reloaded, model, 0) == ('crops/a.flac', 'Trimmed transcript.')
    assert ProjectVoiceUtil.effective_voice_file_path(reloaded, reloaded.voice_references[0]) == str(
        dest_dir / PROJECT_VOICE_SUBDIR / 'crops/a.flac'
    )


def test_transfer_keeps_original_matching_crop_basename_unambiguous(tmp_path: Path) -> None:
    source_dir = tmp_path / 'source'
    dest_dir = tmp_path / 'destination'
    dest_dir.mkdir()
    (source_dir / PROJECT_VOICE_SUBDIR / 'crops').mkdir(parents=True)
    original = source_dir / PROJECT_VOICE_SUBDIR / 'a.flac'
    crop = source_dir / PROJECT_VOICE_SUBDIR / 'crops/a.flac'
    original.write_bytes(b'complete original audio')
    crop.write_bytes(b'explicit trimmed audio')
    entry = {**_cropped_reference(), 'file_name': 'a.flac'}
    source = Project(dir_path=str(source_dir), voice_references=[entry])
    dest = Project(dir_path=str(dest_dir))
    ProjectTransferUtil.apply_project_settings(dest, source)
    _, voice_names = ProjectTransferUtil.collect_supporting_project_file_names(source)

    assert voice_names == ['a.flac', 'crops/a.flac']
    assert ProjectTransferUtil.copy_supporting_project_files(dest, str(source_dir), [], voice_names) == []
    assert dest.voice_references == [entry]
    assert (dest_dir / PROJECT_VOICE_SUBDIR / 'a.flac').read_bytes() == original.read_bytes()
    assert (dest_dir / PROJECT_VOICE_SUBDIR / 'crops/a.flac').read_bytes() == crop.read_bytes()
    assert ProjectVoiceUtil.resolve_voice_file_path(dest, 'a.flac') == str(
        dest_dir / PROJECT_VOICE_SUBDIR / 'a.flac'
    )
    assert ProjectVoiceUtil.get_cropped_voice_relative_path(dest.voice_references[0]) == 'crops/a.flac'
    assert ProjectVoiceUtil.resolve_cropped_voice_file_path(dest, dest.voice_references[0]) == str(
        dest_dir / PROJECT_VOICE_SUBDIR / 'crops/a.flac'
    )
    model = TtsModelType.require_by_id('fish_s2_local')
    assert ProjectVoiceUtil.current_voice_reference_pair(dest, model, 0) == ('crops/a.flac', 'Trimmed transcript.')
    assert ProjectVoiceUtil.discard_voice_crop_and_save(dest, 0) == ''
    assert not (dest_dir / PROJECT_VOICE_SUBDIR / 'crops/a.flac').exists()
    assert (dest_dir / PROJECT_VOICE_SUBDIR / 'a.flac').read_bytes() == original.read_bytes()
    assert ProjectVoiceUtil.current_voice_reference_pair(dest, model, 0) == ('a.flac', 'Original transcript.')


def test_voice_transfer_prefers_generation_original_over_same_named_root_file(tmp_path: Path) -> None:
    source_dir = tmp_path / 'source'
    dest_dir = tmp_path / 'destination'
    dest_dir.mkdir()
    (source_dir / PROJECT_VOICE_SUBDIR / 'crops').mkdir(parents=True)
    root_original = source_dir / 'narrator.flac'
    voice_original = source_dir / PROJECT_VOICE_SUBDIR / 'narrator.flac'
    root_original.write_bytes(b'old root original')
    voice_original.write_bytes(b'generation original')
    (source_dir / PROJECT_VOICE_SUBDIR / 'crops/a.flac').write_bytes(b'active crop audio')
    source = Project(dir_path=str(source_dir), voice_references=[_cropped_reference()])
    dest = Project(dir_path=str(dest_dir))
    ProjectTransferUtil.apply_project_settings(dest, source)
    _, voice_names = ProjectTransferUtil.collect_supporting_project_file_names(source)

    assert ProjectVoiceUtil.resolve_voice_file_path(source, 'narrator.flac') == str(voice_original)
    # Generic/text callers still retain their original root-first behavior.
    assert ProjectTransferUtil.find_supporting_project_file_source_path(
        str(source_dir), 'narrator.flac',
    ).path == str(root_original)
    assert ProjectTransferUtil.copy_supporting_project_files(dest, str(source_dir), [], voice_names) == []
    assert dest.save() == ''
    reloaded = ProjectLoadUtil.load_using_dir_path(str(dest_dir))
    assert isinstance(reloaded, Project)
    assert reloaded.voice_references == source.voice_references == [_cropped_reference()]
    assert (dest_dir / PROJECT_VOICE_SUBDIR / 'narrator.flac').read_bytes() == voice_original.read_bytes()
    assert (dest_dir / PROJECT_VOICE_SUBDIR / 'crops/a.flac').read_bytes() == b'active crop audio'
    model = TtsModelType.require_by_id('fish_s2_local')
    assert ProjectVoiceUtil.effective_voice_reference(reloaded, model, 0) == ('crops/a.flac', 'Trimmed transcript.')
    assert ProjectVoiceUtil.effective_voice_file_path(reloaded, reloaded.voice_references[0]) == str(
        dest_dir / PROJECT_VOICE_SUBDIR / 'crops/a.flac'
    )


@pytest.mark.parametrize('original_subdir', ['', PROJECT_VOICE_SUBDIR], ids=['legacy-root', 'voice'])
def test_missing_crop_never_copies_unrelated_root_basename_or_implicit_sibling(
    tmp_path: Path, original_subdir: str,
) -> None:
    source_dir = tmp_path / 'source'
    dest_dir = tmp_path / 'destination'
    dest_dir.mkdir()
    (source_dir / PROJECT_VOICE_SUBDIR).mkdir(parents=True)
    (source_dir / original_subdir / 'narrator.flac').write_bytes(b'original audio')
    for relative_path in ['crops/a.flac', 'a.flac', 'voice/a.flac',
                          'narrator_crop.flac', 'voice/narrator_crop.flac']:
        decoy = source_dir / relative_path
        decoy.parent.mkdir(parents=True, exist_ok=True)
        decoy.write_bytes(b'unrelated audio')
    source = Project(dir_path=str(source_dir), voice_references=[_cropped_reference()])
    dest = Project(dir_path=str(dest_dir))
    ProjectTransferUtil.apply_project_settings(dest, source)
    _, voice_names = ProjectTransferUtil.collect_supporting_project_file_names(source)

    missing = ProjectTransferUtil.copy_supporting_project_files(dest, str(source_dir), [], voice_names)

    assert missing == [str(source_dir / PROJECT_VOICE_SUBDIR / 'crops/a.flac')]
    assert (dest_dir / PROJECT_VOICE_SUBDIR / 'narrator.flac').read_bytes() == b'original audio'
    assert not (dest_dir / PROJECT_VOICE_SUBDIR / 'crops/a.flac').exists()
    assert not (dest_dir / PROJECT_VOICE_SUBDIR / 'a.flac').exists()
    assert dest.voice_references == source.voice_references
    model = TtsModelType.require_by_id('fish_s2_local')
    assert ProjectVoiceUtil.effective_voice_reference(dest, model, 0) == ('narrator.flac', 'Original transcript.')
    assert ProjectVoiceUtil.effective_voice_file_path(dest, dest.voice_references[0]) == str(
        dest_dir / PROJECT_VOICE_SUBDIR / 'narrator.flac'
    )


@pytest.mark.parametrize('nested_subdir', ['', PROJECT_VOICE_SUBDIR], ids=['legacy-root', 'voice'])
def test_transfer_preserves_nested_original_paths_without_basename_overwrites(
    tmp_path: Path, nested_subdir: str,
) -> None:
    source_dir = tmp_path / 'source'
    dest_dir = tmp_path / 'destination'
    dest_dir.mkdir()
    (source_dir / PROJECT_VOICE_SUBDIR).mkdir(parents=True)
    expected = {'first/narrator.flac': b'first voice', 'second/narrator.flac': b'second voice'}
    for name, content in expected.items():
        path = source_dir / nested_subdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    # A root basename must not take precedence over an exact nested voice path.
    (source_dir / 'narrator.flac').write_bytes(b'legacy bare voice')
    expected['narrator.flac'] = b'legacy bare voice'
    source = Project(dir_path=str(source_dir), voice_references=[
        {'file_name': name, 'transcript': name} for name in expected
    ])
    dest = Project(dir_path=str(dest_dir))
    ProjectTransferUtil.apply_project_settings(dest, source)
    _, voice_names = ProjectTransferUtil.collect_supporting_project_file_names(source)

    assert ProjectTransferUtil.copy_supporting_project_files(dest, str(source_dir), [], voice_names) == []

    for name, content in expected.items():
        assert (dest_dir / PROJECT_VOICE_SUBDIR / name).read_bytes() == content
        assert Path(ProjectVoiceUtil.resolve_voice_file_path(dest, name)).read_bytes() == content


def test_get_snapshot_source_dir_falls_back_to_the_abr_files_own_directory(tmp_path: Path) -> None:
    abr_dir = tmp_path / 'mybook'
    abr_dir.mkdir()
    (abr_dir / PROJECT_VOICE_SUBDIR).mkdir()
    abr_path = abr_dir / 'mybook.abr.m4b'
    abr_path.write_bytes(b'')

    assert ProjectTransferUtil.get_snapshot_source_dir({}, str(abr_path)) == str(abr_dir)

    # A directory that does not look like a project is not offered as a source.
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    assert ProjectTransferUtil.get_snapshot_source_dir({}, str(elsewhere / 'x.abr.m4b')) == ''


def test_get_snapshot_source_dir_prefers_a_recorded_display_dir(tmp_path: Path) -> None:
    recorded = tmp_path / 'recorded'
    (recorded / PROJECT_TEXT_FILE_NAME).parent.mkdir(parents=True)
    (recorded / PROJECT_TEXT_FILE_NAME).write_text('{}')

    snapshot = {'source_dir_display': str(recorded)}
    assert ProjectTransferUtil.get_snapshot_source_dir(snapshot, str(tmp_path / 'other.abr.m4b')) == str(recorded)

    # A recorded directory from another machine is not usable, so the fallback applies.
    abr_dir = tmp_path / 'abr'
    abr_dir.mkdir()
    (abr_dir / PROJECT_VOICE_SUBDIR).mkdir()
    snapshot = {'source_dir_display': 'C:\\Users\\lee\\mybook'}
    assert ProjectTransferUtil.get_snapshot_source_dir(snapshot, str(abr_dir / 'x.abr.m4b')) == str(abr_dir)


def test_get_snapshot_source_dir_display_reads_older_snapshots(tmp_path: Path) -> None:
    assert ProjectTransferUtil.get_snapshot_source_dir_display({'dir_path': 'C:\\Users\\lee\\mybook'}) == 'C:\\Users\\lee\\mybook'
    assert ProjectTransferUtil.get_snapshot_source_dir_display({'source_dir_display': '/home/lee/mybook'}) == '/home/lee/mybook'
    assert ProjectTransferUtil.get_snapshot_source_dir_display({}) == ''


def test_find_foreign_path_targets_warns_without_changing_anything(tmp_path: Path) -> None:
    project = Project.model_validate({
        'vibevoice_target': 'C:\\Users\\lee\\models\\VibeVoice',
        'qwen3_target': 'microsoft/Qwen3-TTS',
        'omnivoice_target': '/home/lee/models/omnivoice',
    })

    found = dict(ProjectTransferUtil.find_foreign_path_targets(project))

    assert found == {'vibevoice_local_target': 'C:\\Users\\lee\\models\\VibeVoice'}
