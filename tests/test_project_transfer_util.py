import os
from pathlib import Path
from typing import Any, cast

from tts_audiobook_tool.constants import (
    PROJECT_TEXT_EPUB_FILE_NAME,
    PROJECT_TEXT_FILE_NAME,
    PROJECT_TEXT_RAW_FILE_NAME,
    PROJECT_VOICE_SUBDIR,
)
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil


def test_make_supporting_project_file_names_collects_project_local_voice_files(tmp_path: Path) -> None:
    # Absolute values are kept, reduced to their portable form: a saved path
    # written by another operating system still names a file that may have been
    # copied into the source project, and dropping it would lose the copy.
    project = Project.model_validate({
        'chatterbox_voice_file_name': ['primary-a.flac', 'shared.flac'],
        'mira_voice_file_name': ['primary-b.flac', 'shared.flac'],
        'indextts2_emo_voice_file_name': 'emotion.flac',
        'fish_s2_voice_file_name': ['fish-s2.flac'],
        'fish_s2_server_voice_target': ['server-target.flac'],
        'higgs_v3_voice_file_name': ['higgs-v3.flac'],
        'higgs_v3_voice_target': ['https://example.com/voice.flac'],
    })
    # A project object that never passed through the load funnel can still
    # hold malformed entries; the collector must skip them rather than crash.
    project.model_settings.models['pocket_local'] = cast(Any, {
        'voice_references': [
            {'file_name': ''},
            {'file_name': 123},
            {'file_name': str(tmp_path / 'absolute.flac')},
            {'file_name': 'C:\\Users\\lee\\mybook\\voice\\narrator.flac'},
            {'file_name': 'primary-c.flac'},
        ],
    })

    result = ProjectTransferUtil.make_supporting_project_file_names(project)

    assert result == [
        PROJECT_TEXT_FILE_NAME,
        PROJECT_TEXT_RAW_FILE_NAME,
        PROJECT_TEXT_EPUB_FILE_NAME,
        'primary-a.flac',
        'shared.flac',
        'fish-s2.flac',
        'emotion.flac',
        'primary-b.flac',
        'absolute.flac',
        'narrator.flac',
        'primary-c.flac',
        'higgs-v3.flac',
    ]
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
