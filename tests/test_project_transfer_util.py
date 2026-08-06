from pathlib import Path
from typing import Any, cast

from tts_audiobook_tool.constants import (
    PROJECT_TEXT_EPUB_FILE_NAME,
    PROJECT_TEXT_FILE_NAME,
    PROJECT_TEXT_RAW_FILE_NAME,
)
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil


def test_make_supporting_project_file_names_collects_project_local_voice_files(tmp_path: Path) -> None:
    project = Project.model_validate({
        'chatterbox_voice_file_name': ['primary-a.flac', 'shared.flac'],
        'mira_voice_file_name': ['primary-b.flac', 'shared.flac'],
        'indextts2_emo_voice_file_name': 'emotion.flac',
        'oute_voice_file_name': 'oute-voice.json',
        'fish_s2_server_voice_target': ['server-target.flac'],
        'higgs_v3_voice_target': ['https://example.com/voice.flac'],
    })
    project.pocket_voice_file_name = cast(Any, [
        '',
        123,
        str(tmp_path / 'absolute.flac'),
        'primary-c.flac',
    ])

    result = ProjectTransferUtil.make_supporting_project_file_names(project)

    assert result == [
        PROJECT_TEXT_FILE_NAME,
        PROJECT_TEXT_RAW_FILE_NAME,
        PROJECT_TEXT_EPUB_FILE_NAME,
        'oute-voice.json',
        'primary-a.flac',
        'shared.flac',
        'emotion.flac',
        'primary-b.flac',
        'primary-c.flac',
    ]
    assert 'server-target.flac' not in result
    assert 'https://example.com/voice.flac' not in result


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
        oute_voice_file_name='oute-voice.json',
    )
    contents = {
        PROJECT_TEXT_FILE_NAME: b'project text',
        PROJECT_TEXT_RAW_FILE_NAME: b'raw text',
        PROJECT_TEXT_EPUB_FILE_NAME: b'epub text',
        'voice-a.flac': b'voice a',
        'voice-b.flac': b'voice b',
        'emotion.flac': b'emotion voice',
        'oute-voice.json': b'{"voice": "oute"}',
    }
    for file_name, content in contents.items():
        (source_dir / file_name).write_bytes(content)

    file_names = ProjectTransferUtil.make_supporting_project_file_names(source_project)
    missing_paths = ProjectTransferUtil.copy_supporting_project_files(
        Project(dir_path=str(dest_dir)),
        str(source_dir),
        file_names,
    )

    assert missing_paths == [str(source_dir / 'missing.flac')]
    for file_name, content in contents.items():
        assert (dest_dir / file_name).read_bytes() == content
    assert not (dest_dir / 'missing.flac').exists()
