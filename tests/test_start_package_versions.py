import pytest

from tts_audiobook_tool import start as start_module


def _start_with(monkeypatch, *, versions: dict[str, str], minimums: dict[str, str], missing=()):
    """A Start whose package checks see the given installed versions and missing imports."""
    from importlib import metadata

    start = object.__new__(start_module.Start)
    printed = []

    def version(distribution: str) -> str:
        if distribution not in versions:
            raise metadata.PackageNotFoundError(distribution)
        return versions[distribution]

    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(start, "get_new_packages", lambda: list(missing))
    monkeypatch.setattr(start, "get_minimum_package_versions", lambda: minimums)
    monkeypatch.setattr("importlib.util.find_spec", lambda package: None)
    monkeypatch.setattr(start_module.Tts, "get_requirements_file_name", lambda: "requirements-base.txt")
    monkeypatch.setattr(start_module.hints, "print_hint", printed.append)
    return start, printed


@pytest.mark.parametrize(
    "installed, minimum, expected",
    [
        ("15.1.0", "16.1", "15.1.0"),
        ("16.0.0", "16.1", "16.0.0"),
        # Trailing zeros don't make a version newer: 16.1 == 16.1.0
        ("16.1", "16.1.0", ""),
        ("16.1.0", "16.1", ""),
        ("18.1.0", "16.1", ""),
        # A pre-release counts as its release
        ("16.1.0rc1", "16.1", ""),
    ],
)
def test_installed_version_below(monkeypatch, installed, minimum, expected):
    from importlib import metadata

    monkeypatch.setattr(metadata, "version", lambda distribution: installed)
    assert start_module.installed_version_below("av", minimum) == expected


def test_unreadable_or_absent_version_does_not_block(monkeypatch):
    # Only a known-old version blocks; absence is the missing-package check's job.
    from importlib import metadata

    def absent(distribution):
        raise metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(metadata, "version", absent)
    assert start_module.installed_version_below("av", "16.1") == ""
    monkeypatch.setattr(metadata, "version", lambda distribution: "unknown")
    assert start_module.installed_version_below("av", "16.1") == ""


def test_av_minimum_supports_m4b_chapters():
    # set_chapters() arrived in PyAV 16.0; the vibevoice range starts at 16.1
    minimums = object.__new__(start_module.Start).get_minimum_package_versions()
    assert start_module._release_numbers(minimums["av"]) >= (16, 1)


def test_outdated_package_blocks_and_says_why(monkeypatch):
    # An old venv already has av (through faster-whisper), so the message must
    # explain that the installed version is too old, not that it's missing.
    start, printed = _start_with(monkeypatch, versions={"av": "15.1.0"}, minimums={"av": "16.1"})

    with pytest.raises(SystemExit) as exc:
        start.exit_on_missing_or_outdated_packages()

    assert exc.value.code == 1
    text = printed[0].text
    assert "missing or below the required version" in text
    assert "av (installed: 15.1.0, required: 16.1 or newer)" in text
    assert "pip install -r requirements-base.txt" in text


def test_missing_and_outdated_are_listed_together(monkeypatch):
    start, printed = _start_with(
        monkeypatch, versions={"av": "14.0.0"}, minimums={"av": "16.1"}, missing=["wakepy"],
    )

    with pytest.raises(SystemExit):
        start.exit_on_missing_or_outdated_packages()

    text = printed[0].text
    assert "wakepy (not installed)" in text
    assert "av (installed: 14.0.0" in text


def test_current_versions_pass(monkeypatch):
    start, printed = _start_with(monkeypatch, versions={"av": "18.1.0"}, minimums={"av": "16.1"})

    start.exit_on_missing_or_outdated_packages()

    assert printed == []
