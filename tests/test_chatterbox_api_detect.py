from importlib import metadata
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from tts_audiobook_tool.tts_models import chatterbox_api_detect as detect_module
from tts_audiobook_tool.tts_models.chatterbox_api_detect import ChatterboxApiDetect


COMPATIBLE_SOURCE = '''
MULTILINGUAL_T3_MODELS = {
    "v3": "t3_mtl23ls_v3.safetensors",
}


def _resolve_multilingual_t3_model(t3_model):
    return MULTILINGUAL_T3_MODELS[t3_model]


class ChatterboxMultilingualTTS:
    @classmethod
    def from_pretrained(cls, device, t3_model=None):
        pass
'''


@pytest.mark.parametrize(
    "source",
    [
        # Legacy loader lacks the checkpoint selector.
        '''
def _resolve_multilingual_t3_model(t3_model):
    pass

V3_MODEL = "t3_mtl23ls_v3.safetensors"

class ChatterboxMultilingualTTS:
    def from_pretrained(cls, device):
        pass
''',
        # A similarly named method on another class is not sufficient.
        '''
def _resolve_multilingual_t3_model(t3_model):
    pass

V3_MODEL = "t3_mtl23ls_v3.safetensors"

class OtherModel:
    def from_pretrained(cls, device, t3_model=None):
        pass

class ChatterboxMultilingualTTS:
    pass
''',
        COMPATIBLE_SOURCE.replace(
            "def _resolve_multilingual_t3_model(t3_model):",
            "def resolve_model(t3_model):",
        ),
        COMPATIBLE_SOURCE.replace(
            '"t3_mtl23ls_v3.safetensors"',
            '"t3_mtl23ls_v2.safetensors"',
        ),
        "not valid Python !!!",
    ],
)
def test_source_probe_rejects_missing_v3_features(source):
    assert not ChatterboxApiDetect.source_has_required_v3_features(source)


def test_source_probe_accepts_required_v3_feature_cluster():
    assert ChatterboxApiDetect.source_has_required_v3_features(COMPATIBLE_SOURCE)


def test_installed_package_probe_reads_source_without_importing_package(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "mtl_tts.py"
    source_path.write_text(COMPATIBLE_SOURCE, encoding="utf-8")
    source_entry = PurePosixPath(ChatterboxApiDetect.MULTILINGUAL_MODULE_PATH)
    fake_distribution = SimpleNamespace(
        files=[source_entry],
        locate_file=lambda entry: source_path,
    )
    monkeypatch.setattr(
        detect_module.metadata,
        "distribution",
        lambda name: fake_distribution,
    )

    assert ChatterboxApiDetect.has_required_v3_features()


def test_installed_package_probe_rejects_missing_distribution(monkeypatch):
    def missing_distribution(_):
        raise metadata.PackageNotFoundError(ChatterboxApiDetect.DISTRIBUTION_NAME)

    monkeypatch.setattr(
        detect_module.metadata,
        "distribution",
        missing_distribution,
    )

    assert not ChatterboxApiDetect.has_required_v3_features()


def test_installed_package_probe_rejects_missing_source_file(monkeypatch):
    monkeypatch.setattr(
        detect_module.metadata,
        "distribution",
        lambda _: SimpleNamespace(files=[]),
    )

    assert not ChatterboxApiDetect.has_required_v3_features()
