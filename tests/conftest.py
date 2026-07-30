import pytest

from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@pytest.fixture(autouse=True)
def initialize_tts_type_for_tests():
    had_type = hasattr(Tts, "_type")
    original_type = getattr(Tts, "_type", None)

    if not had_type or Tts._type is None:
        setattr(Tts, "_type", TtsModelType.NONE)

    try:
        yield
    finally:
        if had_type:
            setattr(Tts, "_type", original_type)
        elif hasattr(Tts, "_type"):
            delattr(Tts, "_type")
